"""
FastAPI Serving Gateway with Semantic Adapter Routing.
Exposes a unified /v1/chat endpoint:
  1. Receives incoming user query.
  2. Routes query to specialized adapter ('sql-adapter', 'json-adapter', 'code-adapter')
     or frozen base model ('Qwen/Qwen2.5-1.5B-Instruct') via src.router.
  3. Forwards request to local vLLM multi-LoRA server (port 8000).
  4. Returns generated completion + adapter used + routing latency.

Runs on port 8080 (leaving vLLM private on port 8000).
Supports optional public exposure via pyngrok.
"""

import os
import sys
import time
import argparse
from typing import Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.router import get_router

# Configuration
VLLM_HOST = os.environ.get("VLLM_HOST", "http://localhost:8000")
VLLM_COMPLETIONS_URL = f"{VLLM_HOST}/v1/chat/completions"
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

ADAPTER_MAP = {
    "sql": "sql-adapter",
    "json": "json-adapter",
    "code": "code-adapter",
    "base": BASE_MODEL_NAME,
}

app = FastAPI(
    title="Routed Multi-Adapter LLM Serving System",
    description="Intelligent semantic routing gateway across specialized LoRA adapters on top of Qwen2.5-1.5B.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Schemas
class QueryRequest(BaseModel):
    prompt: str = Field(..., description="User prompt or instruction")
    max_tokens: int = Field(256, description="Maximum tokens to generate")
    temperature: float = Field(0.0, description="Sampling temperature")
    force_adapter: Optional[str] = Field(None, description="Optional override: 'sql', 'json', 'code', or 'base'")

class QueryResponse(BaseModel):
    response: str
    adapter_used: str
    model_identifier: str
    routing_latency_ms: float
    total_latency_ms: float
    router_confidence: float

# Global mock mode flag for testing without active GPU vLLM server
MOCK_VLLM = os.environ.get("MOCK_VLLM", "false").lower() == "true"

def mock_generate_response(model_name: str, prompt: str) -> str:
    """Mock generator for local testing when vLLM GPU engine is not running."""
    if "sql" in model_name:
        return "SELECT count(*) FROM table WHERE condition = 1;"
    elif "json" in model_name:
        return '{"user": "Demo User", "order_id": "ORD-12345", "amount": 99.99}'
    elif "code" in model_name:
        return "def solution():\n    return 'Hello from Code LoRA'"
    else:
        return f"Zero-shot base model response to: {prompt[:40]}..."

@app.get("/health")
def health_check():
    """Health check endpoint reporting gateway status and router state."""
    vllm_online = False
    try:
        r = requests.get(f"{VLLM_HOST}/health", timeout=2.0)
        vllm_online = r.status_code == 200
    except Exception:
        vllm_online = False

    return {
        "status": "healthy",
        "gateway_port": 8080,
        "vllm_online": vllm_online or MOCK_VLLM,
        "mock_mode": MOCK_VLLM,
        "registered_adapters": list(ADAPTER_MAP.keys()),
        "base_model": BASE_MODEL_NAME,
    }

@app.post("/v1/chat", response_model=QueryResponse)
def generate(req: QueryRequest):
    """Main routing and inference endpoint."""
    t_start = time.perf_counter()

    # 1. Route query (or use force_adapter override)
    router = get_router()
    if req.force_adapter and req.force_adapter in ADAPTER_MAP:
        route = req.force_adapter
        confidence = 1.0
        routing_latency_ms = 0.0
    else:
        route_info = router.route_detailed(req.prompt)
        route = route_info["route"]
        confidence = route_info["confidence"]
        routing_latency_ms = route_info["latency_ms"]

    model_name = ADAPTER_MAP.get(route, BASE_MODEL_NAME)

    # 2. Forward request to vLLM (or mock if enabled)
    if MOCK_VLLM:
        time.sleep(0.05)  # Simulate small generation latency
        content = mock_generate_response(model_name, req.prompt)
    else:
        try:
            vllm_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": req.prompt}],
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
            }
            resp = requests.post(VLLM_COMPLETIONS_URL, json=vllm_payload, timeout=60.0)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"vLLM server error: {resp.text}",
                )
            content = resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail=f"vLLM backend unreachable at {VLLM_COMPLETIONS_URL}. Ensure vLLM is running or set MOCK_VLLM=true.",
            )

    total_latency_ms = (time.perf_counter() - t_start) * 1000

    return QueryResponse(
        response=content,
        adapter_used=route,
        model_identifier=model_name,
        routing_latency_ms=round(routing_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        router_confidence=round(confidence, 4),
    )

def start_gateway(port: int = 8080, host: str = "0.0.0.0", use_ngrok: bool = False, ngrok_token: Optional[str] = None):
    if use_ngrok:
        try:
            from pyngrok import ngrok
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)
            tunnel = ngrok.connect(port)
            print("\n" + "=" * 65)
            print(f"PUBLIC NGROK GATEWAY TUNNEL: {tunnel.public_url}")
            print(f"API Endpoint: {tunnel.public_url}/v1/chat")
            print(f"Swagger Docs: {tunnel.public_url}/docs")
            print("=" * 65 + "\n")
        except ImportError:
            print("WARNING: pyngrok is not installed. Running gateway locally only.")
        except Exception as e:
            print(f"WARNING: Could not establish ngrok tunnel: {e}")

    print(f"Starting Gateway on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Routed Multi-Adapter Serving Gateway")
    parser.add_argument("--port", type=int, default=8080, help="Gateway port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gateway host")
    parser.add_argument("--ngrok", action="store_true", help="Enable public pyngrok tunnel")
    parser.add_argument("--ngrok-token", type=str, default=None, help="Optional ngrok authtoken")
    parser.add_argument("--mock-vllm", action="store_true", help="Enable mock mode for testing without GPU")
    args = parser.parse_args()

    if args.mock_vllm:
        MOCK_VLLM = True

    start_gateway(
        port=args.port,
        host=args.host,
        use_ngrok=args.ngrok,
        ngrok_token=args.ngrok_token,
    )

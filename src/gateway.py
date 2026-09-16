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
GATEWAY_ENGINE = os.environ.get("GATEWAY_ENGINE", "peft").lower()

ADAPTER_MAP = {
    "sql": "sql-adapter",
    "json": "json-adapter",
    "code": "code-adapter",
    "base": BASE_MODEL_NAME,
}

ADAPTER_PATHS = {
    "sql": "adapters/sql_lora",
    "json": "adapters/json_lora",
    "code": "adapters/code_lora",
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

# Native PEFT Engine State
_peft_model = None
_peft_base_model = None
_peft_tokenizer = None

def init_peft_engine():
    """Initializes and mounts 4-bit base model and all 3 LoRA adapters into unified VRAM."""
    global _peft_model, _peft_base_model, _peft_tokenizer
    if _peft_model is not None and _peft_base_model is not None:
        return _peft_model, _peft_base_model, _peft_tokenizer

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    print("\n" + "=" * 65, flush=True)
    print("INITIALIZING NATIVE PEFT MULTI-ADAPTER ENGINE (RTX 4050 GPU)", flush=True)
    print("=" * 65, flush=True)

    print("1. Loading Tokenizer...", flush=True)
    _peft_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if _peft_tokenizer.pad_token is None:
        _peft_tokenizer.pad_token = _peft_tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    target_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"2. Loading Base Model ({BASE_MODEL_NAME}) in 4-bit VRAM...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=target_dtype,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config if has_cuda else None,
        torch_dtype=target_dtype if has_cuda else torch.float32,
        device_map="auto" if has_cuda else None,
        trust_remote_code=True,
    )

    print("3. Registering LoRA Adapters into Unified VRAM...", flush=True)
    model = None
    first = True
    for name, path in ADAPTER_PATHS.items():
        if os.path.exists(path):
            if first:
                print(f"   [+] Registering '{name}' adapter from {path}", flush=True)
                model = PeftModel.from_pretrained(base_model, path, adapter_name=name)
                first = False
            else:
                print(f"   [+] Registering '{name}' adapter from {path}", flush=True)
                model.load_adapter(path, adapter_name=name)

    _peft_base_model = base_model
    _peft_model = model if model is not None else base_model
    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if has_cuda else 0
    print(f"Native PEFT Engine ONLINE! GPU VRAM Allocated: {vram_mb:.1f} MB", flush=True)
    print("=" * 65 + "\n", flush=True)
    return _peft_model, _peft_base_model, _peft_tokenizer

def peft_generate_response(route: str, prompt: str, max_tokens: int = 256) -> str:
    """Generates completion using active LoRA adapter on local GPU."""
    import torch
    model, base_model, tokenizer = init_peft_engine()

    # Hot-swap adapter in VRAM (0 ms) or use base model directly
    if route in ADAPTER_PATHS and model is not None and hasattr(model, "set_adapter"):
        model.set_adapter(route)
        gen_model = model
    else:
        # Pure base model fallback for general / out-of-domain queries
        gen_model = base_model

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def mock_generate_response(model_name: str, prompt: str) -> str:
    """Mock generator for quick testing without GPU inference."""
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
    """Health check endpoint reporting gateway status and active engine."""
    import torch
    has_cuda = torch.cuda.is_available()
    vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 1) if has_cuda else 0.0

    return {
        "status": "healthy",
        "gateway_port": 8080,
        "engine": GATEWAY_ENGINE,
        "gpu_device": torch.cuda.get_device_name(0) if has_cuda else "CPU",
        "vram_allocated_mb": vram_mb,
        "registered_adapters": list(ADAPTER_MAP.keys()),
        "base_model": BASE_MODEL_NAME,
    }

@app.get("/v1/models")
def list_models():
    """OpenAI-compatible models listing endpoint."""
    return {
        "object": "list",
        "data": [
            {"id": "sql-adapter", "object": "model", "owned_by": "custom-lora"},
            {"id": "json-adapter", "object": "model", "owned_by": "custom-lora"},
            {"id": "code-adapter", "object": "model", "owned_by": "custom-lora"},
            {"id": BASE_MODEL_NAME, "object": "model", "owned_by": "base"},
        ]
    }

@app.post("/v1/chat", response_model=QueryResponse)
def generate(req: QueryRequest):
    """Main routing and dynamic inference endpoint."""
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

    # 2. Execute Generation via configured Engine
    if GATEWAY_ENGINE == "peft":
        content = peft_generate_response(route, req.prompt, req.max_tokens)
    elif GATEWAY_ENGINE == "mock":
        time.sleep(0.05)
        content = mock_generate_response(model_name, req.prompt)
    else:
        # Forward to remote/WSL vLLM server
        try:
            vllm_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": req.prompt}],
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
            }
            resp = requests.post(VLLM_COMPLETIONS_URL, json=vllm_payload, timeout=60.0)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"vLLM error: {resp.text}")
            content = resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            print(f"[GATEWAY NOTICE] vLLM offline. Falling back to local PEFT generator.")
            content = peft_generate_response(route, req.prompt, req.max_tokens)

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

    # Pre-warm PEFT engine if active
    if GATEWAY_ENGINE == "peft":
        init_peft_engine()

    print(f"Starting Gateway ({GATEWAY_ENGINE.upper()} Engine) on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Routed Multi-Adapter Serving Gateway")
    parser.add_argument("--port", type=int, default=8080, help="Gateway port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gateway host")
    parser.add_argument("--engine", type=str, choices=["peft", "vllm", "mock"], default="peft",
                        help="Backend serving engine: 'peft' (native GPU), 'vllm' (port 8000), or 'mock'")
    parser.add_argument("--ngrok", action="store_true", help="Enable public pyngrok tunnel")
    parser.add_argument("--ngrok-token", type=str, default=None, help="Optional ngrok authtoken")
    parser.add_argument("--mock-vllm", action="store_true", help="Shortcut for --engine mock")
    args = parser.parse_args()

    if args.mock_vllm:
        GATEWAY_ENGINE = "mock"
    else:
        GATEWAY_ENGINE = args.engine

    os.environ["GATEWAY_ENGINE"] = GATEWAY_ENGINE

    start_gateway(
        port=args.port,
        host=args.host,
        use_ngrok=args.ngrok,
        ngrok_token=args.ngrok_token,
    )

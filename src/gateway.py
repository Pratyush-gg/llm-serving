import os
import sys
import time
import json
import argparse
from typing import Dict, List, Optional, Any

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.router import get_router
from src.cascade import CascadeRouter

# Configuration
VLLM_HOST = os.environ.get("VLLM_HOST", "http://localhost:8000")
VLLM_COMPLETIONS_URL = f"{VLLM_HOST}/v1/chat/completions"
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
GATEWAY_ENGINE = os.environ.get("GATEWAY_ENGINE", "peft").lower()
DEFAULT_ROUTER_STRATEGY = os.environ.get("ROUTER_STRATEGY", "auto").lower()
ENABLE_CASCADE = os.environ.get("ENABLE_CASCADE", "false").lower() in ["true", "1", "yes"]

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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboard")

app = FastAPI(
    title="Routed Multi-Adapter LLM Serving System",
    description="Intelligent semantic & learned routing gateway across specialized LoRA adapters on top of Qwen2.5-1.5B.",
    version="1.1.0",
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
    enable_cascade: Optional[bool] = Field(None, description="Optional toggle for confidence-based cascade")
    router_strategy: Optional[str] = Field(None, description="Routing strategy: 'centroid', 'learned', or 'auto'")

class QueryResponse(BaseModel):
    response: str
    adapter_used: str
    model_identifier: str
    routing_latency_ms: float
    total_latency_ms: float
    router_confidence: float
    router_strategy: str = "centroid"
    cascade_triggered: bool = False
    selection_reason: Optional[str] = None
    candidates_evaluated: Optional[List[Dict[str, Any]]] = None

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
        full_path = os.path.join(REPO_ROOT, path)
        if os.path.exists(full_path):
            if first:
                print(f"   [+] Registering '{name}' adapter from {full_path}", flush=True)
                model = PeftModel.from_pretrained(base_model, full_path, adapter_name=name)
                first = False
            else:
                print(f"   [+] Registering '{name}' adapter from {full_path}", flush=True)
                model.load_adapter(full_path, adapter_name=name)

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

    if route in ADAPTER_PATHS and model is not None and hasattr(model, "set_adapter"):
        model.set_adapter(route)
        gen_model = model
    else:
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
    """Mock generator for instantaneous testing and live dashboard demo without GPU."""
    p_lower = prompt.lower()
    if "sql" in model_name:
        # Match table or columns from prompt if available
        return "SELECT id, created_at, status, count(*) FROM database_records WHERE active = 1 GROUP BY status ORDER BY created_at DESC;"
    elif "json" in model_name:
        return '{\n  "user": "Sarah Jenkins",\n  "order_id": "TXN-98421",\n  "amount": 149.50\n}'
    elif "code" in model_name:
        return 'def solution(input_data: list) -> list:\n    """Process and return optimized result."""\n    return [item for item in input_data if item is not None]'
    else:
        return f"This response is generated by the shared frozen base model ({BASE_MODEL_NAME}). It provides general factual and reasoning capabilities across arbitrary domains."

def execute_engine_generation(route: str, prompt: str, max_tokens: int, temperature: float) -> str:
    """Dispatches generation request to active backend engine."""
    model_name = ADAPTER_MAP.get(route, BASE_MODEL_NAME)
    if GATEWAY_ENGINE == "peft":
        return peft_generate_response(route, prompt, max_tokens)
    elif GATEWAY_ENGINE == "mock":
        time.sleep(0.04)  # Small realistic latency simulation
        return mock_generate_response(model_name, prompt)
    else:
        # Forward to vLLM server
        try:
            vllm_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            resp = requests.post(VLLM_COMPLETIONS_URL, json=vllm_payload, timeout=60.0)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"vLLM error: {resp.text}")
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            print(f"[GATEWAY NOTICE] vLLM offline. Falling back to local PEFT generator.")
            return peft_generate_response(route, prompt, max_tokens)

# Endpoints
@app.get("/health")
def health_check():
    """Health check endpoint reporting gateway status and active engine."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 1) if has_cuda else 0.0
        device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    except Exception:
        has_cuda = False
        vram_mb = 0.0
        device_name = "CPU"

    return {
        "status": "healthy",
        "gateway_port": 8080,
        "engine": GATEWAY_ENGINE,
        "gpu_device": device_name,
        "vram_allocated_mb": vram_mb,
        "registered_adapters": list(ADAPTER_MAP.keys()),
        "base_model": BASE_MODEL_NAME,
        "router_strategy": DEFAULT_ROUTER_STRATEGY,
        "cascade_enabled": ENABLE_CASCADE,
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

@app.get("/v1/router/scores")
def get_router_scores(prompt: str = Query(..., description="Query prompt to route"), strategy: Optional[str] = None):
    """
    Computes and returns real-time routing scores across all 4 domains without running token generation.
    Ideal for dynamic radar charts and live router telemetry.
    """
    strat = strategy or DEFAULT_ROUTER_STRATEGY
    router = get_router(strategy=strat)
    info = router.route_detailed(prompt)
    return {
        "prompt": prompt,
        "strategy": getattr(router, "strategy_name", strat),
        "route": info["route"],
        "confidence": info["confidence"],
        "scores": info.get("scores", {}),
        "latency_ms": info["latency_ms"],
    }

@app.get("/dashboard/data/{filename}")
def get_benchmark_data(filename: str):
    """Serves empirical benchmark JSON datasets directly to the dashboard."""
    safe_name = os.path.basename(filename)
    if not safe_name.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON benchmark files are accessible.")
    filepath = os.path.join(RESULTS_DIR, safe_name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Benchmark file '{safe_name}' not found.")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/v1/chat", response_model=QueryResponse)
def generate(req: QueryRequest):
    """Main routing, cascade, and dynamic inference endpoint."""
    t_start = time.perf_counter()
    strat = req.router_strategy or DEFAULT_ROUTER_STRATEGY
    router = get_router(strategy=strat)
    strategy_name = getattr(router, "strategy_name", strat)

    use_cascade = req.enable_cascade if req.enable_cascade is not None else ENABLE_CASCADE

    if use_cascade:
        # Confidence-based cascade path
        cascade_router = CascadeRouter(base_router=router, cascade_threshold=0.70)
        cascade_res = cascade_router.route_and_generate(
            prompt=req.prompt,
            generate_fn=lambda r, p: execute_engine_generation(r, p, req.max_tokens, req.temperature),
            force_adapter=req.force_adapter,
        )
        total_lat = (time.perf_counter() - t_start) * 1000
        return QueryResponse(
            response=cascade_res.response,
            adapter_used=cascade_res.final_adapter,
            model_identifier=ADAPTER_MAP.get(cascade_res.final_adapter, BASE_MODEL_NAME),
            routing_latency_ms=cascade_res.routing_latency_ms,
            total_latency_ms=round(total_lat, 2),
            router_confidence=cascade_res.router_confidence,
            router_strategy=strategy_name,
            cascade_triggered=cascade_res.cascade_triggered,
            selection_reason=cascade_res.selection_reason,
            candidates_evaluated=cascade_res.candidates_evaluated,
        )

    # Standard direct routing path
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
    content = execute_engine_generation(route, req.prompt, req.max_tokens, req.temperature)
    total_latency_ms = (time.perf_counter() - t_start) * 1000

    return QueryResponse(
        response=content,
        adapter_used=route,
        model_identifier=model_name,
        routing_latency_ms=round(routing_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        router_confidence=round(confidence, 4),
        router_strategy=strategy_name,
        cascade_triggered=False,
        selection_reason=f"Directly routed via {strategy_name} router (confidence: {confidence:.3f}).",
        candidates_evaluated=[{
            "adapter": route,
            "router_score": round(confidence, 4),
            "quality_score": 1.0,
            "combined_score": round(confidence, 4),
            "response_snippet": content[:80].replace("\n", " "),
            "validation_notes": "Single direct execution without cascade",
        }],
    )

# Mount Web Dashboard
os.makedirs(DASHBOARD_DIR, exist_ok=True)
app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")

@app.get("/")
def root_redirect():
    """Redirect root access to interactive dashboard."""
    return RedirectResponse(url="/dashboard/")

def start_gateway(
    port: int = 8080,
    host: str = "0.0.0.0",
    use_ngrok: bool = False,
    ngrok_token: Optional[str] = None,
    engine: str = "peft",
    router_strategy: str = "auto",
    cascade: bool = False,
):
    global GATEWAY_ENGINE, DEFAULT_ROUTER_STRATEGY, ENABLE_CASCADE
    GATEWAY_ENGINE = engine
    DEFAULT_ROUTER_STRATEGY = router_strategy
    ENABLE_CASCADE = cascade

    os.environ["GATEWAY_ENGINE"] = GATEWAY_ENGINE
    os.environ["ROUTER_STRATEGY"] = DEFAULT_ROUTER_STRATEGY
    os.environ["ENABLE_CASCADE"] = "true" if cascade else "false"

    if use_ngrok:
        try:
            from pyngrok import ngrok
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)
            tunnel = ngrok.connect(port)
            print("\n" + "=" * 65)
            print(f"PUBLIC NGROK GATEWAY TUNNEL : {tunnel.public_url}")
            print(f"Interactive Dashboard       : {tunnel.public_url}/dashboard/")
            print(f"API Endpoint                : {tunnel.public_url}/v1/chat")
            print(f"Swagger Docs                : {tunnel.public_url}/docs")
            print("=" * 65 + "\n")
        except ImportError:
            print("WARNING: pyngrok is not installed. Running gateway locally only.")
        except Exception as e:
            print(f"WARNING: Could not establish ngrok tunnel: {e}")

    # Pre-warm PEFT engine if active
    if GATEWAY_ENGINE == "peft":
        init_peft_engine()

    print("\n" + "=" * 65)
    print(f"ROUTED MULTI-ADAPTER SERVING GATEWAY v1.1")
    print(f"Engine          : {GATEWAY_ENGINE.upper()}")
    print(f"Router Strategy : {DEFAULT_ROUTER_STRATEGY.upper()}")
    print(f"Cascade Mode    : {'ENABLED' if ENABLE_CASCADE else 'DISABLED'}")
    print(f"Dashboard URL   : http://localhost:{port}/dashboard/")
    print(f"API Endpoint    : http://localhost:{port}/v1/chat")
    print("=" * 65 + "\n")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Routed Multi-Adapter Serving Gateway")
    parser.add_argument("--port", type=int, default=8080, help="Gateway port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gateway host")
    parser.add_argument("--engine", type=str, choices=["peft", "vllm", "mock"], default="peft",
                        help="Backend engine: 'peft' (local GPU), 'vllm' (port 8000), or 'mock'")
    parser.add_argument("--router-strategy", type=str, choices=["centroid", "learned", "auto"], default="auto",
                        help="Default routing strategy (default: 'auto')")
    parser.add_argument("--cascade", action="store_true", help="Enable confidence-based cascade routing")
    parser.add_argument("--ngrok", action="store_true", help="Enable public pyngrok tunnel")
    parser.add_argument("--ngrok-token", type=str, default=None, help="Optional ngrok authtoken")
    parser.add_argument("--mock-vllm", action="store_true", help="Shortcut for --engine mock")
    args = parser.parse_args()

    eng = "mock" if args.mock_vllm else args.engine

    start_gateway(
        port=args.port,
        host=args.host,
        use_ngrok=args.ngrok,
        ngrok_token=args.ngrok_token,
        engine=eng,
        router_strategy=args.router_strategy,
        cascade=args.cascade,
    )

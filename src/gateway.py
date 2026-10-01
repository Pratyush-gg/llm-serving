import os
import sys
import time
import json
import argparse
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Literal

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.local_cache import local_files_only  # sets cache env vars before ML imports
from src.router import get_router
from src.cascade import CascadeRouter

# Configuration
VLLM_HOST = os.environ.get("VLLM_HOST", "http://localhost:8000")
VLLM_COMPLETIONS_URL = f"{VLLM_HOST}/v1/chat/completions"
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
GATEWAY_ENGINE = os.environ.get("GATEWAY_ENGINE", "peft").lower()
DEFAULT_ROUTER_STRATEGY = os.environ.get("ROUTER_STRATEGY", "auto").lower()
ENABLE_CASCADE = os.environ.get("ENABLE_CASCADE", "false").lower() in ["true", "1", "yes"]
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8080"))
# Cascade thresholds calibrated for the v2 router (eval/calibrate_cascade.py).
CASCADE_THRESHOLD = float(os.environ.get("CASCADE_THRESHOLD", "1.0"))
CASCADE_MARGIN = float(os.environ.get("CASCADE_MARGIN", "0.0"))
MAX_TOKENS_LIMIT = 2048

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

# Wildcard origins are only valid without credentials (the API uses no cookies/auth).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Schemas
class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt or instruction")
    max_tokens: int = Field(256, ge=1, le=MAX_TOKENS_LIMIT, description="Maximum tokens to generate")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature (0 = greedy)")
    force_adapter: Optional[Literal["sql", "json", "code", "base"]] = Field(None, description="Optional override: 'sql', 'json', 'code', or 'base'")
    enable_cascade: Optional[bool] = Field(None, description="Optional toggle for confidence-based cascade")
    router_strategy: Optional[Literal["centroid", "learned", "auto"]] = Field(None, description="Routing strategy: 'centroid', 'learned', or 'auto'")

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
    # Per-stage timings (summed over candidates when the cascade runs).
    generation_latency_ms: Optional[float] = Field(None, description="Time spent in token generation (excludes routing and queueing)")
    adapter_switch_ms: Optional[float] = Field(None, description="Time to activate the adapter (PEFT engine only; null when not measurable)")
    prompt_tokens: Optional[int] = Field(None, description="Prompt length in tokens (null when the engine does not report it)")
    completion_tokens: Optional[int] = Field(None, description="Generated tokens (null when the engine does not report it)")
    tokens_per_second: Optional[float] = Field(None, description="completion_tokens / generation time")


@dataclass
class GenerationResult:
    text: str
    generation_ms: float
    adapter_switch_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


def summarize_generation(results: List["GenerationResult"]) -> Dict[str, Any]:
    """Aggregate timing fields over one or more generations (cascade may generate several)."""
    if not results:
        return {}
    gen_ms = sum(r.generation_ms for r in results)

    def total(attr):
        values = [getattr(r, attr) for r in results]
        return None if any(v is None for v in values) else sum(values)

    completion = total("completion_tokens")
    switch = total("adapter_switch_ms")
    return {
        "generation_latency_ms": round(gen_ms, 2),
        "adapter_switch_ms": round(switch, 3) if switch is not None else None,
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": completion,
        "tokens_per_second": round(completion / (gen_ms / 1000), 2) if completion and gen_ms > 0 else None,
    }

# Native PEFT Engine State
_peft_model = None
_peft_base_model = None
_peft_tokenizer = None

# One shared active adapter + threaded requests: serialize adapter switch and generation.
_peft_lock = threading.RLock()

def init_peft_engine():
    """Initializes and mounts 4-bit base model and all 3 LoRA adapters into unified VRAM."""
    global _peft_model, _peft_base_model, _peft_tokenizer
    with _peft_lock:
        if _peft_model is not None and _peft_base_model is not None:
            return _peft_model, _peft_base_model, _peft_tokenizer
        return _load_peft_engine()

def load_base_model():
    """Load tokenizer and the 4-bit NF4 base model (fp32 on CPU when no GPU is available)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, trust_remote_code=True, local_files_only=local_files_only(BASE_MODEL_NAME)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.get_device_capability()[0] >= 8  # native bf16 only (not emulated on T4)
    target_dtype = torch.bfloat16 if use_bf16 else torch.float16

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
        local_files_only=local_files_only(BASE_MODEL_NAME),
    )
    return base_model, tokenizer


def attach_adapters(base_model):
    """Register every available LoRA adapter on the base model. Returns the PeftModel (or base if none)."""
    from peft import PeftModel

    model = None
    for name, path in ADAPTER_PATHS.items():
        full_path = os.path.join(REPO_ROOT, path)
        if not os.path.exists(full_path):
            continue
        print(f"   [+] Registering '{name}' adapter from {full_path}", flush=True)
        if model is None:
            model = PeftModel.from_pretrained(base_model, full_path, adapter_name=name)
        else:
            model.load_adapter(full_path, adapter_name=name)
    return model if model is not None else base_model


def _load_peft_engine():
    global _peft_model, _peft_base_model, _peft_tokenizer
    import torch

    print("\n" + "=" * 65, flush=True)
    print("INITIALIZING NATIVE PEFT MULTI-ADAPTER ENGINE", flush=True)
    print("=" * 65, flush=True)

    print(f"1. Loading tokenizer and base model ({BASE_MODEL_NAME}, 4-bit NF4)...", flush=True)
    base_model, _peft_tokenizer = load_base_model()

    print("2. Registering LoRA adapters...", flush=True)
    _peft_model = attach_adapters(base_model)
    _peft_base_model = base_model

    has_cuda = torch.cuda.is_available()
    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if has_cuda else 0
    print(f"Native PEFT Engine ONLINE! GPU VRAM Allocated: {vram_mb:.1f} MB", flush=True)
    print("=" * 65 + "\n", flush=True)
    return _peft_model, _peft_base_model, _peft_tokenizer

def peft_generate_response(route: str, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerationResult:
    """Generates completion using active LoRA adapter on local GPU."""
    import contextlib
    import torch
    model, base_model, tokenizer = init_peft_engine()
    sampling_kwargs = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
    has_cuda = torch.cuda.is_available()

    with _peft_lock:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda" if has_cuda else "cpu")
        prompt_len = inputs.input_ids.shape[1]

        t_switch = time.perf_counter()
        is_peft = hasattr(model, "set_adapter")
        if route in ADAPTER_PATHS and is_peft and route in model.peft_config:
            model.set_adapter(route)
            adapter_ctx = contextlib.nullcontext()
        elif is_peft:
            # LoRA layers live inside the base model; disable them to get the plain base model.
            adapter_ctx = model.disable_adapter()
        else:
            adapter_ctx = contextlib.nullcontext()

        with adapter_ctx, torch.no_grad():
            switch_ms = (time.perf_counter() - t_switch) * 1000
            t_gen = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **sampling_kwargs,
            )
            if has_cuda:
                torch.cuda.synchronize()
            gen_ms = (time.perf_counter() - t_gen) * 1000

        new_tokens = outputs[0][prompt_len:]
        return GenerationResult(
            text=tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            generation_ms=gen_ms,
            adapter_switch_ms=switch_ms,
            prompt_tokens=int(prompt_len),
            completion_tokens=len(new_tokens),
        )

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

def execute_engine_generation(route: str, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
    """Dispatches generation request to active backend engine."""
    model_name = ADAPTER_MAP.get(route, BASE_MODEL_NAME)
    if GATEWAY_ENGINE == "peft":
        return peft_generate_response(route, prompt, max_tokens, temperature)
    elif GATEWAY_ENGINE == "mock":
        t0 = time.perf_counter()
        time.sleep(0.04)  # Small realistic latency simulation
        text = mock_generate_response(model_name, prompt)
        return GenerationResult(text=text, generation_ms=(time.perf_counter() - t0) * 1000)
    else:
        # Forward to vLLM server
        vllm_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(VLLM_COMPLETIONS_URL, json=vllm_payload, timeout=60.0)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise HTTPException(
                status_code=503,
                detail=f"vLLM backend unavailable at {VLLM_HOST} ({type(e).__name__}).",
            )
        gen_ms = (time.perf_counter() - t0) * 1000  # includes the HTTP hop to vLLM
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"vLLM error ({resp.status_code}): {resp.text}")
        data = resp.json()
        usage = data.get("usage") or {}
        return GenerationResult(
            text=data["choices"][0]["message"]["content"],
            generation_ms=gen_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

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
        "gateway_port": GATEWAY_PORT,
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
def get_router_scores(
    prompt: str = Query(..., min_length=1, description="Query prompt to route"),
    strategy: Optional[Literal["centroid", "learned", "auto"]] = None,
):
    """Per-adapter router scores without generation."""
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
        cascade_router = CascadeRouter(base_router=router, cascade_threshold=CASCADE_THRESHOLD,
                                       margin_threshold=CASCADE_MARGIN)
        generations: List[GenerationResult] = []

        def generate_and_record(route: str, prompt: str) -> str:
            result = execute_engine_generation(route, prompt, req.max_tokens, req.temperature)
            generations.append(result)
            return result.text

        cascade_res = cascade_router.route_and_generate(
            prompt=req.prompt,
            generate_fn=generate_and_record,
            force_adapter=req.force_adapter,
        )
        total_lat = (time.perf_counter() - t_start) * 1000
        return QueryResponse(
            **summarize_generation(generations),
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
    generation = execute_engine_generation(route, req.prompt, req.max_tokens, req.temperature)
    content = generation.text
    total_latency_ms = (time.perf_counter() - t_start) * 1000

    return QueryResponse(
        **summarize_generation([generation]),
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
    cascade_threshold: Optional[float] = None,
    cascade_margin: Optional[float] = None,
):
    global GATEWAY_ENGINE, DEFAULT_ROUTER_STRATEGY, ENABLE_CASCADE, GATEWAY_PORT
    global CASCADE_THRESHOLD, CASCADE_MARGIN
    GATEWAY_ENGINE = engine
    DEFAULT_ROUTER_STRATEGY = router_strategy
    ENABLE_CASCADE = cascade
    GATEWAY_PORT = port
    if cascade_threshold is not None:
        CASCADE_THRESHOLD = cascade_threshold
    if cascade_margin is not None:
        CASCADE_MARGIN = cascade_margin

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
    print(f"Cascade Mode    : {'ENABLED' if ENABLE_CASCADE else 'DISABLED'} "
          f"(threshold={CASCADE_THRESHOLD}, margin={CASCADE_MARGIN})")
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
    parser.add_argument("--cascade-threshold", type=float, default=None,
                        help="Cascade confidence threshold (default 1.0, calibrated for the v2 router; see eval/calibrate_cascade.py)")
    parser.add_argument("--cascade-margin", type=float, default=None,
                        help="Cascade top-2 margin threshold (default 0.0)")
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
        cascade_threshold=args.cascade_threshold,
        cascade_margin=args.cascade_margin,
    )

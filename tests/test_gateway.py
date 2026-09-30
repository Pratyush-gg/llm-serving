"""Gateway tests. They use a fake router and a fake PEFT model: no GPU or model downloads needed."""
import contextlib
import sys
import threading
import time
import types

from fastapi.testclient import TestClient

try:
    import torch  # noqa: F401
except ImportError:
    # The PEFT code path only needs torch.no_grad / torch.cuda here; a stub keeps the
    # tests runnable in a lightweight environment without a multi-GB torch install.
    sys.modules["torch"] = types.SimpleNamespace(
        no_grad=contextlib.nullcontext,
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )

import src.gateway as gateway


class FakeRouter:
    strategy_name = "fake"

    def route_detailed(self, prompt):
        route = "sql" if "sql" in prompt.lower() else "base"
        scores = {"sql": 0.9, "json": 0.05, "code": 0.03, "base": 0.02}
        return {"route": route, "confidence": 0.9, "scores": scores, "latency_ms": 0.1}


@contextlib.contextmanager
def patched(**attrs):
    """Temporarily replace module attributes on src.gateway."""
    old = {k: getattr(gateway, k) for k in attrs}
    for k, v in attrs.items():
        setattr(gateway, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(gateway, k, v)


def client():
    return TestClient(gateway.app)


def test_chat_mock_engine_routes_and_responds():
    with patched(GATEWAY_ENGINE="mock", get_router=lambda strategy=None: FakeRouter()):
        r = client().post("/v1/chat", json={"prompt": "write sql for users"})
    assert r.status_code == 200
    body = r.json()
    assert body["adapter_used"] == "sql"
    assert body["response"].startswith("SELECT")
    assert body["generation_latency_ms"] > 0
    assert body["completion_tokens"] is None  # mock engine does not count tokens


def test_summarize_generation_sums_cascade_candidates():
    results = [
        gateway.GenerationResult("a", generation_ms=100.0, adapter_switch_ms=0.1, prompt_tokens=10, completion_tokens=20),
        gateway.GenerationResult("b", generation_ms=300.0, adapter_switch_ms=0.2, prompt_tokens=10, completion_tokens=60),
    ]
    s = gateway.summarize_generation(results)
    assert s["generation_latency_ms"] == 400.0
    assert s["completion_tokens"] == 80
    assert s["tokens_per_second"] == 200.0
    # unknown values stay unknown rather than being treated as zero
    assert gateway.summarize_generation([gateway.GenerationResult("c", generation_ms=5.0)])["completion_tokens"] is None


def test_request_validation_rejects_bad_input():
    bad_payloads = [
        {"prompt": ""},
        {"prompt": "hi", "max_tokens": 0},
        {"prompt": "hi", "max_tokens": gateway.MAX_TOKENS_LIMIT + 1},
        {"prompt": "hi", "temperature": -0.1},
        {"prompt": "hi", "force_adapter": "nonexistent"},
        {"prompt": "hi", "router_strategy": "magic"},
    ]
    with patched(GATEWAY_ENGINE="mock", get_router=lambda strategy=None: FakeRouter()):
        c = client()
        for payload in bad_payloads:
            assert c.post("/v1/chat", json=payload).status_code == 422, payload


def test_vllm_unreachable_returns_503():
    with patched(
        GATEWAY_ENGINE="vllm",
        VLLM_COMPLETIONS_URL="http://127.0.0.1:9/v1/chat/completions",
        get_router=lambda strategy=None: FakeRouter(),
    ):
        r = client().post("/v1/chat", json={"prompt": "hello"})
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"]


def test_health_reports_configured_port():
    with patched(GATEWAY_PORT=9123):
        assert client().get("/health").json()["gateway_port"] == 9123


# --- PEFT engine: adapter switching under concurrency -------------------------

class FakeInputs(dict):
    class _Ids:
        shape = (1, 0)

    input_ids = _Ids()

    def to(self, device):
        return self


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return messages[0]["content"]

    def __call__(self, text, return_tensors):
        return FakeInputs()

    def decode(self, tokens, skip_special_tokens):
        return tokens


class FakePeftModel:
    """Mimics PeftModel's single shared 'active adapter' state."""

    peft_config = {"sql": None, "json": None, "code": None}

    def __init__(self):
        self.active = None
        self.disabled = False

    def set_adapter(self, name):
        self.active = name

    @contextlib.contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False

    def generate(self, **kwargs):
        seen = "base" if self.disabled else self.active
        time.sleep(0.005)  # give other threads a chance to interfere
        now = "base" if self.disabled else self.active
        return [seen if seen == now else f"CORRUPTED({seen}->{now})"]


def test_base_route_disables_adapters():
    model = FakePeftModel()
    model.set_adapter("sql")
    with patched(_peft_model=model, _peft_base_model=object(), _peft_tokenizer=FakeTokenizer()):
        assert gateway.peft_generate_response("base", "hi").text == "base"
        result = gateway.peft_generate_response("json", "hi")
        assert result.text == "json"
        assert result.generation_ms >= 0 and result.adapter_switch_ms >= 0


def test_concurrent_requests_keep_their_own_adapter():
    model = FakePeftModel()
    routes = ["sql", "json", "code", "base"] * 6
    results = [None] * len(routes)

    def worker(i, route):
        results[i] = gateway.peft_generate_response(route, "hi").text

    with patched(_peft_model=model, _peft_base_model=object(), _peft_tokenizer=FakeTokenizer()):
        threads = [threading.Thread(target=worker, args=(i, r)) for i, r in enumerate(routes)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert results == routes

Routed Multi Adapter LLM Serving System
7 Day Cloud GPU Build Plan (Colab / Kaggle T4)
1. Framing
This is a routed multi adapter LLM serving system, not an autonomous agent. A lightweight semantic router selects which LoRA adapter should handle an incoming request, and a shared frozen base model runs the specialized adapter. The value of the project is proving two things at once with real numbers: that this approach saves memory compared to running separate fine tuned models, and that it does so without losing task accuracy. That second claim was missing from the original plan, so this version adds a full correctness evaluation suite alongside the original system benchmarks.

2. Architecture
Incoming Query ("SELECT * FROM users WHERE age > 21")
                       │
                       ▼
         Semantic Router (fastembed, ~2.5ms)
                       │
               Routes to: "sql_adapter"
                       │
                       ▼
       vLLM Multi LoRA Engine (Qwen2.5 1.5B Instruct base)
       ┌────────────────────────────────────────┐
       │  Frozen Base LLM (Always in Memory)     │
       │  Adapter: SQL      <-- Activated        │
       │  Adapter: JSON     (dormant)             │
       │  Adapter: Code     (dormant)             │
       └────────────────────────────────────────┘
                       │
                       ▼
         Streamed Output + adapter used + routing latency
3. Tech Stack (Cloud Path Only)
Compute: one NVIDIA T4, 16 GB VRAM, via Google Colab or Kaggle
Serving: vLLM with native multi LoRA support, OpenAI compatible API server
Training: transformers, peft, trl (SFTTrainer), bitsandbytes for QLoRA
Routing: fastembed with a small embedding model (BAAI/bge small en v1.5)
API layer: FastAPI, exposed publicly through pyngrok
Base model: Qwen/Qwen2.5 1.5B Instruct (fits comfortably on a T4 with room for three adapters)
4. Datasets
SQL adapter: sample of b mc2/sql create context, roughly 600 training examples plus a fixed held out set of 60 examples never used in training
JSON adapter: synthetic dataset of prompt to structured JSON pairs (fields such as user, order_id, amount), roughly 600 training examples plus 60 held out
Code adapter: sample of flytech/python codes 25k, each paired with one or two hand written assertions that check the generated function's behavior, roughly 600 training examples plus 60 held out with their own assertions
The held out sets are the backbone of the correctness evaluation in Day 3 and Day 4. They must never appear in training.

5. Correctness Evaluation Design
This is the main addition to the original plan. Each adapter gets a task appropriate correctness metric, not just a system level latency or memory number.

SQL adapter. For each held out example, build a small in memory SQLite database from the example's schema context. Run the generated SQL and the gold SQL against that database and compare the resulting rows, not the raw query text, since two different SQL strings can be equally correct. Report two numbers: the percentage of generated queries that execute without error, and the percentage whose result set exactly matches the gold query's result set.

JSON adapter. Define a pydantic schema for the expected fields. For each held out example, check whether the model's output parses as valid JSON and validates against the schema, then compare each field's value against the gold label. Report schema validity rate and per field accuracy.

Code adapter. For each held out example, run the generated function against its paired assertions inside a subprocess with a short timeout, so a hanging or malicious generation cannot stall the eval run. Report pass at one, the fraction of generated functions that pass all of their assertions.

Router. Build a held out prompt set that mixes real SQL, JSON, and code prompts with a handful of ambiguous or unrelated prompts that should fall back to the base model with no adapter. Report a confusion matrix across the four outcomes (sql, json, code, fallback).

Baseline comparison. Run the same three held out sets and the same eval scripts against the frozen base model with no adapter attached at all. This baseline number is what makes the final memory and quality claim credible: the adapters must match or beat the zero shot baseline on their own task while adding only tens of megabytes each, not gigabytes.

6. Seven Day Timeline
Day	Goal	Deliverable
1	Data preparation and eval design	Three training JSONL files, three held out JSONL files, and a short eval_spec.md describing every metric above
2	LoRA training on Colab T4	Three QLoRA adapters, rank 16, one to two epochs, each under 20 MB
3	Build the correctness evaluation suite	eval/sql_eval.py, eval/json_eval.py, eval/code_eval.py, and a first correctness results table for the tuned adapters
4	Baseline comparison and router	Zero shot baseline correctness numbers using the same eval scripts, plus the semantic router with its own latency and confusion matrix results
5	vLLM engine and FastAPI wrapper	A running vLLM server with all three LoRA modules registered, a FastAPI endpoint that routes and forwards requests, exposed through ngrok, with a manual smoke test log
6	Full benchmarking and combined report	Memory conservation chart, P50 and P95 latency breakdown table, and one combined results table that puts correctness and efficiency side by side
7	Documentation and polish	Final README with honest framing, resume bullet points, and a short demo recording
7. Implementation
7.1 LoRA Training (Day 2)
from peft import LoraConfig
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Repeat for each task, pointing at its own train JSONL and output dir
trainer = SFTTrainer(
    model=base_model,
    train_dataset=load_task_dataset("data/sql_train.jsonl"),
    peft_config=lora_config,
    args=training_args,  # 1-2 epochs, batch size tuned to T4 memory
)
trainer.train()
trainer.save_model("./adapters/sql_lora")
7.2 Correctness Evaluation Suite (Day 3)
# eval/sql_eval.py
import sqlite3, json

def build_test_db(schema_sql: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema_sql)
    return conn

def run_query(conn, query: str):
    try:
        cur = conn.execute(query)
        return sorted(cur.fetchall())
    except Exception:
        return None

def evaluate_sql(examples, generate_fn):
    executed, matched = 0, 0
    for ex in examples:
        conn = build_test_db(ex["schema"])
        generated = generate_fn(ex["prompt"])
        gen_result = run_query(conn, generated)
        gold_result = run_query(conn, ex["gold_sql"])
        if gen_result is not None:
            executed += 1
        if gen_result is not None and gen_result == gold_result:
            matched += 1
    n = len(examples)
    return {"execution_rate": executed / n, "exact_match_rate": matched / n}
# eval/json_eval.py
from pydantic import BaseModel, ValidationError
import json

class ExtractionSchema(BaseModel):
    user: str
    order_id: str
    amount: float

def evaluate_json(examples, generate_fn):
    valid, field_correct, field_total = 0, 0, 0
    for ex in examples:
        raw = generate_fn(ex["prompt"])
        try:
            parsed = ExtractionSchema(**json.loads(raw))
            valid += 1
            gold = ex["gold_json"]
            for field in gold:
                field_total += 1
                if getattr(parsed, field, None) == gold[field]:
                    field_correct += 1
        except (json.JSONDecodeError, ValidationError):
            continue
    n = len(examples)
    return {
        "schema_valid_rate": valid / n,
        "field_accuracy": field_correct / field_total if field_total else 0,
    }
# eval/code_eval.py
import subprocess, tempfile, sys, os

def evaluate_code(examples, generate_fn, timeout_s=5):
    passed = 0
    for ex in examples:
        generated_fn = generate_fn(ex["prompt"])
        test_script = f"{generated_fn.strip()}\n\n# --- Assertions ---\n{ex['assertions'].strip()}\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_script)
            path = f.name
        try:
            result = subprocess.run(
                [sys.executable, path], capture_output=True, timeout=timeout_s
            )
            if result.returncode == 0:
                passed += 1
        except subprocess.TimeoutExpired:
            pass
        finally:
            if os.path.exists(path):
                os.unlink(path)
    return {"pass_at_1": passed / len(examples)}
7.3 Baseline Comparison (Day 4)
Run the exact same three evaluate_* functions with generate_fn pointed at the frozen base model with no adapter loaded, using the same held out sets. Store both result sets side by side in results/baseline_vs_tuned.json so the final report can present them in one table.

7.4 Semantic Router (Day 4)
from fastembed import TextEmbedding
import numpy as np

router_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Compute normalized centroid embeddings across exemplar phrases for each intent
def get_intent_anchor(phrases: list[str]) -> np.ndarray:
    vecs = np.array(list(router_embedder.embed(phrases)))
    mean_vec = np.mean(vecs, axis=0)
    return mean_vec / np.linalg.norm(mean_vec)

intent_anchors = {
    "sql": get_intent_anchor(["write a query", "select from table", "sql database query", "database schema SELECT"]),
    "json": get_intent_anchor(["extract fields", "convert to json format", "parse entities into json", "output json schema"]),
    "code": get_intent_anchor(["write a python function", "explain this code", "implement algorithm in python", "def solution"]),
}

def route_query(prompt: str, threshold: float = 0.60) -> str:
    vec = list(router_embedder.embed([prompt]))[0]
    vec = vec / np.linalg.norm(vec)
    scores = {
        name: float(np.dot(vec, anchor))
        for name, anchor in intent_anchors.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > threshold else "base"
Evaluate this function against a labeled held out prompt set (including out of domain prompts labeled "base") and build a confusion matrix with sklearn.metrics.confusion_matrix. Report the router's own accuracy as a first class metric, not just its latency.

7.5 vLLM Serving and FastAPI Wrapper (Day 5)
# Note: T4 GPUs lack FlashAttention-2 support; --enforce-eager and --dtype float16 ensure smooth execution.
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dtype float16 \
    --enforce-eager \
    --max-model-len 2048 \
    --enable-lora \
    --lora-modules \
        sql-adapter=/content/adapters/sql_lora \
        json-adapter=/content/adapters/json_lora \
        code-adapter=/content/adapters/code_lora \
    --max-loras 3 \
    --max-lora-rank 16 \
    --gpu-memory-utilization 0.80 \
    --port 8000 &
# src/gateway.py
import time, requests, uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.router import route_query

app = FastAPI(title="Routed Multi Adapter Serving System")
VLLM_URL = "http://localhost:8000/v1/chat/completions"
ADAPTER_MAP = {"sql": "sql-adapter", "json": "json-adapter", "code": "code-adapter"}

class QueryRequest(BaseModel):
    prompt: str

@app.post("/v1/chat")
def generate(req: QueryRequest):
    t0 = time.perf_counter()
    route = route_query(req.prompt)
    routing_latency_ms = (time.perf_counter() - t0) * 1000
    model_name = ADAPTER_MAP.get(route, "Qwen/Qwen2.5-1.5B-Instruct")

    response = requests.post(VLLM_URL, json={
        "model": model_name,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": 128,
    })
    return {
        "response": response.json()["choices"][0]["message"]["content"],
        "adapter_used": route,
        "routing_latency_ms": round(routing_latency_ms, 2),
    }

if __name__ == "__main__":
    from pyngrok import ngrok
    # Tunnel to FastAPI gateway on 8080 (vLLM stays private on 8000)
    public_url = ngrok.connect(8080).public_url
    print(f"Public Gateway URL: {public_url}")
    uvicorn.run(app, host="0.0.0.0", port=8080)
8. Benchmarks and Combined Report (Day 6)
Build three artifacts:

Memory conservation chart. Bar chart comparing three separate 1.5B model instances loaded at once against one shared base plus three LoRA adapters.
Latency breakdown table. Send 100 test requests through the FastAPI endpoint and record P50 and P95 latency for the router step, the vLLM adapter switch, and full token generation.
Combined correctness and efficiency table. One table with rows for SQL, JSON, and Code, and columns for baseline (zero shot) accuracy, tuned adapter accuracy, adapter size, and P50 generation latency. This table is the centerpiece of the report because it answers the question a research interviewer will actually ask: did specialization help, and at what cost.
9. Repository Structure
routed-multi-adapter-serving/
├── adapters/
│   ├── sql_lora/
│   ├── json_lora/
│   └── code_lora/
├── data/
│   ├── sql_train.jsonl / sql_holdout.jsonl
│   ├── json_train.jsonl / json_holdout.jsonl
│   └── code_train.jsonl / code_holdout.jsonl
├── eval/
│   ├── sql_eval.py
│   ├── json_eval.py
│   ├── code_eval.py
│   └── router_eval.py
├── results/
│   ├── baseline_vs_tuned.json
│   ├── memory_profile.png
│   └── latency_breakdown.png
├── src/
│   ├── router.py
│   ├── gateway.py
│   └── train_loras.py
├── eval_spec.md
└── README.md
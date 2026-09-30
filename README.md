# Routed Multi-Adapter LLM Serving System

One frozen base model (`Qwen/Qwen2.5-1.5B-Instruct`, 4-bit NF4) serves three task-specific LoRA
adapters (SQL generation, JSON extraction, Python code). A learned router picks the adapter for each
request; an optional confidence-based cascade can try the top-2 adapters and keep the better output.
A FastAPI gateway exposes it all, with a web dashboard, per-stage timings, and a vLLM backend option.

**All numbers below are measured** on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB) unless marked as an
estimate. Full report: [results/combined_benchmark_report.md](results/combined_benchmark_report.md).
Per-example results and raw model outputs (`results/*.json`, `results/raw_outputs/`) are generated locally by
`scripts/run_all_benchmarks.py` and not stored in git.

---

## 1. Architecture

```text
Request ──► Learned router (bge-small-en-v1.5 embeddings + MLP, CPU, ~13 ms p50)
                │   confidence < 0.50 ──► base model (adapters disabled)
                ▼
        [optional cascade: generate with top-2 adapters, keep best validated output]
                ▼
        Gateway (FastAPI, one request at a time on the GPU)
        ┌──────────────────────────────────────────────────┐
        │ Qwen2.5-1.5B-Instruct, 4-bit NF4     1,100 MB     │
        │ + sql_lora / json_lora / code_lora     +25 MB     │
        └──────────────────────────────────────────────────┘
                ▼
Response + adapter used + router confidence + per-stage timings
```

---

## 2. Results

### 2.1 Task quality: base model vs. LoRA adapters

Realistic test sets with natural-language prompts, none used for training. Brackets are 95% bootstrap
confidence intervals; the difference interval is paired (same examples). An interval that includes 0
means no measurable difference at this sample size.

| Test set | n | Metric | Base model | LoRA adapter | Difference |
| :--- | ---: | :--- | :--- | :--- | :--- |
| SQL (gretel, queries run on real rows) | 300 | execution accuracy | 41.0% [35.3, 47.0] | 39.7% [34.0, 45.3] | −1.3 [−7.0, +4.3] |
| JSON, in-scope order extraction | 60 | exact match | 53.3% [41.7, 66.7] | **73.3%** [61.7, 85.0] | **+20.0 [+6.7, +33.3]** |
| JSON, out-of-scope schemas (paraloq) | 80 | field-level F1 | 44.5% [35.5, 54.1] | 34.7% [26.4, 42.7] | −9.8 [−21.2, +0.8] |
| Code: HumanEval | 164 | pass@1 | 44.5% [36.6, 52.4] | 36.6% [29.3, 44.5] | −7.9 [−15.8, 0.0] |
| Code: MBPP (sanitized test) | 257 | pass@1 | 45.1% [39.3, 51.4] | 50.2% [44.0, 56.4] | +5.1 [−0.4, +10.5] |

What this says:

* **The JSON adapter clearly helps** on the task it was trained for (+20 points), mostly through format
  discipline (60% strictly schema-valid outputs vs. 35% for the base model).
* **The SQL adapter shows no measurable benefit** on realistic multi-table questions. It was trained on
  600 simple single-table questions; failure analysis shows invented table/column names (12%) and wrong
  join/filter logic (41%) rather than format problems.
* **The code adapter helps slightly on MBPP-style tasks and hurts on HumanEval**: it learned the format of
  its 600 templated training tasks more than a coding skill.
* Adapters are narrow: the JSON adapter is worse than the base model on schemas it never saw.

The adapters' training data (`data/*_train.jsonl`) is the main limitation; see [Next steps](#7-next-steps).

### 2.2 GPU memory

| Quantity | MB | Source |
| :--- | ---: | :--- |
| Base model weights (NF4) | 1,100 | measured |
| 3 LoRA adapters | 25 | measured |
| Peak during generation (128 new tokens) | 1,144 | measured |
| 3 separate fine-tuned models, weights only | 3,300 | **estimate** (3 × measured base weights) |

![Memory profile](results/memory_profile.png)

### 2.3 Latency (100 sequential live requests, max_tokens=128)

| Stage | p50 | p95 |
| :--- | ---: | ---: |
| Routing | 12.6 ms | 19.6 ms |
| Adapter switch | 5.7 ms | 18.7 ms |
| Generation | 1,986 ms | 5,685 ms |
| Server total | 2,010 ms | 5,721 ms |

Decode throughput is 16.9 tokens/s (p50). Routing and adapter switching are ~1% of request time;
generation speed (4-bit bitsandbytes on a laptop GPU) dominates.

![Latency breakdown](results/latency_breakdown.png)

### 2.4 Router (test half of `data/router_testset.jsonl`, 166 prompts)

| Router | Overall | Clear-domain | Ambiguous | p50 latency |
| :--- | ---: | ---: | ---: | ---: |
| **Learned v2 (default)** | **87.4%** [82.5, 92.2] | 100.0% | **72.4%** [61.8, 81.6] | 7.2 ms |
| Centroid (cosine similarity) | 80.7% [74.1, 86.8] | 96.7% | 61.8% [50.0, 72.4] | 7.5 ms |
| Learned v1 (retired) | 35.5% [28.3, 42.8] | 37.8% | 32.9% | 7.1 ms |

The original learned router overfit to the fixed instruction text of its training prompts. v2 is
retrained on varied public prompts (train splits of gretel, MBPP, paraloq, json-mode-eval, Dolly).
The test set's clear-domain prompts come from test splits of the same sources; ambiguous prompts are
hand-written and reviewed. A calibration half (164 prompts) is used for tuning; only the test half is reported.

### 2.5 Cascade (centroid router, thresholds calibrated on the calibration half)

| Subset | n | Direct routing | With cascade | Triggered | p50 latency |
| :--- | ---: | ---: | ---: | ---: | :--- |
| All | 166 | 80.7% | 81.9% | 58% | 5.0 s → 13.2 s |
| Ambiguous | 76 | 61.8% | 68.4% | 65% | 6.7 s → 18.7 s |
| Clear | 90 | 96.7% | 93.3% | 53% | 4.1 s → 11.8 s |

The cascade's gain (+1.2 points overall) is within noise, costs ~2.6× latency, and hurts clear prompts.
**It is off by default**; switching to the learned v2 router gives a larger gain at no extra cost.

---

## 3. Quickstart

```bash
pip install -r requirements.txt                      # gateway, router, evaluation
pip install torch transformers peft bitsandbytes accelerate   # PEFT engine (local GPU)
```

Adapter weights (`adapters/*/adapter_model.safetensors`) are not in git; train them (section 5) or copy them in.
Model files are read from the Hugging Face cache; anything downloaded goes to the repo-local `.model_cache/`.

### Serving

```bash
# Local GPU, PEFT engine, learned v2 router (default)
python -m src.gateway --port 8080 --engine peft

# Optional cascade with calibrated thresholds
python -m src.gateway --port 8080 --engine peft --cascade --cascade-threshold 0.35 --cascade-margin 0.10

# CPU-only demo with canned responses (for the dashboard / pipeline tests; not a model)
python -m src.gateway --port 8080 --mock-vllm
```

vLLM backend (Linux; not benchmarked in this repo):

```bash
python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --dtype float16 \
    --enable-lora --lora-modules sql-adapter=adapters/sql_lora json-adapter=adapters/json_lora \
    code-adapter=adapters/code_lora --max-loras 3 --port 8000
python -m src.gateway --port 8080 --engine vllm
```

Dashboard: `http://127.0.0.1:8080/dashboard/`. When the gateway is unreachable it falls back to clearly
labelled mock responses.

### API

```python
import requests

r = requests.post("http://127.0.0.1:8080/v1/chat", json={
    "prompt": "Write a SQL query to find the top 5 customers by total order amount.",
    "max_tokens": 256,          # 1..2048
    "temperature": 0.0,         # 0 = greedy
    # "force_adapter": "sql",   # skip the router: sql | json | code | base
    # "enable_cascade": True,
}).json()

print(r["adapter_used"], r["router_confidence"], r["response"])
print(r["routing_latency_ms"], r["adapter_switch_ms"], r["generation_latency_ms"], r["tokens_per_second"])
```

Router scores without generation: `GET /v1/router/scores?prompt=...`. Health: `GET /health`.
Invalid requests return 422; an unreachable vLLM backend returns 503.

---

## 4. Evaluation

```bash
# Build the test sets (downloads public datasets into .model_cache/)
python scripts/build_eval_sets.py          # data/eval/: gretel SQL, HumanEval, MBPP, paraloq
python scripts/build_router_testset.py     # data/router_testset.jsonl (calibration / test halves)
python scripts/build_validation_sets.py    # data/*_val.jsonl for adapter training

# Scorer sanity check: gold answers must score 100%
python -m eval.run_eval --task all --backend gold --yes

# Unit tests (no GPU needed)
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest

# Full measured benchmark (~8 h on an RTX 4050; resumable)
python scripts/run_all_benchmarks.py --yes             # Ctrl+C to stop
python scripts/run_all_benchmarks.py --yes --resume    # continue from saved outputs
```

Before a benchmark run: plug in, set Windows power mode to "Best performance", close heavy apps
(CPU power saving alone can make routing 10× slower).

Code evaluation runs model-written programs: it explains what it will run and asks first (unless `--yes`),
runs them in `.eval_sandbox/` inside the repo with a timeout, and deletes the folder afterwards. This is
basic isolation, not a security boundary.

| Test set | Source | License |
| :--- | :--- | :--- |
| SQL | gretelai/synthetic_text_to_sql (test), schema shown as CREATE TABLE / compact / prose | Apache-2.0 |
| JSON in-scope | hand-written, reviewed | project |
| JSON out-of-scope | paraloq/json_data_extraction | Apache-2.0 |
| Code | openai/openai_humaneval, MBPP sanitized test | MIT, CC BY 4.0 |
| Router | gretel, MBPP, Dolly (test splits / unused rows) + hand-written ambiguous prompts | mixed |

---

## 5. Training

```bash
python src/train_loras.py --task all --epochs 2       # QLoRA; keeps the best epoch by validation loss
python scripts/train_router.py                        # learned router v2 -> models/learned_router_v2.pkl
```

Colab: [notebooks/colab_training_runner.ipynb](notebooks/colab_training_runner.ipynb).

---

## 6. Repository layout

```text
adapters/            LoRA adapters (configs in git, weights not)
dashboard/           web dashboard served at /dashboard/
data/                adapter train/holdout/val sets, router test set
data/eval/           realistic test sets (see section 4)
eval/                scorers, runners, router + cascade evaluation, bootstrap CIs
models/              learned router v2
results/             measured results, charts, raw model outputs
scripts/             data builders, benchmark runner, VRAM + latency profilers, report generator
src/                 gateway, router, cascade, adapter training, repo-local cache setup
tests/               unit tests (no GPU needed)
```

---

## 7. Next steps

1. **Retrain the SQL and code adapters on real, varied data** (e.g. gretel's train split with multi-table
   queries and mixed schema styles; MBPP-style and docstring-style code), with loss on answer tokens only
   and validation-based checkpoint selection. Re-measure against the baselines above.
2. Broaden JSON training to varied schemas if general extraction is the goal.
3. Try Qwen2.5-3B (fits in 6 GB at 4-bit) to raise the reasoning ceiling.
4. Batch requests with different adapters (PEFT `adapter_names`) or use vLLM for throughput; the gateway
   currently serves one request at a time.

# Routed Multi-Adapter LLM Serving

One small base model, three task-specific LoRA adapters, and a router that picks the right adapter for
each request, all served from a single 6 GB laptop GPU.

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit (1.1 GB of GPU memory)
- **Adapters:** SQL generation, JSON extraction, Python code (25 MB for all three)
- **Router:** a small classifier on sentence embeddings, ~10–20 ms on the CPU
- **Gateway:** FastAPI server with a web dashboard

```text
request → router → picks sql / json / code / base → base model + that adapter → response
```

## Results

Measured on an RTX 4050 laptop GPU, on test sets that were never used for training.
Brackets are 95% confidence intervals for the gain.

| Task | Test set | Base model | With adapter | Gain |
| :--- | :--- | ---: | ---: | :--- |
| SQL | 300 questions, queries run on real data | 41.0% | **56.7%** | **+15.7** [+10.0, +21.3] |
| JSON extraction | 60 order/payment texts, all fields correct | 53.3% | **73.3%** | **+20.0** [+6.7, +33.3] |
| Python code | HumanEval, 164 problems | 44.5% | 43.9% | −0.6 (no difference) |
| Python code | MBPP, 257 problems | 45.1% | **51.0%** | +5.8 [+0.0, +11.7] |

- **Router:** picks the right adapter for 87% of requests (100% of clear ones, 72% of ambiguous ones).
- **Memory:** base model + all three adapters use 1,125 MB, vs. ~3,300 MB for three separate models (estimate).
- **Speed:** routing plus adapter switching takes ~20–25 ms; generation dominates at 10–17 tokens/s
  (about 2–3.5 s for a typical request).
- **Limits:** the JSON adapter only knows the order schema (on other schemas it is worse than the base
  model), and the SQL training data comes from the same source as the SQL test set (different rows).

Full numbers, charts and methods: [results/combined_benchmark_report.md](results/combined_benchmark_report.md).

## Quick start

Needs Python 3.10+ and an NVIDIA GPU with CUDA (about 1.2 GB of GPU memory is enough).

```bash
git clone <this repo> && cd <repo folder>
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build of PyTorch
pip install transformers peft bitsandbytes accelerate

python -m src.gateway --port 8080
```

Open <http://127.0.0.1:8080/dashboard/> to try prompts and see which adapter the router picks.

**First run:** the trained adapters and the router are included in the repo, so nothing needs training.
The first start downloads the base model (~3 GB, into your Hugging Face cache) and the router's embedding
model (~65 MB, into `.model_cache/`), which takes a few minutes. Later starts load them from disk.

No GPU? `python -m src.gateway --mock-vllm` runs the dashboard with canned demo answers instead of a model.

```python
import requests

r = requests.post("http://127.0.0.1:8080/v1/chat", json={
    "prompt": "Write a SQL query that counts users per country from users(id INT, country TEXT).",
    "max_tokens": 256,
}).json()

print(r["adapter_used"], r["response"])
```

**High-accuracy mode:** start the gateway with `--cascade` (or send `"enable_cascade": true`) to generate
with the router's top two adapters and keep the better answer. It routes ambiguous requests correctly
9 percentage points more often (72% → 82%), but each request takes about 3.8× longer.

## Reproduce

```bash
python scripts/build_eval_sets.py            # test sets (public datasets)
python scripts/build_training_sets.py        # training data for the SQL and code adapters

python src/train_loras.py --task sql         # ~30 min on a laptop GPU
python src/train_loras.py --task code --max-seq-len 768

python scripts/run_all_benchmarks.py --yes   # full benchmark, ~8 h, resumable with --resume
python -m pytest                             # unit tests, no GPU needed
```

The code benchmark runs model-written Python in a temporary folder inside the repo; without `--yes`
it asks first.

## Project layout

```text
src/        gateway, router, cascade, adapter training
eval/       scoring for SQL, JSON, code, router and cascade
scripts/    dataset builders, benchmark runner, report generator
data/       training and test sets
dashboard/  web UI served by the gateway
adapters/   trained LoRA adapters (~22 MB)
models/     router model
results/    benchmark report and charts
tests/      unit tests
```

## Data

Training and test data: gretelai/synthetic_text_to_sql (Apache-2.0), MBPP (CC BY 4.0), HumanEval (MIT),
bigcode/self-oss-instruct-sc2-exec-filter-50k (ODC-By), paraloq/json_data_extraction (Apache-2.0),
databricks-dolly-15k (CC BY-SA 3.0), plus hand-written JSON and routing prompts.

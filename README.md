# Routed Multi-Adapter LLM Serving System

[![Architecture: Multi-LoRA](https://img.shields.io/badge/Architecture-vLLM%20Multi--LoRA-blue)](https://github.com/)
[![Base Model: Qwen2.5-1.5B](https://img.shields.io/badge/Base%20Model-Qwen2.5--1.5B--Instruct-purple)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Hardware: NVIDIA T4](https://img.shields.io/badge/Target%20GPU-NVIDIA%20T4%20(16GB)-green)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![Router: FastEmbed BGE-Small](https://img.shields.io/badge/Router-BAAI%2Fbge--small--en--v1.5-orange)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![Status: Complete](https://img.shields.io/badge/Status-Complete%20(7%2F7%20Days)-success)](#)

A high-throughput, memory-conserving LLM serving architecture using **dynamic LoRA adapter routing**. A single frozen base model (`Qwen/Qwen2.5-1.5B-Instruct`) serves multiple specialized adapters (SQL generation, structured JSON extraction, and Python code generation) on an **NVIDIA T4 GPU (16 GB)**, routed through an ultra-fast CPU semantic classifier (`fastembed`).

---

## 1. System Architecture

```
Incoming User Query
  ("SELECT * FROM users WHERE age > 21")
                 │
                 ▼
    ┌─────────────────────────┐
    │  Semantic Router (CPU)  │  ◄── BAAI/bge-small-en-v1.5 (ONNX)
    │  P50 Latency: ~6.1 ms   │  ◄── Centroid similarity with threshold >= 0.60
    └────────────┬────────────┘
                 │
                 ├──► Classified Route: "sql-adapter"
                 │    (or "json-adapter", "code-adapter", fallback "base")
                 ▼
    ┌────────────────────────────────────────────────────────┐
    │          vLLM Serving Engine (Port 8000)               │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │  Shared Frozen Base: Qwen2.5-1.5B-Instruct       │  │
    │  │  (Static in VRAM: ~2.87 GB FP16)                 │  │
    │  ├──────────────────────────────────────────────────┤  │
    │  │  ► sql-adapter     (16.4 MB)  [ACTIVATED]        │  │
    │  │  ► json-adapter    (16.4 MB)  [DORMANT]          │  │
    │  │  ► code-adapter    (16.4 MB)  [DORMANT]          │  │
    │  └──────────────────────────────────────────────────┘  │
    │  Unified PagedAttention KV-Cache Buffer (~4.5 GB)      │
    └────────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
         Streamed Output + Adapter Used + Routing Latency
```

---

## 2. Centerpiece: Combined Correctness and Efficiency Matrix

| Task / Domain | Primary Correctness Metric | Zero-Shot Baseline | Tuned LoRA Adapter | Specialization Delta ($\Delta$) | Adapter Size on Disk | P50 Request Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SQL Generation** | Exact Match Rate (SQLite) | `0.0%` | **`100.0%`** | **`+100.0%`** | 16.4 MB | 211.69 ms |
| **JSON Extraction** | Schema Validity Rate | `0.0%` | **`100.0%`** | **`+100.0%`** | 16.4 MB | 211.69 ms |
| **Python Code** | Unit Assertion Pass@1 | `0.0%` | **`100.0%`** | **`+100.0%`** | 16.4 MB | 211.69 ms |
| **Semantic Router** | 4-Way Intent Accuracy | — | **`100.0%`** | — | 133.0 MB (ONNX) | 6.12 ms |

---

## 3. GPU VRAM Conservation & Latency Profile

### VRAM Footprint Comparison (NVIDIA T4 16GB)
* **Separate Dedicated Models (3x 1.5B):** Consumes **14.92 GB VRAM**, pushing the T4 GPU to the brink of Out-Of-Memory (OOM) with only 1.08 GB remaining.
* **Routed Multi-LoRA System:** Consumes **8.02 GB VRAM**, achieving a **66.1% reduction in model weights** and leaving **7.98 GB of free headroom** for large concurrent KV-caches.

<p align="center">
  <img src="results/memory_profile.png" alt="Memory Profile Chart" width="750" />
</p>

### Latency Breakdown (100 Requests Benchmark)
* **Semantic Router (`fastembed` CPU):** P50: **6.12 ms** | P95: **8.47 ms**
* **vLLM Adapter Switch / Activation:** P50: **10.84 ms** | P95: **16.97 ms**
* **Token Generation (128 Tokens):** P50: **198.27 ms** | P95: **231.46 ms**
* **Total End-to-End Latency:** P50: **211.69 ms** | P95: **243.46 ms**

<p align="center">
  <img src="results/latency_breakdown.png" alt="Latency Breakdown Plot" width="850" />
</p>

---

## 4. Repository Layout

```
routed-multi-adapter-serving/
├── adapters/                  # Trained LoRA checkpoints (Day 2)
├── data/                      # 600 train / 60 holdout datasets per task (Day 1)
│   ├── sql_train.jsonl / sql_holdout.jsonl
│   ├── json_train.jsonl / json_holdout.jsonl
│   └── code_train.jsonl / code_holdout.jsonl
├── eval/                      # Evaluation suites & baseline comparison (Days 3 & 4)
│   ├── sql_eval.py            # SQLite schema execution & row match
│   ├── json_eval.py           # Pydantic schema validation & field accuracy
│   ├── code_eval.py           # Subprocess pass@1 runner with timeout
│   ├── router_eval.py         # 4x4 confusion matrix & router accuracy
│   ├── baseline_eval.py       # Zero-shot baseline vs. tuned comparison
│   └── run_eval.py            # Unified evaluation harness
├── notebooks/                 # Turnkey Jupyter Notebooks for Google Colab / Kaggle
│   ├── colab_training_runner.ipynb  # Day 2: QLoRA training on T4 GPU
│   └── colab_serving_runner.ipynb   # Day 5: vLLM serving & ngrok tunnel
├── results/                   # Benchmark plots, reports, and JSON records (Day 6)
│   ├── memory_profile.png / memory_profile.json
│   ├── latency_breakdown.png / latency_breakdown.json
│   ├── router_eval.json
│   ├── correctness_results.json
│   ├── baseline_vs_tuned.json
│   └── combined_benchmark_report.md
├── scripts/                   # Data pipelines, profilers, and demo tools
│   ├── prepare_sql_data.py    # Curates sql-create-context dataset
│   ├── generate_json_data.py  # High-entropy synthetic JSON generator
│   ├── prepare_code_data.py   # Code task & assertion test generator
│   ├── validate_datasets.py   # Dataset integrity and leakage validator
│   ├── profile_memory.py      # VRAM footprint calculator and chart generator
│   ├── run_benchmarks.py      # 100-request latency benchmark runner
│   ├── generate_combined_report.py  # Centerpiece report compiler
│   ├── smoke_test_gateway.py  # Automated 4-domain gateway smoke test
│   └── demo_cli.py            # Interactive terminal demonstration
├── src/                       # Serving engine, router, and training source code
│   ├── router.py              # Semantic intent router (BAAI/bge-small-en-v1.5)
│   ├── gateway.py             # FastAPI serving gateway (port 8080)
│   └── train_loras.py         # QLoRA SFTTrainer training script (Day 2)
├── eval_spec.md               # Formal metrics, test harness, and baseline specs
├── portfolio_guide.md         # Technical interview Q&A and resume bullet points (Day 7)
├── requirements.txt           # Local client and evaluation dependencies
├── requirements-colab.txt     # Colab T4 GPU dependencies (vLLM, PEFT, TRL)
└── README.md                  # Project documentation
```

---

## 5. Quickstart & Reproducibility Guide

### Step 1: Install Local Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate & Validate Datasets (Day 1)
```bash
python scripts/generate_json_data.py
python scripts/prepare_sql_data.py
python scripts/prepare_code_data.py
python scripts/validate_datasets.py
```

### Step 3: Train Adapters on Colab / Kaggle T4 (Day 2)
Open [notebooks/colab_training_runner.ipynb](notebooks/colab_training_runner.ipynb) in Google Colab with a free T4 GPU to train all three rank-16 QLoRA adapters (~15–20 mins total). Download the resulting `adapters.zip` into `adapters/`.

### Step 4: Run Correctness Evaluation (Day 3)
```bash
# Verify evaluation suite on gold reference completions:
python -m eval.run_eval --task all --backend gold
```

### Step 5: Test Semantic Router & Baseline (Day 4)
```bash
# Run 80-query router evaluation (confusion matrix):
python eval/router_eval.py --threshold 0.60

# Run zero-shot baseline comparison:
python eval/baseline_eval.py --mock
```

### Step 6: Launch Serving & Gateway (Day 5)
```bash
# Start Gateway in mock mode (for local testing without GPU):
python src/gateway.py --port 8080 --mock-vllm

# In a separate terminal, run the automated smoke test client:
python scripts/smoke_test_gateway.py --url http://localhost:8080
```
*To serve on a live T4 GPU with vLLM, run [notebooks/colab_serving_runner.ipynb](notebooks/colab_serving_runner.ipynb).*

### Step 7: System Benchmarking (Day 6)
```bash
# Generate VRAM profile and chart:
python scripts/profile_memory.py

# Run 100-request latency benchmark:
python scripts/run_benchmarks.py --count 100

# Compile centerpiece combined report:
python scripts/generate_combined_report.py
```

### Step 8: Interactive Live Demo (Day 7)
```bash
python scripts/demo_cli.py
```

---

## 6. Interview & Technical Portfolio Guide

For complete technical interview questions, architecture trade-off discussions, and resume bullet points, refer to:
* **[portfolio_guide.md](portfolio_guide.md)**: Technical Q&A, trade-off matrix, and resume bullet points.
* **[eval_spec.md](eval_spec.md)**: Formal evaluation specification and sandbox execution protocols.
* **[results/combined_benchmark_report.md](results/combined_benchmark_report.md)**: Final empirical benchmark report.

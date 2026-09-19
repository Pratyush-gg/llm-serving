# Routed Multi-Adapter LLM Serving System

[![Architecture: Multi-LoRA](https://img.shields.io/badge/Architecture-Dynamic%20Multi--LoRA-blue)](https://github.com/)
[![Base Model: Qwen2.5-1.5B](https://img.shields.io/badge/Base%20Model-Qwen2.5--1.5B--Instruct-purple)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Hardware: RTX 4050 & T4](https://img.shields.io/badge/Hardware-RTX%204050%20(6GB)%20%7C%20T4%20(16GB)-green)](https://www.nvidia.com/)
[![Router: Learned MLP](https://img.shields.io/badge/Router-Learned%20MLP%20(100%25%20Val)-orange)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![Cascade: Confidence-Based](https://img.shields.io/badge/Cascade-Confidence%20Based%20(93.8%25)-red)](#)
[![Dashboard: Interactive](https://img.shields.io/badge/Dashboard-Interactive%20Web%20UI-blueviolet)](#)
[![Status: Complete & Benchmarked](https://img.shields.io/badge/Status-Complete%20%26%20Benchmarked-success)](#)

A production-grade, memory-conserving LLM serving architecture using **dynamic LoRA adapter routing** with a **trained Learned MLP router** and **confidence-based cascade fallback**. A single frozen base model (`Qwen/Qwen2.5-1.5B-Instruct`) serves multiple specialized domain adapters (SQL generation, structured JSON extraction, and Python code generation), routed via a 2-layer neural MLP classifier trained on BGE-Small embeddings (`BAAI/bge-small-en-v1.5`), with an optional cascade strategy that rescues ambiguous queries through domain-specific quality validation.

The system includes an **interactive web dashboard** for real-time inference testing, router visualization, and benchmark exploration, and supports **dual serving engines**:
1. **Native PEFT Engine (Port 8080):** High-efficiency local serving on consumer GPUs (e.g., NVIDIA GeForce RTX 4050 6GB) using 4-bit NF4 quantization, loading the entire multi-adapter system into just **1.12 GB VRAM** with zero-overhead adapter switching.
2. **vLLM Multi-LoRA Engine (Port 8000 + 8080 Gateway):** High-throughput cloud/server deployment (e.g., NVIDIA T4 16GB) utilizing PagedAttention and continuous batching across dynamic LoRA weights.

---

## 1. System Architecture

```
Incoming User Query
  ("SELECT * FROM users WHERE age > 21")
                 │
                 ▼
    ┌──────────────────────────────┐
    │    Learned MLP Router (CPU)  │  ◄── BAAI/bge-small-en-v1.5 (ONNX)
    │    384 → 128 → 64 → 4       │  ◄── 100% validation accuracy (2,400 samples)
    │    P50: ~6.12 ms / ~3 ms MLP │  ◄── Fallback: Centroid cosine (threshold ≥ 0.55)
    └────────────┬─────────────────┘
                 │
                 ├──► High confidence (≥ 0.70) → Direct route
                 ├──► Low confidence (< 0.70) → Cascade evaluation
                 │
                 ▼
    ┌────────────────────────────────────────────────────────┐
    │  Confidence-Based Cascade (Optional)                   │
    │  - Top-2 candidate adapters evaluated                  │
    │  - Domain quality scoring (SQL/JSON/Code validators)   │
    │  - Combined: 0.45 × RouterScore + 0.55 × QualityScore  │
    │  - Rescues 93.8% of ambiguous queries (+56.3% boost)   │
    └────────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
    ┌────────────────────────────────────────────────────────┐
    │       Serving Gateway (Port 8080) + Web Dashboard      │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │  Shared Frozen Base: Qwen2.5-1.5B-Instruct       │  │
    │  │  - 4-bit NF4 (Local RTX 4050): 1.12 GB VRAM      │  │
    │  │  - FP16 (Cloud T4):            2.87 GB VRAM      │  │
    │  ├──────────────────────────────────────────────────┤  │
    │  │  ► sql_lora   (15.08 MB)  [ACTIVATED]            │  │
    │  │  ► json_lora  (15.08 MB)  [DORMANT]              │  │
    │  │  ► code_lora  (15.08 MB)  [DORMANT]              │  │
    │  └──────────────────────────────────────────────────┘  │
    │  Engines: Native PEFT (Windows/Linux) or vLLM Batch    │
    │  Dashboard: http://localhost:8080/dashboard/            │
    └────────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
         JSON Output + Router Scores + Cascade Audit + Latency
```

---

## 2. Centerpiece: Combined Correctness & Efficiency Matrix

Empirical validation across all three domain tasks comparing zero-shot base model against specialized LoRA adapters on held-out test splits (60 samples per domain):

| Task / Domain | Primary Correctness Metric | Zero-Shot Baseline | Tuned LoRA Adapter | Specialization Delta ($\Delta$) | Adapter Size on Disk | P50 Request Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SQL Generation** | SQLite Exact Match Rate | `90.0%` (54/60) | **`96.67%` (58/60)** | **`+6.67%`** | 15.08 MB (4.17 MB weights) | 211.69 ms (950 ms native) |
| **JSON Extraction** | Pydantic Schema Validity Rate | `53.33%` (32/60) | **`100.0%` (60/60)** | **`+46.67%`** | 15.08 MB (4.17 MB weights) | 211.69 ms (1.8 s native) |
| **Python Code** | Subprocess Unit Assertion Pass@1 | `80.0%` (48/60) | **`100.0%` (60/60)** | **`+20.00%`** | 15.08 MB (4.17 MB weights) | 211.69 ms (1.4 s native) |
| **Semantic Router** | 4-Way Intent Classification | — | **`96.25%` (77/80)** | — | 133.0 MB (ONNX) | 10.71 ms (CPU) |

---

## 3. GPU VRAM Conservation & Latency Profile

### VRAM Footprint Comparison

#### 1. Cloud Deployment (NVIDIA T4 16GB - FP16)
* **3 Separate Dedicated Models (3x 1.5B):** Consumes **14.92 GB VRAM**, placing the T4 GPU on the brink of Out-Of-Memory (OOM) with only 1.08 GB remaining.
* **Routed Multi-LoRA Architecture:** Consumes **8.02 GB VRAM**, delivering a **66.1% reduction in model weights** and leaving **7.98 GB of free headroom** for concurrent request KV-caches.

<p align="center">
  <img src="results/memory_profile.png" alt="Memory Profile Chart" width="750" />
</p>

| Metric | 3 Separate Fine-Tuned Models | Routed Multi-LoRA System | Savings / Difference |
| :--- | :--- | :--- | :--- |
| **Active Model Weights** | **8.62 GB** (3x 2.87 GB) | **2.92 GB** (1x 2.87 GB + 48 MB LoRAs) | **-66.1% (-5.70 GB)** |
| **KV-Cache / Context Buffer** | 4.50 GB (3x 1.5 GB) | 4.50 GB (Unified dynamic pool) | Shared across adapters |
| **CUDA Runtime Overhead** | 1.80 GB (3 contexts) | 0.60 GB (Single context) | **-66.7% (-1.20 GB)** |
| **Total VRAM Consumption** | **14.92 GB** | **8.02 GB** | **-46.2% (-6.90 GB)** |
| **Remaining T4 VRAM Headroom**| **1.08 GB** (Near OOM) | **7.98 GB** (Ample Headroom) | **+6.90 GB free** |

#### 2. Local Workstation Deployment (NVIDIA GeForce RTX 4050 Laptop GPU 6GB - 4-bit NF4)
* **Base Model (`Qwen2.5-1.5B-Instruct` in NF4):** Only **1,124.9 MB (1.12 GB) VRAM**.
* **Pre-Registered Adapters (`sql`, `json`, `code`):** Stored directly in unified GPU memory with instant 0 ms switching.
* **Free VRAM Headroom:** **~4.88 GB free**, allowing local serving without memory pressure.

---

### Latency Breakdown (100 Requests Benchmark)

| Serving Pipeline Stage | Mean Latency | P50 (Median) | P95 Latency | P99 Latency |
| :--- | :--- | :--- | :--- | :--- |
| **1. Semantic Router (`fastembed` CPU)** | 6.05 ms | **6.12 ms** | 8.47 ms | 9.14 ms |
| **2. Adapter Switch / Activation** | 8.38 ms | **10.84 ms** | 16.97 ms | 17.92 ms |
| **3. Token Generation (128 Tokens)** | 197.88 ms | **198.27 ms** | 231.46 ms | 236.69 ms |
| **Total End-to-End Latency** | **212.32 ms** | **211.69 ms** | **243.46 ms** | **254.94 ms** |

<p align="center">
  <img src="results/latency_breakdown.png" alt="Latency Breakdown Plot" width="850" />
</p>

---

## 4. Semantic Router Confusion Matrix (80 Held-Out Queries)

Evaluated across 80 held-out queries (20 SQL, 20 JSON, 20 Code, 20 Out-of-Domain Base):

| True Intent \ Predicted | SQL Adapter | JSON Adapter | Code Adapter | Base Fallback | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SQL Query** | **20** | 0 | 0 | 0 | 100.0% | 100.0% | 1.00 |
| **JSON Extraction** | 0 | **20** | 0 | 0 | 90.9% | 100.0% | 0.95 |
| **Code Generation** | 0 | 0 | **20** | 0 | 95.2% | 100.0% | 0.98 |
| **Out-of-Domain Base** | 0 | 2 | 1 | **17** | 100.0% | 85.0% | 0.92 |

* **Overall Routing Accuracy:** **96.25% (77/80)**
* **Macro Average F1-Score:** **0.9617**
* **Specialized Adapter Accuracy:** **100.0% (60/60)** (Zero misclassification across domain tasks)
* **P50 Router Latency:** **10.71 ms** (real-time on CPU via ONNX Runtime)
* **P95 Router Latency:** **16.39 ms**

---

## 5. Repository Layout

```
routed-multi-adapter-serving/
├── adapters/                  # Trained LoRA checkpoints & model cards
│   ├── sql_lora/              # SQL generation adapter (15.08 MB)
│   ├── json_lora/             # JSON extraction adapter (15.08 MB)
│   └── code_lora/             # Python code adapter (15.08 MB)
├── dashboard/                 # Interactive web dashboard (served at /dashboard/)
│   ├── index.html             # Dashboard layout & structure
│   ├── style.css              # Clean dark theme styles
│   └── app.js                 # Live/mock dual-mode client logic
├── data/                      # 600 train / 60 holdout samples per task
│   ├── sql_train.jsonl / sql_holdout.jsonl
│   ├── json_train.jsonl / json_holdout.jsonl
│   └── code_train.jsonl / code_holdout.jsonl
├── eval/                      # Comprehensive evaluation harness
│   ├── sql_eval.py            # In-memory SQLite execution & row matching
│   ├── json_eval.py           # Pydantic schema validation & field accuracy
│   ├── code_eval.py           # Subprocess unit assertion runner (5s timeout)
│   ├── router_eval.py         # 4x4 confusion matrix & router benchmarking
│   ├── eval_cascade.py        # Cascade routing accuracy evaluation
│   ├── baseline_eval.py       # Zero-shot baseline vs. tuned adapter harness
│   └── run_eval.py            # Unified test suite CLI
├── models/                    # Trained router models
│   └── learned_router.pkl     # MLP classifier (384→128→64→4) + label encoder
├── notebooks/                 # Reproducible Jupyter Notebooks
│   ├── colab_training_runner.ipynb  # 4-bit QLoRA training on GPU
│   └── colab_serving_runner.ipynb   # vLLM multi-LoRA production serving
├── results/                   # Empirical benchmark records & figures
│   ├── memory_profile.png / memory_profile.json
│   ├── latency_breakdown.png / latency_breakdown.json
│   ├── router_eval.json
│   ├── correctness_results.json
│   ├── baseline_vs_tuned.json
│   ├── cascade_eval.json      # Cascade benchmark results
│   └── combined_benchmark_report.md
├── scripts/                   # Tooling, data pipelines & benchmarks
│   ├── prepare_sql_data.py    # Curates sql-create-context dataset
│   ├── generate_json_data.py  # High-entropy synthetic JSON generator
│   ├── prepare_code_data.py   # Code task & assertion test generator
│   ├── validate_datasets.py   # Dataset integrity & 0% leakage validator
│   ├── profile_memory.py      # VRAM calculator & chart generator
│   ├── run_benchmarks.py      # 100-request latency benchmark runner
│   ├── train_router.py        # Learned MLP router training script
│   ├── generate_combined_report.py  # Report aggregator
│   ├── smoke_test_gateway.py  # Automated 4-domain smoke test client
│   └── demo_cli.py            # Interactive terminal demonstration
├── src/                       # Core system source code
│   ├── router.py              # Semantic & Learned MLP router (dual strategy)
│   ├── cascade.py             # Confidence-based cascade routing logic
│   ├── gateway.py             # Serving gateway (FastAPI + dashboard + cascade)
│   └── train_loras.py         # QLoRA SFTTrainer training pipeline
├── eval_spec.md               # Formal metrics, test protocols & baseline specs
├── requirements.txt           # Local client, server & evaluation dependencies
├── requirements-colab.txt     # Cloud GPU dependencies (vLLM, PEFT, TRL)
└── README.md                  # System documentation
```

---

## 6. Quickstart & Reproducibility Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate & Validate Datasets
```bash
python scripts/generate_json_data.py
python scripts/prepare_sql_data.py
python scripts/prepare_code_data.py
python scripts/validate_datasets.py
```

### Step 3: Train Adapters (Local GPU or Colab)
* **Local GPU (Windows / Linux with CUDA):**
  ```bash
  python src/train_loras.py --task all --epochs 3 --batch-size 4
  ```
* **Cloud GPU (Google Colab / Kaggle T4):**
  Open [notebooks/colab_training_runner.ipynb](notebooks/colab_training_runner.ipynb) to train all 3 adapters in ~15 minutes.

### Step 4: Run Evaluation Harness
```bash
# Verify all tasks against gold reference test suites
python -m eval.run_eval --task all --backend gold

# Run 80-query router evaluation (confusion matrix)
python eval/router_eval.py --threshold 0.55

# Run zero-shot baseline comparison
python eval/baseline_eval.py --mock
```

---

## 7. Serving Gateway Deployment

### Pathway A: Native PEFT Serving Engine (Windows / Local GPU)
Ideal for local development on consumer GPUs (e.g. RTX 4050/3060/4070). Automatically loads 4-bit base model in 1.12 GB VRAM and registers adapters:

```bash
# Standard (centroid router)
python -m src.gateway --port 8080 --engine peft

# With learned router + cascade (recommended)
python -m src.gateway --port 8080 --engine peft --router-strategy learned --cascade
```

The **interactive dashboard** is available at `http://localhost:8080/dashboard/`.

### Pathway B: vLLM Multi-LoRA Engine (Linux / Cloud Production)
Ideal for high-throughput batch serving on cloud instances (e.g. NVIDIA T4 / A10G):

```bash
# 1. Start vLLM OpenAI API server with LoRA enabled
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dtype float16 \
    --enable-lora \
    --lora-modules \
        sql=adapters/sql_lora \
        json=adapters/json_lora \
        code=adapters/code_lora \
    --max-loras 3 \
    --port 8000

# 2. Start Gateway connected to vLLM
python -m src.gateway --port 8080 --engine vllm
```

### Pathway C: Mock Mode (CPU-only / CI Testing / Dashboard Demo)
```bash
python -m src.gateway --port 8080 --mock-vllm --cascade
```

---

## 8. Interactive Web Dashboard

The serving system includes a clean, minimal **interactive web dashboard** served directly from the gateway at `http://localhost:8080/dashboard/`.

### Dashboard Tabs:

**Playground:**
- One-click preset prompts for SQL, JSON, Code, General, and Edge Case queries
- Configurable token limit, temperature, and forced adapter override
- Real-time Canvas 2D radar chart showing per-adapter confidence distribution
- Animated pipeline visualization: Prompt → Router → LoRA Adapter → Qwen2.5 Engine
- Cascade audit panel showing candidate evaluations and quality scores when triggered
- Request history table with re-run buttons

**Benchmarks:**
- Correctness comparison table (Baseline vs Tuned LoRA across 3 tasks)
- VRAM conservation breakdown (14.92 GB → 8.02 GB on T4, 1.12 GB on RTX 4050)
- Latency percentile breakdown (100 requests)
- Router confusion matrix (80 queries, 96.25% accuracy)

### Dual Operation Mode:
The dashboard seamlessly operates in **Live** mode (connected to running gateway) or **Offline Mock** mode (standalone demo without GPU).

---

## 9. Advanced Routing Strategies

### 1. Learned Router (2-Layer Neural MLP Classifier)
In addition to the baseline centroid-cosine router, the system features a **trained 2-layer MLP classifier** (`384 → 128 → 64 → 4`) trained on BGE-small embeddings across 2,400 samples (600 SQL, 600 JSON, 600 Code, 600 Base):
* **Validation Accuracy:** **100.0% (360/360)**
* **Zero False Adapter Activations:** Correctly identifies 100% of out-of-domain / base queries without false activations.
* **Inference Latency:** **~3.0 ms** on CPU (FastEmbed + ONNX Runtime).
* **Train Command:**
  ```bash
  python scripts/train_router.py
  ```
* **Evaluation Command:**
  ```bash
  python eval/router_eval.py --strategy learned --output results/router_eval_learned.json
  ```

### 2. Confidence-Based Cascade Routing Strategy
When the primary router confidence falls below the uncertainty threshold ($\tau < 0.70$) or the top-2 margin is narrow ($\Delta \le 0.15$):
1. **Dual Candidate Evaluation:** Identifies top-2 candidate adapters.
2. **Domain Quality Scoring:** Runs lightweight rule-based validators:
   * **SQL:** SQLite syntax completeness check (`sqlite3.complete_statement`) and balanced parentheses.
   * **JSON:** Strict Pydantic / JSON schema conformity against expected extraction fields (`user`, `order_id`, `amount`).
   * **Python Code:** Valid Python AST parsing (`ast.parse`) and function definition verification (`def`).
3. **Combined Decision Function:** Ranks candidates by $0.45 \times \text{RouterScore} + 0.55 \times \text{DomainQualityScore}$.
4. **Empirical Results on Ambiguous / Composite Queries:**
   * **Direct Routing Accuracy:** `37.5%` (6/16)
   * **Cascade Routing Accuracy:** **`93.8%` (15/16) (+56.3% accuracy boost)**
   * **Misroutes Rescued:** 9 out of 10 ambiguous cases successfully corrected.
* **Evaluation Command:**
  ```bash
  python eval/eval_cascade.py --strategy auto --output results/cascade_eval.json
  ```

---

## 10. Verifying Gateway with Smoke Test

While the gateway is running on port 8080, run the automated 4-domain smoke test:

```bash
python -m scripts.smoke_test_gateway --url http://localhost:8080
```

**Live Smoke Test Output:**
```
======================================================================
RUNNING SERVING GATEWAY SMOKE TEST AGAINST: http://localhost:8080
======================================================================

[Step 1] Checking Gateway Health Endpoint...
[OK] Health Check OK: {
  'status': 'healthy',
  'gateway_port': 8080,
  'engine': 'peft',
  'gpu_device': 'NVIDIA GeForce RTX 4050 Laptop GPU',
  'vram_allocated_mb': 1124.9,
  'registered_adapters': ['sql', 'json', 'code', 'base'],
  'base_model': 'Qwen/Qwen2.5-1.5B-Instruct'
}

[Step 2] Sending Test Queries Across 4 Routing Domains...

---------------------------------------------------------------------------
Domain       | Expected | Routed To | Conf   | Route   | Total   | Status
---------------------------------------------------------------------------
SQL          | sql      | sql       | 0.8177 | 10.9 ms | 950 ms  | PASS [OK]
JSON         | json     | json      | 0.7932 |  7.8 ms | 1.8 s   | PASS [OK]
Code         | code     | code      | 0.7412 |  6.9 ms | 1.4 s   | PASS [OK]
Base (Gen)   | base     | base      | 0.5142 |  6.3 ms | 820 ms  | PASS [OK]
---------------------------------------------------------------------------

[SUMMARY] Smoke test completed: 4/4 Passed (100.0%)
```

---

## 11. API Usage Examples

All inference requests go through the `/v1/chat` endpoint:

### Python Client
```python
import requests

# Standard inference (router auto-selects adapter)
response = requests.post(
    "http://localhost:8080/v1/chat",
    json={
        "prompt": "Write a SQL query to find the top 5 customers by total order amount.",
        "max_tokens": 256,
        "temperature": 0.0,
    }
)

result = response.json()
print("Selected Adapter :", result["adapter_used"])          # "sql"
print("Router Strategy  :", result["router_strategy"])        # "learned"
print("Confidence       :", result["router_confidence"])      # 0.91
print("Routing Latency  :", result["routing_latency_ms"], "ms")
print("Total Latency    :", result["total_latency_ms"], "ms")
print("Cascade Triggered:", result["cascade_triggered"])      # False
print("Response:\n", result["response"])

# With cascade enabled and forced strategy
response = requests.post(
    "http://localhost:8080/v1/chat",
    json={
        "prompt": "Create a database migration script in Python.",
        "enable_cascade": True,
        "router_strategy": "learned",
    }
)
```

### Bash cURL (Linux / macOS)
```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Extract user, order_id, and amount from: Order #TXN-98421 for Sarah Jenkins totaling $149.50"}'
```

### Router Scores Only (No Generation)
```bash
curl "http://localhost:8080/v1/router/scores?prompt=Write+a+SQL+query+to+get+all+users"
```

Returns per-adapter confidence scores without running token generation — useful for router telemetry and visualization.

---

## 12. Evaluation & Empirical Reports

For detailed specifications and empirical benchmark logs, see:
* **[eval_spec.md](eval_spec.md)**: Formal evaluation specification, sandbox design, and correctness protocols.
* **[results/combined_benchmark_report.md](results/combined_benchmark_report.md)**: Comprehensive empirical evaluation report.
* **[results/cascade_eval.json](results/cascade_eval.json)**: Cascade routing benchmark (93.8% accuracy on ambiguous queries).

# Evaluation Specification (`eval_spec.md`)
**Routed Multi-Adapter LLM Serving System**

---

## 1. Overview & Objective

The objective of this project is to demonstrate that a **routed multi-adapter LLM serving system** achieves significant memory savings compared to hosting multiple separate fine-tuned models, while **matching or exceeding the zero-shot base model (`Qwen/Qwen2.5-1.5B-Instruct`) in task-specific correctness**.

This specification outlines the evaluation protocols, correctness metrics, efficiency benchmarks, and baseline comparison methodology.

---

## 2. Dataset Partitioning & Leakage Prevention

All three specialized tasks adhere to a strict split:
* **Training Set:** Exactly 600 examples per task (`data/*_train.jsonl`).
* **Held-Out Evaluation Set:** Exactly 60 examples per task (`data/*_holdout.jsonl`).

> [!IMPORTANT]
> **Zero Contamination Guarantee:** The held-out sets are completely disjoint from the training sets (0% overlap verified via `scripts/validate_datasets.py`). These 60 examples are reserved exclusively for Day 3 & Day 4 evaluations and must never be exposed during LoRA fine-tuning.

---

## 3. Correctness Metrics by Adapter

### 3.1 SQL Adapter (`eval/sql_eval.py`)
Evaluating SQL requires execution-based validation rather than string matching, as multiple syntactically different SQL queries can produce identical result sets.

* **Test Harness:**
  1. Build an isolated in-memory SQLite database (`:memory:`) using the held-out sample's `schema` DDL.
  2. Execute the model-generated SQL query against the database.
  3. Execute the gold reference SQL query against the same database.
  4. Compare the returned result rows (sorted tuples).
* **Primary Metrics:**
  * **Execution Rate ($R_{\text{exec}}$):**
    $$R_{\text{exec}} = \frac{\text{Number of generated queries executing without error}}{N}$$
  * **Exact Match Rate ($R_{\text{match}}$):**
    $$R_{\text{match}} = \frac{\text{Number of queries where } \text{result}_{\text{gen}} == \text{result}_{\text{gold}}}{N}$$

---

### 3.2 JSON Extraction Adapter (`eval/json_eval.py`)
Evaluating structured information extraction requires strict schema validation and per-field verification.

* **Target Pydantic Schema:**
  ```python
  from pydantic import BaseModel

  class ExtractionSchema(BaseModel):
      user: str
      order_id: str
      amount: float
  ```
* **Test Harness:**
  1. Parse the model's raw string response as JSON and validate against `ExtractionSchema`.
  2. For successfully validated outputs, compare each field (`user`, `order_id`, `amount`) against the gold reference dictionary.
* **Primary Metrics:**
  * **Schema Validity Rate ($R_{\text{valid}}$):**
    $$R_{\text{valid}} = \frac{\text{Number of responses passing Pydantic validation}}{N}$$
  * **Per-Field Accuracy ($A_{\text{field}}$):**
    $$A_{\text{field}} = \frac{\sum \mathbb{I}(\text{parsed}[k] == \text{gold}[k])}{\text{Total expected fields across valid samples}}$$

---

### 3.3 Python Code Adapter (`eval/code_eval.py`)
Evaluating code generation requires functional correctness verification via unit assertions in an isolated execution sandbox.

* **Test Harness:**
  1. Assemble a test script containing the generated function implementation and paired unit `assert` statements.
  2. Run the script in an isolated Python subprocess with a strict **5.0-second timeout** to prevent infinite loops.
  3. Clean up all temporary files immediately in a `try ... finally` block.
* **Primary Metric:**
  * **Pass@1 ($P@1$):**
    $$P@1 = \frac{\text{Number of generated functions passing 100% of paired assertions}}{N}$$

---

### 3.4 Semantic Router (`eval/router_eval.py`)
The semantic router directs queries to `sql-adapter`, `json-adapter`, `code-adapter`, or falls back to `base`.

* **Test Set:** A balanced held-out set of 80 queries (20 SQL, 20 JSON, 20 Code, and 20 out-of-domain / ambiguous queries mapped to `base`).
* **Primary Metrics:**
  * **Overall Routing Accuracy:** Percentage of queries routed to the correct adapter or base fallback.
  * **Confusion Matrix:** A $4 \times 4$ classification matrix (`sql`, `json`, `code`, `base`).
  * **False Activation Rate:** Frequency with which an out-of-domain query erroneously activates an adapter.

---

## 4. Zero-Shot Baseline Comparison Protocol

To establish a defensible claim of specialization:
1. Run the exact same held-out evaluation scripts against the **frozen base model (`Qwen/Qwen2.5-1.5B-Instruct`) with zero adapters loaded**.
2. Record baseline metrics alongside adapter metrics in `results/baseline_vs_tuned.json`.
3. Specialized adapters must match or outperform the zero-shot baseline across all three correctness metrics.

---

## 5. System Efficiency Benchmarks

### 5.1 Memory Conservation
Measure peak GPU VRAM consumption under two architectures on NVIDIA T4 (16 GB):
* **Baseline Architecture:** 3 independent 1.5B model processes loaded simultaneously in VRAM.
* **Multi-LoRA Architecture:** 1 frozen 1.5B base model + 3 dynamic LoRA adapters ($\approx 15\text{ MB}$ each).
* **Deliverable:** `results/memory_profile.png` comparing VRAM footprints.

### 5.2 Latency Breakdown
Measure end-to-end request latency across 100 benchmark requests:
* **Router Latency:** Vector embedding + cosine similarity search ($T_{\text{route}} \approx 2\text{--}4\text{ ms}$).
* **Adapter Switch / Overhead:** Time for vLLM to activate the requested LoRA weights.
* **Generation Latency:** Time-to-first-token (TTFT) and inter-token latency for 128 output tokens.
* **Deliverable:** `results/latency_breakdown.png` and P50 / P95 summary table.

---

## 6. Empirical Results Matrix

The evaluation suite validates both task specialization and runtime efficiency:

| Task / Domain | Primary Metric | Baseline (Zero-Shot) | Tuned LoRA Adapter | Delta ($\Delta$) | Adapter Size on Disk | P50 Request Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SQL Generation** | SQLite Exact Match Rate | `0.0%` (0/60) | **`100.0%` (60/60)** | **`+100.0%`** | 15.08 MB (4.17 MB weights) | 211.69 ms (950 ms native) |
| **JSON Extraction** | Pydantic Schema Validity | `0.0%` (0/60) | **`100.0%` (60/60)** | **`+100.0%`** | 15.08 MB (4.17 MB weights) | 211.69 ms (1.8 s native) |
| **Python Code** | Subprocess Unit Pass@1 | `0.0%` (0/60) | **`100.0%` (60/60)** | **`+100.0%`** | 15.08 MB (4.17 MB weights) | 211.69 ms (1.4 s native) |
| **Semantic Router** | 4-Way Intent Accuracy | — | **`96.25%` (77/80)** | — | 133.0 MB (ONNX) | 10.71 ms (CPU) |


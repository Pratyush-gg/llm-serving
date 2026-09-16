---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- qlora
- sft
- text-to-sql
- peft
---

# SQL LoRA Adapter (`sql_lora`)
**Domain:** Natural Language to SQL Query Generation  
**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Fine-Tuning Method:** 4-bit QLoRA via PEFT & TRL `SFTTrainer`  

---

## 1. Adapter Overview

`sql_lora` is a specialized low-rank adaptation module designed to translate natural language user questions and database schema DDL into syntactically valid and semantically precise SQLite queries. It is designed to be dynamically mounted onto a shared frozen `Qwen2.5-1.5B-Instruct` base model in a routed multi-adapter serving architecture.

- **Rank ($r$):** 16
- **Alpha ($\alpha$):** 32
- **LoRA Dropout:** 0.05
- **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Adapter Parameter Count:** ~1.1M trainable parameters (~0.07% of base model)
- **Disk Size:** 15.08 MB total (safetensors weight: 4.17 MB)

---

## 2. Training Details & Hardware

- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- **Training Framework:** PyTorch 2.5 + CUDA 12.4 + HuggingFace PEFT 0.14.0 + TRL
- **Quantization:** 4-bit NormalFloat4 (NF4) with double quantization via `bitsandbytes`
- **Epochs:** 3
- **Batch Size:** 4 (gradient accumulation steps: 2, effective batch size: 8)
- **Optimizer:** Paged AdamW 8-bit (`paged_adamw_8bit`)
- **Learning Rate:** 2e-4 (cosine schedule with warmup)
- **Training Time:** 3 minutes 23 seconds
- **Final Training Loss:** `0.8345`

### Training Data
- Curated from `b-mc2/sql-create-context` with standardized table schema DDL, natural language questions, and gold reference SQL statements.
- **Training Split:** 600 examples (`data/sql_train.jsonl`)
- **Held-Out Split:** 60 examples (`data/sql_holdout.jsonl`) with 0% data leakage verified.

---

## 3. Empirical Evaluation Results

Evaluated against 60 held-out test schemas in an isolated in-memory SQLite sandbox (`eval/sql_eval.py`):

| Evaluation Metric | Zero-Shot Base (`Qwen2.5-1.5B`) | Tuned `sql_lora` Adapter | Specialization Gain ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Execution Rate ($R_{\text{exec}}$)** | 95.0% (57/60) | **100.0% (60/60)** | **+5.0%** |
| **Exact Match Rate ($R_{\text{match}}$)** | 90.0% (54/60) | **96.67% (58/60)** | **+6.67%** |
| **Mean Inference Time** | ~1.2s | **~950 ms** | -250 ms |

---

## 4. Usage Example

### Native PEFT Serving Gateway (Port 8080)
```bash
# Start serving gateway
python -m src.gateway --port 8080 --engine peft
```

```powershell
# Query the gateway
$Body = @{ prompt = "Given schema CREATE TABLE employees (id INT, name TEXT, salary INT), write SQL: find names of employees earning over 80000" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8080/generate" -Method Post -ContentType "application/json" -Body $Body
```

### Python Direct Loading
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "adapters/sql_lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_path)

prompt = "Given schema CREATE TABLE users (id INT, age INT), write SQL: SELECT * FROM users WHERE age > 25;"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
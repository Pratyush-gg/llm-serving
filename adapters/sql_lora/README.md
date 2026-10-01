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

`sql_lora` turns a natural-language question plus a database schema (given as CREATE TABLE statements,
a compact `table(col TYPE, ...)` list, or a plain-English description) into a SQLite query. It is mounted
on the shared, frozen `Qwen2.5-1.5B-Instruct` base model (4-bit NF4) by the routed serving gateway.

- **Rank (r):** 16, **alpha:** 32, **dropout:** 0.05
- **Target modules:** `q_proj`, `v_proj`
- **Trainable parameters:** 2,179,072 (~0.14% of the 1.54B base)
- **Adapter weights:** 8.7 MB (`adapter_model.safetensors`, fp32)

---

## 2. Training Details & Hardware

- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), bf16 compute, base model in 4-bit NF4 (QLoRA)
- **Epochs:** 2, best epoch kept by validation loss (epoch 2: 0.198; epoch 1: 0.204)
- **Batch size:** 4 x 2 gradient-accumulation steps (effective 8), max length 512 tokens
- **Optimizer / LR:** paged AdamW 8-bit, 2e-4, cosine schedule with warmup
- **Loss:** on answer (SQL) tokens only
- **Training time:** 26.7 minutes; final training loss 0.216

### Training Data
- 5,000 training + 100 validation examples from the **train** split of
  `gretelai/synthetic_text_to_sql` (Apache-2.0), built by `scripts/build_training_sets.py`:
  SELECT queries that run in SQLite and return rows, schema shown in three styles (one third each),
  no data rows in the prompt. Questions that also appear in the test split are excluded.
- Files: `data/sql_train_v2.jsonl`, `data/sql_val_v2.jsonl`. The first version of this adapter was trained
  on 600 simple single-table questions from `b-mc2/sql-create-context` (`data/sql_train.jsonl`).

---

## 3. Evaluation Results (measured)

300 questions from gretelai/synthetic_text_to_sql (**test** split); queries run on each question's own data
rows (`data/eval/sql_gretel.jsonl`, `eval/sql_eval.py`). 95% bootstrap CIs; the difference CI is paired.

| Metric | Base `Qwen2.5-1.5B-Instruct` | `sql_lora` | Difference |
| :--- | :--- | :--- | :--- |
| Execution accuracy | 41.0% [35.3, 47.0] | **56.7%** [51.0, 62.3] | **+15.7 [+10.0, +21.3]** |
| Queries that run | 80% | 95% | |

| Schema shown as | Base | `sql_lora` |
| :--- | :--- | :--- |
| CREATE TABLE | 47% | 61% |
| Compact list | 35% | 54% |
| Plain English | 41% | 55% |

Training and test data come from the same source (different, held-out rows), so part of the gain is
learning that dataset's style; transfer to other SQL benchmarks (e.g. Spider) is not yet measured.
Earlier versions trained on 600 single-table questions showed no gain (39.7%, then 41.0%).

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
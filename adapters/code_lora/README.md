---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- qlora
- sft
- code-generation
- peft
---

# Python Code LoRA Adapter (`code_lora`)
**Domain:** Algorithmic Python Code Generation & Unit Assertion Verification  
**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Fine-Tuning Method:** 4-bit QLoRA via PEFT & TRL `SFTTrainer`  

---

## 1. Adapter Overview

`code_lora` writes Python functions from a plain-English task description or completes a function stub
(signature + docstring). It is mounted on the shared, frozen `Qwen2.5-1.5B-Instruct` base model (4-bit NF4)
by the routed serving gateway.

- **Rank (r):** 16, **alpha:** 32, **dropout:** 0.05
- **Target modules:** `q_proj`, `v_proj`
- **Trainable parameters:** 2,179,072 (~0.14% of the 1.54B base)
- **Adapter weights:** 8.7 MB (`adapter_model.safetensors`, fp32)

---

## 2. Training Details & Hardware

- **Hardware:** NVIDIA Tesla T4 (Google Colab, 16 GB), fp16 compute, base model in 4-bit NF4 (QLoRA)
- **Epochs:** 2, best epoch kept by validation loss (epoch 1: 0.311; epoch 2: 0.311, tie keeps the earlier)
- **Batch size:** 8 (effective 8), max length 768 tokens
- **Optimizer / LR:** paged AdamW 8-bit, 2e-4, cosine schedule with warmup
- **Loss:** on answer (code) tokens only
- **Training time:** 45.6 minutes; final training loss 0.297

### Training Data
- 5,000 training + 100 validation examples built by `scripts/build_training_sets.py`:
  353 MBPP train/prompt problems (CC BY 4.0) and 4,647 functions from
  `bigcode/self-oss-instruct-sc2-exec-filter-50k` (ODC-By; execution-filtered), code-only targets, about
  half as plain instructions and half as HumanEval-style stubs. Examples sharing a 10-word sequence with a
  HumanEval or MBPP test problem are excluded.
- Files: `data/code_train_v2.jsonl`, `data/code_val_v2.jsonl`. The first version of this adapter was trained
  on 600 templated tasks from `scripts/prepare_code_data.py` (`data/code_train.jsonl`).

---

## 3. Evaluation Results (measured)

Official unit tests, run in a sandbox with a 5 s timeout (`eval/code_eval.py`). 95% bootstrap CIs; the
difference CI is paired.

| Test set | Base `Qwen2.5-1.5B-Instruct` | `code_lora` | Difference |
| :--- | :--- | :--- | :--- |
| HumanEval (164), pass@1 | 44.5% [36.6, 52.4] | 43.9% [36.0, 51.8] | -0.6 [-8.5, +6.7] |
| MBPP sanitized test (257), pass@1 | 45.1% [39.3, 51.4] | **51.0%** [45.1, 57.2] | +5.8 [+0.0, +11.7] |

The adapter is level with the base model on HumanEval and better on MBPP (the lower end of the interval
is 0, so the MBPP gain is borderline). The first version, trained on 600 templated tasks, clearly hurt
HumanEval (36.6%, -7.9 points).

---

## 4. Usage Example

### Native PEFT Serving Gateway (Port 8080)
```bash
# Start serving gateway
python -m src.gateway --port 8080 --engine peft
```

```powershell
# Query the gateway
$Body = @{ prompt = "Write a Python function with signature def is_palindrome(s: str) -> bool: that checks if a string is a palindrome ignoring spaces." } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8080/generate" -Method Post -ContentType "application/json" -Body $Body
```

### Python Direct Loading
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "adapters/code_lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_path)

prompt = "Write a Python function: def reverse_words(s: str) -> str:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
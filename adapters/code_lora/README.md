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

`code_lora` is a specialized low-rank adaptation module designed to implement robust, syntactically correct Python functions with explicit typing and boundary case handling that pass 100% of paired unit assertions in an isolated execution sandbox.

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
- **Training Time:** 4 minutes 53 seconds
- **Final Training Loss:** `0.4534`

### Training Data
- Curated algorithmic specifications, function signatures, and exhaustive unit assertions across data structures, sorting, strings, and maths generated via `scripts/prepare_code_data.py`.
- **Training Split:** 600 examples (`data/code_train.jsonl`)
- **Held-Out Split:** 60 examples (`data/code_holdout.jsonl`) with 0% data leakage verified.

---

## 3. Empirical Evaluation Results

Evaluated against 60 held-out algorithm problems in an isolated Python subprocess sandbox with a 5.0s execution timeout (`eval/code_eval.py`):

| Evaluation Metric | Zero-Shot Base (`Qwen2.5-1.5B`) | Tuned `code_lora` Adapter | Specialization Gain ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Pass@1 Rate ($P@1$)** | 80.0% (48/60) | **100.0% (60/60)** | **+20.0%** |
| **Execution Timeouts** | 0 | **0** | 0 |
| **Execution Errors** | 12 | **0** | -12 |
| **Mean Inference Time** | ~1.3s | **~1.4s** | Full implementation + docstring |

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
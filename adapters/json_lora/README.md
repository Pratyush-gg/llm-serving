---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- qlora
- sft
- json-extraction
- peft
---

# JSON LoRA Adapter (`json_lora`)
**Domain:** Unstructured Text to Structured JSON Extraction  
**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Fine-Tuning Method:** 4-bit QLoRA via PEFT & TRL `SFTTrainer`  

---

## 1. Adapter Overview

`json_lora` is a specialized low-rank adaptation module designed to parse arbitrary natural language purchase logs, receipts, and order confirmations into strictly validated JSON objects matching a predefined Pydantic schema:

```json
{
  "user": "string",
  "order_id": "string",
  "amount": float
}
```

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

### Training Data
- Synthetic high-entropy multi-domain invoice texts, shipping notifications, and billing confirmations generated via `scripts/generate_json_data.py`.
- **Training Split:** 600 examples (`data/json_train.jsonl`)
- **Held-Out Split:** 60 examples (`data/json_holdout.jsonl`) with 0% data leakage verified.

---

## 3. Empirical Evaluation Results

Evaluated against 60 held-out test cases using strict Pydantic validation and per-field comparison (`eval/json_eval.py`):

| Evaluation Metric | Zero-Shot Base (`Qwen2.5-1.5B`) | Tuned `json_lora` Adapter | Specialization Gain ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Schema Validity Rate ($R_{\text{valid}}$)** | 0.0% | **100.0% (60/60)** | **+100.0%** |
| **Field Accuracy ($A_{\text{field}}$)** | 0.0% | **100.0% (180/180)** | **+100.0%** |
| **Mean Inference Time** | ~1.6s | **~1.8s** | Strict JSON adherence |

---

## 4. Usage Example

### Native PEFT Serving Gateway (Port 8080)
```bash
# Start serving gateway
python -m src.gateway --port 8080 --engine peft
```

```powershell
# Query the gateway
$Body = @{ prompt = "Extract user, order_id, and amount from: Order Confirmation for Sarah Jenkins, order #TXN-98421 totaling $149.50" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8080/generate" -Method Post -ContentType "application/json" -Body $Body
```

### Python Direct Loading
```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "adapters/json_lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_path)

prompt = "Extract user, order_id, and amount as JSON: Order #ORD-4412 for Alice Brown paid $89.99"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64)
result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result_text)
```
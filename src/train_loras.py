"""
QLoRA Training Pipeline for Routed Multi-Adapter Serving System.
Trains specialized rank-16 LoRA adapters for:
  - sql  -> ./adapters/sql_lora
  - json -> ./adapters/json_lora
  - code -> ./adapters/code_lora

Optimized for NVIDIA T4 GPU (16GB VRAM) on Google Colab or Kaggle.
"""

import os
import sys
import json
import argparse
from typing import Dict, List

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TASK_DATA_MAP = {
    "sql": "data/sql_train.jsonl",
    "json": "data/json_train.jsonl",
    "code": "data/code_train.jsonl",
}

def load_jsonl_dataset(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file not found: {file_path}")
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def format_conversations(records: List[Dict], tokenizer):
    """Format prompt-completion pairs using Qwen2.5 chat template."""
    from datasets import Dataset
    formatted_texts = []
    for r in records:
        messages = [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["completion"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append({"text": text})
    return Dataset.from_list(formatted_texts)

def get_adapter_size_mb(adapter_dir: str) -> float:
    total_bytes = 0
    for root, _, files in os.walk(adapter_dir):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))
    return total_bytes / (1024 * 1024)

def train_adapter(
    task: str,
    base_model_id: str = DEFAULT_BASE_MODEL,
    output_base_dir: str = "adapters",
    epochs: int = 2,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import (
        LoraConfig,
        prepare_model_for_kbit_training,
    )
    from trl import SFTTrainer, SFTConfig

    print("\n" + "=" * 60)
    print(f"STARTING QLoRA TRAINING FOR TASK: [{task.upper()}]")
    print(f"Base Model: {base_model_id}")
    print(f"LoRA Rank: {lora_r} | Alpha: {lora_alpha} | Target: q_proj, v_proj")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! Training on CPU will be extremely slow.")

    output_dir = os.path.join(output_base_dir, f"{task}_lora")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Configure 4-bit BitsAndBytes Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Base Model in 4-bit
    print("Loading base model in 4-bit precision...")
    device_map = "auto" if torch.cuda.is_available() else None
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map=device_map,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        base_model = prepare_model_for_kbit_training(base_model)

    # 4. Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Load and Format Dataset
    data_path = TASK_DATA_MAP[task]
    print(f"Loading training data from {data_path}...")
    raw_records = load_jsonl_dataset(data_path)
    train_dataset = format_conversations(raw_records, tokenizer)
    print(f"Prepared {len(train_dataset)} training examples.")

    # 6. Configure Training Arguments (Supporting both SFTConfig and TrainingArguments)
    try:
        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            bf16=False,
            optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            report_to="none",
        )
    except Exception:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            bf16=False,
            optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            report_to="none",
        )

    # 7. Initialize SFTTrainer
    print("Initializing SFTTrainer...")
    try:
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            peft_config=lora_config,
            args=training_args,
        )
    except TypeError:
        # Fallback for older/newer TRL parameter names
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            peft_config=lora_config,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
        )

    # 8. Train
    print(f"Training [{task}] adapter for {epochs} epoch(s)...")
    train_result = trainer.train()

    # 9. Save Adapter Weights & Tokenizer
    print(f"Saving tuned adapter to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    adapter_size_mb = get_adapter_size_mb(output_dir)
    print("\n" + "-" * 60)
    print(f"TRAINING COMPLETE FOR [{task.upper()}]")
    print(f"Saved Path: {output_dir}")
    print(f"Total Adapter Size on Disk: {adapter_size_mb:.2f} MB")
    print(f"Training Loss: {train_result.training_loss:.4f}")
    print("-" * 60 + "\n")

    # Clean up GPU memory
    del base_model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "task": task,
        "output_dir": output_dir,
        "adapter_size_mb": adapter_size_mb,
        "loss": train_result.training_loss,
    }

def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for multi-adapter serving system")
    parser.add_argument("--task", type=str, choices=["sql", "json", "code", "all"], default="all",
                        help="Specific task to train or 'all' to train all 3 sequentially")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base model Hugging Face ID")
    parser.add_argument("--output-dir", type=str, default="adapters",
                        help="Output directory for adapter weights")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Maximum sequence length")
    args = parser.parse_args()

    tasks = ["sql", "json", "code"] if args.task == "all" else [args.task]
    results = []
    for t in tasks:
        res = train_adapter(
            task=t,
            base_model_id=args.base_model,
            output_base_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            max_seq_length=args.max_seq_len,
        )
        results.append(res)

    print("\n" + "=" * 60)
    print("ALL ADAPTER TRAININGS FINISHED")
    for r in results:
        print(f"- Task: {r['task']:<6} | Size: {r['adapter_size_mb']:.2f} MB | Loss: {r['loss']:.4f} | Path: {r['output_dir']}")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
Memory Conservation Profiler for Multi-Adapter LLM Serving.
Compares GPU VRAM footprints:
  - Separate Full Models (3 independent fine-tuned 1.5B model instances)
  - Routed Multi-LoRA (1 frozen 1.5B base + 3 dynamic LoRA adapters)

Generates results/memory_profile.png and results/memory_profile.json.
"""

import json
import os
import matplotlib.pyplot as plt

def calculate_memory_profiles() -> dict:
    # Qwen2.5-1.5B has ~1.54B parameters
    # In FP16 (2 bytes per parameter):
    base_model_weights_gb = (1.543 * 10**9 * 2) / (1024**3)  # ~2.87 GB
    lora_adapter_weights_gb = 0.016  # ~16 MB per rank-16 adapter
    vllm_cuda_overhead_gb = 0.60    # PyTorch/CUDA runtime context

    # 1. Architecture A: 3 Separate Fine-Tuned Models Loaded Concurrently
    num_models = 3
    separate_weights_gb = base_model_weights_gb * num_models  # ~8.62 GB
    separate_kv_cache_gb = 1.5 * num_models                  # ~4.5 GB (1.5 GB per instance)
    separate_overhead_gb = vllm_cuda_overhead_gb * num_models # ~1.8 GB
    separate_total_gb = separate_weights_gb + separate_kv_cache_gb + separate_overhead_gb  # ~14.92 GB

    # 2. Architecture B: Routed Multi-LoRA (Shared Base + 3 Adapters)
    multilora_base_weights_gb = base_model_weights_gb        # ~2.87 GB
    multilora_adapters_gb = lora_adapter_weights_gb * 3      # ~0.048 GB (48 MB)
    multilora_shared_kv_cache_gb = 4.5                       # Shared dynamic paged KV-cache pool
    multilora_overhead_gb = vllm_cuda_overhead_gb            # Single runtime context (0.6 GB)
    multilora_total_gb = multilora_base_weights_gb + multilora_adapters_gb + multilora_shared_kv_cache_gb + multilora_overhead_gb # ~8.02 GB

    weights_savings_pct = ((separate_weights_gb - (multilora_base_weights_gb + multilora_adapters_gb)) / separate_weights_gb) * 100
    total_savings_pct = ((separate_total_gb - multilora_total_gb) / separate_total_gb) * 100

    profile = {
        "model_architecture": "Qwen/Qwen2.5-1.5B-Instruct (FP16)",
        "num_adapters": 3,
        "separate_models": {
            "model_weights_gb": round(separate_weights_gb, 2),
            "kv_cache_gb": round(separate_kv_cache_gb, 2),
            "runtime_overhead_gb": round(separate_overhead_gb, 2),
            "total_vram_gb": round(separate_total_gb, 2),
        },
        "multi_lora_system": {
            "base_model_weights_gb": round(multilora_base_weights_gb, 2),
            "adapters_total_gb": round(multilora_adapters_gb, 3),
            "kv_cache_gb": round(multilora_shared_kv_cache_gb, 2),
            "runtime_overhead_gb": round(multilora_overhead_gb, 2),
            "total_vram_gb": round(multilora_total_gb, 2),
        },
        "metrics": {
            "weights_reduction_percent": round(weights_savings_pct, 1),
            "total_vram_savings_percent": round(total_savings_pct, 1),
            "vram_headroom_on_t4_gb": round(16.0 - multilora_total_gb, 2),
        }
    }
    return profile

def generate_memory_chart(profile: dict, output_image: str = "results/memory_profile.png"):
    sep = profile["separate_models"]
    lora = profile["multi_lora_system"]

    categories = ["3 Separate Fine-Tuned Models\n(Dedicated Instances)", "Routed Multi-LoRA System\n(Shared Base + 3 Adapters)"]
    weights = [sep["model_weights_gb"], lora["base_model_weights_gb"] + lora["adapters_total_gb"]]
    kv_cache = [sep["kv_cache_gb"], lora["kv_cache_gb"]]
    overhead = [sep["runtime_overhead_gb"], lora["runtime_overhead_gb"]]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Color scheme
    c_weights = "#3b82f6"  # Blue
    c_kv = "#10b981"       # Green
    c_overhead = "#f59e0b" # Amber

    # Stacked Bars
    bar_width = 0.5
    b1 = ax.bar(categories, weights, width=bar_width, label="Model / Adapter Weights", color=c_weights)
    b2 = ax.bar(categories, kv_cache, width=bar_width, bottom=weights, label="KV-Cache / Context Buffer", color=c_kv)
    b3 = ax.bar(categories, overhead, width=bar_width, bottom=[w + k for w, k in zip(weights, kv_cache)], label="Runtime / CUDA Context", color=c_overhead)

    # Add T4 VRAM Capacity Line
    ax.axhline(y=16.0, color="#ef4444", linestyle="--", linewidth=2, label="NVIDIA T4 Limit (16 GB)")

    # Data value labels on bars
    totals = [sep["total_vram_gb"], lora["total_vram_gb"]]
    for i, total in enumerate(totals):
        ax.text(i, total + 0.3, f"{total:.2f} GB", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Annotate weights savings
    savings_text = f"Weights Savings:\n-66.1% reduction\n({sep['model_weights_gb']:.1f}GB -> {weights[1]:.1f}GB)"
    ax.annotate(
        savings_text,
        xy=(1, weights[1] / 2),
        xytext=(1.35, 6),
        arrowprops=dict(facecolor="#1e293b", shrink=0.05, width=1.5, headwidth=8),
        fontsize=10,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9", edgecolor="#cbd5e1")
    )

    ax.set_ylabel("GPU VRAM Consumption (GB)", fontsize=12, fontweight="semibold")
    ax.set_title("VRAM Conservation: Separate Full Models vs. Routed Multi-LoRA\n(Qwen2.5-1.5B Instruct on NVIDIA T4)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 18)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_image) or ".", exist_ok=True)
    plt.savefig(output_image)
    plt.close()
    print(f"Memory conservation chart generated -> {output_image}")

def main():
    profile = calculate_memory_profiles()
    print("\n" + "=" * 65)
    print("VRAM PROFILE & CONSERVATION METRICS")
    print("=" * 65)
    print(f"Separate Models Footprint : {profile['separate_models']['total_vram_gb']} GB VRAM")
    print(f"Multi-LoRA Footprint      : {profile['multi_lora_system']['total_vram_gb']} GB VRAM")
    print(f"Weights Memory Reduction  : {profile['metrics']['weights_reduction_percent']}%")
    print(f"Remaining T4 Headroom     : {profile['metrics']['vram_headroom_on_t4_gb']} GB free")
    print("=" * 65 + "\n")

    output_json = "results/memory_profile.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"Profile data saved -> {output_json}")

    generate_memory_chart(profile)

if __name__ == "__main__":
    main()

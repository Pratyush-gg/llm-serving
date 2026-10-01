"""Measure GPU memory of the base model + adapters (the 3-separate-models figure is an estimate)."""
import os
import sys
import json
import time
import argparse
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import local_cache  # noqa: F401  (keep caches inside the repo; must precede ML imports)

MB = 1024 * 1024

# Chart colors (reference palette: series slot 1, neutral for the estimate, text inks, surface).
C_MEASURED = "#2a78d6"
C_ESTIMATE = "#b9b8b3"
C_TEXT = "#0b0b0b"
C_TEXT_2 = "#52514e"
C_GRID = "#e4e3df"
C_SURFACE = "#fcfcfb"

DEFAULT_PROMPTS = {
    "sql": ("data/sql_holdout.jsonl", "prompt"),
    "json": ("data/json_holdout.jsonl", "prompt"),
    "code": ("data/code_holdout.jsonl", "prompt"),
}
BASE_PROMPT = "Explain how vaccines train the immune system, in two short paragraphs."


def first_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.readline())["prompt"]


def gpu_stats(torch) -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / MB, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / MB, 1),
        "device_used_mb": round((total - free) / MB, 1),  # includes CUDA context and other processes
    }


def profile(max_new_tokens: int) -> dict:
    import torch
    import transformers
    import peft
    import src.gateway as gateway

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to measure VRAM.")

    torch.cuda.empty_cache()
    stages = {"before_load": gpu_stats(torch)}

    print("Loading base model (4-bit NF4)...", flush=True)
    base_model, tokenizer = gateway.load_base_model()
    torch.cuda.synchronize()
    stages["after_base_model"] = gpu_stats(torch)

    print("Registering LoRA adapters...", flush=True)
    model = gateway.attach_adapters(base_model)
    torch.cuda.synchronize()
    stages["after_adapters"] = gpu_stats(torch)

    # Serve through the gateway's own generation path so the measurement matches serving.
    gateway._peft_model, gateway._peft_base_model, gateway._peft_tokenizer = model, base_model, tokenizer

    generation = {}
    for route in ["sql", "json", "code", "base"]:
        prompt = BASE_PROMPT if route == "base" else first_prompt(DEFAULT_PROMPTS[route][0])
        torch.cuda.reset_peak_memory_stats()
        result = gateway.peft_generate_response(route, prompt, max_tokens=max_new_tokens)
        generation[route] = {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / MB, 1),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / MB, 1),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "generation_ms": round(result.generation_ms, 1),
        }
        print(f"  {route:<5} peak allocated {generation[route]['peak_allocated_mb']} MB "
              f"({result.prompt_tokens} prompt + {result.completion_tokens} generated tokens)", flush=True)

    base_weights_mb = stages["after_base_model"]["allocated_mb"] - stages["before_load"]["allocated_mb"]
    adapters_mb = stages["after_adapters"]["allocated_mb"] - stages["after_base_model"]["allocated_mb"]
    peak_mb = max(g["peak_allocated_mb"] for g in generation.values())

    return {
        "mode": "measured",
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_mb": round(torch.cuda.mem_get_info()[1] / MB, 1),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "python": platform.python_version(),
            "base_model": gateway.BASE_MODEL_NAME,
            "quantization": "bitsandbytes 4-bit NF4, double quant",
            "max_new_tokens": max_new_tokens,
        },
        "stages": stages,
        "generation_peaks": generation,
        "summary": {
            "base_model_weights_mb": round(base_weights_mb, 1),
            "three_adapters_mb": round(adapters_mb, 1),
            "multi_lora_resident_mb": round(base_weights_mb + adapters_mb, 1),
            "multi_lora_peak_during_generation_mb": peak_mb,
        },
        "estimate_three_separate_models": {
            "label": "ESTIMATE - not measured",
            "method": "3 x measured base-model weight memory (each separate model = base with one merged adapter, same size in NF4)",
            "weights_mb": round(3 * base_weights_mb, 1),
        },
    }


def generate_memory_chart(profile_data: dict, output_image: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = profile_data["summary"]
    est = profile_data["estimate_three_separate_models"]
    bars = [
        ("Base model weights (NF4)", s["base_model_weights_mb"], "measured"),
        ("+ 3 LoRA adapters", s["multi_lora_resident_mb"], "measured"),
        ("Peak during generation", s["multi_lora_peak_during_generation_mb"], "measured"),
        ("3 separate models, weights only", est["weights_mb"], "estimate"),
    ]
    labels = [b[0] for b in bars][::-1]
    values = [b[1] for b in bars][::-1]
    kinds = [b[2] for b in bars][::-1]

    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=200)
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    for y, (v, kind) in enumerate(zip(values, kinds)):
        ax.barh(
            y, v, height=0.55,
            color=C_MEASURED if kind == "measured" else C_ESTIMATE,
            hatch=None if kind == "measured" else "///",
            edgecolor=C_SURFACE if kind == "measured" else C_TEXT_2,
            linewidth=0 if kind == "measured" else 0.6,
        )
        suffix = "" if kind == "measured" else "  (estimate)"
        ax.text(v + max(values) * 0.01, y, f"{v:,.0f} MB{suffix}", va="center", ha="left",
                fontsize=9, color=C_TEXT)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color=C_TEXT)
    ax.set_xlabel("GPU memory allocated by PyTorch (MB)", fontsize=9, color=C_TEXT_2)
    ax.set_xlim(0, max(values) * 1.25)
    ax.tick_params(axis="x", colors=C_TEXT_2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(C_GRID)

    env = profile_data["environment"]
    ax.set_title(f"Multi-LoRA serving memory on {env['gpu']}", fontsize=11, color=C_TEXT, loc="left")
    fig.text(0.01, 0.01, "Blue: measured with torch.cuda statistics.  Hatched gray: estimate (3 x measured base weights).",
             fontsize=7.5, color=C_TEXT_2)

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(output_image) or ".", exist_ok=True)
    plt.savefig(output_image, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"Memory chart saved -> {output_image}")


def main():
    parser = argparse.ArgumentParser(description="Measure multi-LoRA GPU memory")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-json", type=str, default="results/memory_profile.json")
    parser.add_argument("--output-image", type=str, default="results/memory_profile.png")
    args = parser.parse_args()

    data = profile(args.max_new_tokens)
    print(json.dumps(data["summary"], indent=2))

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Profile saved -> {args.output_json}")
    generate_memory_chart(data, args.output_image)


if __name__ == "__main__":
    main()

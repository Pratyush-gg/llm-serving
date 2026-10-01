"""Base model vs. LoRA adapters on the same test sets, with paired confidence intervals."""
import json
import os
import sys
import argparse
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.run_eval import run_evaluation, TASKS, DEFAULT_TASKS
from eval.stats import paired_delta_ci

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def compare(baseline_metrics: Dict, tuned_metrics: Dict) -> Dict:
    summary = {}
    for task, base in baseline_metrics.items():
        tuned = tuned_metrics.get(task)
        key = base.get("primary_metric") or TASKS.get(task, {}).get("primary")
        if not tuned or key is None:
            continue
        per_key = "per_example_leaf_f1" if key == "leaf_f1" else "per_example_correct"
        b_vals, t_vals = base.get(per_key), tuned.get(per_key)
        same_examples = base.get("example_ids") == tuned.get("example_ids")
        summary[task] = {
            "metric": key,
            "baseline": base.get(key),
            "tuned": tuned.get(key),
            "baseline_ci95": base.get(f"{key}_ci95"),
            "tuned_ci95": tuned.get(f"{key}_ci95"),
            "delta": round(tuned[key] - base[key], 4) if base.get(key) is not None and tuned.get(key) is not None else None,
            "delta_ci95": paired_delta_ci(b_vals, t_vals) if (b_vals and t_vals and same_examples) else None,
            "n": base.get("total_samples"),
        }
    return summary


def run_baseline_evaluation(
    endpoint_url: str = None,
    base_model_name: str = DEFAULT_BASE_MODEL,
    tuned_results_path: str = "results/correctness_results.json",
    output_path: str = "results/baseline_vs_tuned.json",
    assume_yes: bool = False,
    resume: bool = False,
) -> Dict:
    print("\n" + "=" * 75)
    print(f"RUNNING ZERO-SHOT BASELINE EVALUATION: [{base_model_name}]")
    print("=" * 75)

    baseline_metrics = {}
    if not endpoint_url:
        raise ValueError("Must provide --endpoint-url")
    for task in DEFAULT_TASKS:
        if not os.path.exists(TASKS[task]["path"]):
            print(f"[skip] {task}: {TASKS[task]['path']} not found")
            continue
        baseline_metrics[task] = run_evaluation(
            task=task, backend="endpoint", endpoint_url=endpoint_url,
            model_name=base_model_name, assume_yes=assume_yes, resume=resume,
        )

    tuned_metrics = {}
    if os.path.exists(tuned_results_path):
        with open(tuned_results_path, "r", encoding="utf-8") as f:
            tuned_metrics = json.load(f)

    comparison = {
        "base_model": base_model_name,
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "summary": compare(baseline_metrics, tuned_metrics),
    }

    print("\n" + "=" * 96)
    print("BASE MODEL VS. LORA ADAPTERS (95% CIs; delta CI is paired over the same examples)")
    print("=" * 96)
    print(f"{'Task':<16} | {'Metric':<18} | {'Base':>7} | {'LoRA':>7} | {'Delta':>8} | {'Delta 95% CI':<18} | n")
    print("-" * 96)
    for task, s in comparison["summary"].items():
        fmt = lambda v: f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"
        d = s["delta_ci95"]
        d_ci = f"[{d['low'] * 100:+.1f}, {d['high'] * 100:+.1f}]" if d else "n/a"
        delta = f"{s['delta'] * 100:+.1f}" if s["delta"] is not None else "n/a"
        print(f"{task:<16} | {s['metric']:<18} | {fmt(s['baseline']):>7} | {fmt(s['tuned']):>7} | {delta:>8} | {d_ci:<18} | {s['n']}")
    print("=" * 96 + "\n")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison report saved to {output_path}")
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Zero-shot baseline evaluation and comparison")
    parser.add_argument("--endpoint-url", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="Generation endpoint (gateway /v1/chat or vLLM /v1/chat/completions)")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base model ID")
    parser.add_argument("--tuned-results", type=str, default="results/correctness_results.json",
                        help="Path to tuned evaluation results JSON")
    parser.add_argument("--output", type=str, default="results/baseline_vs_tuned.json",
                        help="Output path for comparison JSON")
    parser.add_argument("--yes", action="store_true",
                        help="Confirm running model-generated code (code tasks) without an interactive prompt")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse outputs already saved in results/raw_outputs (continue an interrupted run)")
    args = parser.parse_args()

    run_baseline_evaluation(
        endpoint_url=args.endpoint_url,
        base_model_name=args.base_model,
        tuned_results_path=args.tuned_results,
        output_path=args.output,
        assume_yes=args.yes,
        resume=args.resume,
    )

if __name__ == "__main__":
    main()

import json
import os
import sys
import argparse
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.run_eval import run_evaluation, load_holdout_examples
from eval.sql_eval import evaluate_sql
from eval.json_eval import evaluate_json
from eval.code_eval import evaluate_code

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def generate_mock_baseline_response(task: str, prompt: str) -> str:
    """
    Simulates zero-shot base model behavior without specialized fine-tuning:
    - Base models produce conversational chatter, markdown fences, partial SQL,
      sometimes non-compliant JSON keys, and unconstrained code.
    """
    if task == "sql":
        return "Here is your SQL query:\n```sql\nSELECT * FROM table;\n```"
    elif task == "json":
        return "I extracted the details:\n```json\n{\"customer\": \"Unknown\", \"order\": \"None\"}\n```"
    elif task == "code":
        return "```python\ndef solution():\n    pass\n```"
    return ""

def run_baseline_evaluation(
    endpoint_url: str = None,
    base_model_name: str = DEFAULT_BASE_MODEL,
    use_mock: bool = False,
    tuned_results_path: str = "results/correctness_results.json",
    output_path: str = "results/baseline_vs_tuned.json",
) -> Dict:
    print("\n" + "=" * 75)
    print(f"RUNNING ZERO-SHOT BASELINE EVALUATION: [{base_model_name}]")
    print("=" * 75)

    baseline_metrics = {}

    for task in ["sql", "json", "code"]:
        print(f"\nEvaluating zero-shot baseline on task [{task.upper()}]...")
        if use_mock:
            examples = load_holdout_examples(f"data/{task}_holdout.jsonl")
            if task == "sql":
                res = evaluate_sql(examples, lambda p: generate_mock_baseline_response("sql", p))
            elif task == "json":
                res = evaluate_json(examples, lambda p: generate_mock_baseline_response("json", p))
            elif task == "code":
                res = evaluate_code(examples, lambda p: generate_mock_baseline_response("code", p))
            res["backend"] = "mock_zero_shot"
            res["model_name"] = f"{base_model_name} (zero-shot mock)"
            baseline_metrics[task] = res
        else:
            if not endpoint_url:
                raise ValueError("Must provide --endpoint-url when not running with --mock")
            res = run_evaluation(
                task=task,
                backend="endpoint",
                endpoint_url=endpoint_url,
                model_name=base_model_name,
            )
            baseline_metrics[task] = res

    # Load tuned results if available
    tuned_metrics = {}
    if os.path.exists(tuned_results_path):
        with open(tuned_results_path, "r", encoding="utf-8") as f:
            tuned_metrics = json.load(f)

    # Build comparison summary
    comparison = {
        "base_model": base_model_name,
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "summary": {},
    }

    # Extract primary metric comparison
    tasks_info = {
        "sql": ("exact_match_rate", "Exact Match Rate"),
        "json": ("schema_valid_rate", "Schema Validity Rate"),
        "code": ("pass_at_1", "Pass@1 Rate"),
    }

    print("\n" + "=" * 80)
    print("DAY 4: ZERO-SHOT BASELINE VS. TUNED ADAPTERS COMPARISON")
    print("=" * 80)
    print(f"{'Task':<8} | {'Primary Metric':<24} | {'Zero-Shot Base':<16} | {'Tuned LoRA':<14} | {'Delta':<10}")
    print("-" * 80)

    for task, (metric_key, metric_name) in tasks_info.items():
        base_val = baseline_metrics.get(task, {}).get(metric_key, 0.0)
        tuned_val = tuned_metrics.get(task, {}).get(metric_key, 0.0)
        delta = tuned_val - base_val

        comparison["summary"][task] = {
            "metric": metric_name,
            "baseline": base_val,
            "tuned": tuned_val,
            "delta": round(delta, 4),
        }

        base_str = f"{base_val * 100:.1f}%"
        tuned_str = f"{tuned_val * 100:.1f}%"
        delta_str = f"{'+' if delta >= 0 else ''}{delta * 100:.1f}%"

        print(f"{task.upper():<8} | {metric_name:<24} | {base_str:<16} | {tuned_str:<14} | {delta_str:<10}")

    print("=" * 80 + "\n")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison report saved to {output_path}")

    return comparison

def main():
    parser = argparse.ArgumentParser(description="Zero-shot baseline evaluation and comparison")
    parser.add_argument("--endpoint-url", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="vLLM endpoint URL")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base model ID")
    parser.add_argument("--mock", action="store_true",
                        help="Run mock zero-shot evaluation pipeline test without live GPU")
    parser.add_argument("--tuned-results", type=str, default="results/correctness_results.json",
                        help="Path to tuned evaluation results JSON")
    parser.add_argument("--output", type=str, default="results/baseline_vs_tuned.json",
                        help="Output path for comparison JSON")
    args = parser.parse_args()

    run_baseline_evaluation(
        endpoint_url=args.endpoint_url,
        base_model_name=args.base_model,
        use_mock=args.mock,
        tuned_results_path=args.tuned_results,
        output_path=args.output,
    )

if __name__ == "__main__":
    main()

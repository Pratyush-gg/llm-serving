import json
import os
import time
import argparse
from typing import Callable, Dict, List

import requests

from eval.sql_eval import evaluate_sql
from eval.json_eval import evaluate_json
from eval.code_eval import evaluate_code

DATASET_MAP = {
    "sql": "data/sql_holdout.jsonl",
    "json": "data/json_holdout.jsonl",
    "code": "data/code_holdout.jsonl",
}

def load_holdout_examples(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Holdout dataset not found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def create_endpoint_generator(
    endpoint_url: str,
    model_name: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> Callable[[str], str]:
    """Factory for HTTP endpoint generator (vLLM or FastAPI gateway)."""
    headers = {"Content-Type": "application/json"}

    def generate_fn(prompt: str) -> str:
        # Check if calling FastAPI gateway (/v1/chat) or vLLM (/v1/chat/completions)
        if endpoint_url.rstrip("/").endswith("/v1/chat"):
            payload = {"prompt": prompt}
            resp = requests.post(endpoint_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        else:
            # OpenAI compatible endpoint
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            resp = requests.post(endpoint_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    return generate_fn

def create_gold_generator(task: str, examples: List[Dict]) -> Callable[[str], str]:
    """Sanity check generator returning gold references."""
    lookup = {}
    for ex in examples:
        if task == "sql":
            lookup[ex["prompt"]] = ex["gold_sql"]
        elif task == "json":
            lookup[ex["prompt"]] = json.dumps(ex["gold_json"])
        elif task == "code":
            lookup[ex["prompt"]] = ex["gold_code"]
    return lambda p: lookup[p]

def run_evaluation(
    task: str,
    backend: str,
    endpoint_url: str = None,
    model_name: str = None,
    data_path: str = None,
) -> Dict:
    path = data_path or DATASET_MAP[task]
    examples = load_holdout_examples(path)
    print(f"\nEvaluating [{task.upper()}] task on {len(examples)} held-out examples from {path}...")

    t0 = time.perf_counter()
    if backend == "gold":
        gen_fn = create_gold_generator(task, examples)
    elif backend == "endpoint":
        if not endpoint_url:
            raise ValueError("Must provide --endpoint-url when backend is 'endpoint'")
        gen_fn = create_endpoint_generator(endpoint_url, model_name or task)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if task == "sql":
        metrics = evaluate_sql(examples, gen_fn)
    elif task == "json":
        metrics = evaluate_json(examples, gen_fn)
    elif task == "code":
        metrics = evaluate_code(examples, gen_fn)
    else:
        raise ValueError(f"Unsupported task: {task}")

    elapsed_s = time.perf_counter() - t0
    metrics["eval_time_seconds"] = round(elapsed_s, 2)
    metrics["backend"] = backend
    metrics["model_name"] = model_name or ("gold_reference" if backend == "gold" else "custom")
    return metrics

def print_summary_table(all_results: Dict[str, Dict]):
    print("\n" + "=" * 75)
    print("DAY 3: CORRECTNESS EVALUATION SUMMARY TABLE")
    print("=" * 75)
    print(f"{'Task':<8} | {'Model / Backend':<20} | {'Primary Metric':<24} | {'Secondary Metric':<16}")
    print("-" * 75)

    for task, res in all_results.items():
        backend_label = f"{res.get('model_name', res.get('backend'))}"[:20]
        if task == "sql":
            p_metric = f"Exact Match: {res.get('exact_match_rate', 0.0) * 100:.1f}%"
            s_metric = f"Exec: {res.get('execution_rate', 0.0) * 100:.1f}%"
        elif task == "json":
            p_metric = f"Schema Valid: {res.get('schema_valid_rate', 0.0) * 100:.1f}%"
            s_metric = f"Field: {res.get('field_accuracy', 0.0) * 100:.1f}%"
        elif task == "code":
            p_metric = f"Pass@1: {res.get('pass_at_1', 0.0) * 100:.1f}%"
            s_metric = f"Total: {res.get('total_samples', 0)}"
        else:
            p_metric = str(res)
            s_metric = ""

        print(f"{task.upper():<8} | {backend_label:<20} | {p_metric:<24} | {s_metric:<16}")

    print("=" * 75 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Unified Correctness Evaluation Runner")
    parser.add_argument("--task", type=str, choices=["sql", "json", "code", "all"], default="all",
                        help="Specific task to evaluate or 'all' to evaluate SQL, JSON, and Code")
    parser.add_argument("--backend", type=str, choices=["gold", "endpoint"], default="gold",
                        help="Generation backend: 'gold' (reference test) or 'endpoint' (live vLLM/FastAPI)")
    parser.add_argument("--endpoint-url", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="URL of the LLM generation endpoint")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model / adapter identifier (e.g. 'sql-adapter' or 'Qwen/Qwen2.5-1.5B-Instruct')")
    parser.add_argument("--output-json", type=str, default="results/correctness_results.json",
                        help="File path to save JSON evaluation results")
    args = parser.parse_args()

    tasks = ["sql", "json", "code"] if args.task == "all" else [args.task]
    all_results = {}

    for t in tasks:
        all_results[t] = run_evaluation(
            task=t,
            backend=args.backend,
            endpoint_url=args.endpoint_url,
            model_name=args.model_name,
        )

    print_summary_table(all_results)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results successfully saved to {args.output_json}")

if __name__ == "__main__":
    main()

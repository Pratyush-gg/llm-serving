"""Evaluate the LoRA adapters on the test sets in data/eval/."""
import json
import os
import re
import time
import argparse
from typing import Callable, Dict, List

import requests

from eval.sql_eval import evaluate_sql
from eval.json_eval import evaluate_json, evaluate_json_schema
from eval.code_eval import evaluate_code
from eval.stats import bootstrap_ci

# name -> data file, adapter to evaluate, scorer, generation budget, headline metric
TASKS = {
    "sql": {"path": "data/eval/sql_gretel.jsonl", "adapter": "sql", "kind": "sql",
            "max_tokens": 256, "primary": "execution_accuracy"},
    "json": {"path": "data/eval/json_inscope.jsonl", "adapter": "json", "kind": "json",
             "max_tokens": 256, "primary": "exact_match_rate"},
    "json_paraloq": {"path": "data/eval/json_paraloq.jsonl", "adapter": "json", "kind": "json_schema",
                     "max_tokens": 1024, "primary": "leaf_f1"},
    "code_humaneval": {"path": "data/eval/code_humaneval.jsonl", "adapter": "code", "kind": "code",
                       "max_tokens": 512, "primary": "pass_at_1"},
    "code_mbpp": {"path": "data/eval/code_mbpp.jsonl", "adapter": "code", "kind": "code",
                  "max_tokens": 512, "primary": "pass_at_1"},
}
DEFAULT_TASKS = list(TASKS)

# Accept vLLM-style model names when talking to the gateway, which expects adapter keys.
GATEWAY_ADAPTER_NAMES = {
    "sql-adapter": "sql",
    "json-adapter": "json",
    "code-adapter": "code",
    "Qwen/Qwen2.5-1.5B-Instruct": "base",
}


def load_holdout_examples(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation dataset not found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def create_endpoint_generator(
    endpoint_url: str,
    model_name: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: float = 300.0,
) -> Callable[[str], str]:
    """Factory for HTTP endpoint generator (vLLM or FastAPI gateway)."""
    headers = {"Content-Type": "application/json"}

    def generate_fn(prompt: str) -> str:
        # Check if calling FastAPI gateway (/v1/chat) or vLLM (/v1/chat/completions)
        if endpoint_url.rstrip("/").endswith("/v1/chat"):
            # Force the adapter under evaluation; otherwise the router would pick one.
            payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                       "force_adapter": GATEWAY_ADAPTER_NAMES.get(model_name, model_name),
                       "enable_cascade": False}
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


def create_gold_generator(kind: str, examples: List[Dict]) -> Callable[[str], str]:
    """Sanity check generator returning gold references (every scorer should give ~100%)."""
    lookup = {}
    for ex in examples:
        if kind == "sql":
            lookup[ex["prompt"]] = ex["gold_sql"]
        elif kind in ("json", "json_schema"):
            lookup[ex["prompt"]] = json.dumps(ex["gold_json"])
        elif kind == "code":
            lookup[ex["prompt"]] = ex["gold_code"]
    return lambda p: lookup[p]


def run_evaluation(
    task: str,
    backend: str,
    endpoint_url: str = None,
    model_name: str = None,
    data_path: str = None,
    assume_yes: bool = False,
    raw_output_dir: str = "results/raw_outputs",
    resume: bool = False,
) -> Dict:
    """model_name: adapter/model to force (default: the task's own adapter; 'base' for the base model)."""
    cfg = TASKS[task]
    path = data_path or cfg["path"]
    examples = load_holdout_examples(path)
    model = model_name or cfg["adapter"]
    print(f"\nEvaluating [{task}] with model '{model}' on {len(examples)} examples from {path}...")

    t0 = time.perf_counter()
    if backend == "gold":
        gen_fn = create_gold_generator(cfg["kind"], examples)
    elif backend == "endpoint":
        if not endpoint_url:
            raise ValueError("Must provide --endpoint-url when backend is 'endpoint'")
        gen_fn = create_endpoint_generator(endpoint_url, model, max_tokens=cfg["max_tokens"])
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Save each output as soon as it exists (audit, re-score, resume).
    raw_path, cached, raw_file = None, {}, None
    if backend == "endpoint" and raw_output_dir:
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", GATEWAY_ADAPTER_NAMES.get(model, model))
        os.makedirs(raw_output_dir, exist_ok=True)
        raw_path = os.path.join(raw_output_dir, f"{label}_{task}.jsonl")
        if resume and os.path.exists(raw_path):
            cached = {r["prompt"]: r["output"] for r in load_holdout_examples(raw_path)}
            print(f"Resuming: {len(cached)}/{len(examples)} outputs already in {raw_path}")
        raw_file = open(raw_path, "a" if resume else "w", encoding="utf-8")

    def recording_gen_fn(prompt: str) -> str:
        if prompt in cached:
            return cached[prompt]
        out = gen_fn(prompt)
        if raw_file:
            raw_file.write(json.dumps({"prompt": prompt, "output": out}, ensure_ascii=False) + "\n")
            raw_file.flush()
        return out

    kind = cfg["kind"]
    if kind == "sql":
        metrics = evaluate_sql(examples, recording_gen_fn)
    elif kind == "json":
        metrics = evaluate_json(examples, recording_gen_fn)
    elif kind == "json_schema":
        metrics = evaluate_json_schema(examples, recording_gen_fn)
    elif kind == "code":
        metrics = evaluate_code(examples, recording_gen_fn, assume_yes=assume_yes)
    else:
        raise ValueError(f"Unsupported task kind: {kind}")

    elapsed_s = time.perf_counter() - t0
    metrics["eval_time_seconds"] = round(elapsed_s, 2)
    metrics["backend"] = backend
    metrics["model_name"] = model if backend == "endpoint" else "gold_reference"
    metrics["dataset"] = path
    metrics["primary_metric"] = cfg["primary"]
    metrics["example_ids"] = [ex.get("id") or ex.get("name") for ex in examples]

    # Per-format breakdown (e.g. SQL schema shown as CREATE TABLE / compact / prose).
    per_example = metrics.get("per_example_correct")
    if per_example and all("schema_style" in ex for ex in examples):
        by_style = {}
        for ex, ok in zip(examples, per_example):
            by_style.setdefault(ex["schema_style"], []).append(ok)
        metrics["by_schema_style"] = {s: bootstrap_ci(v) for s, v in by_style.items()}

    if raw_file:
        raw_file.close()
        metrics["raw_outputs_file"] = raw_path.replace("\\", "/")
    return metrics


def print_summary_table(all_results: Dict[str, Dict]):
    print("\n" + "=" * 86)
    print("CORRECTNESS EVALUATION SUMMARY")
    print("=" * 86)
    print(f"{'Task':<16} | {'Model':<8} | {'Headline metric':<22} | {'Value':>7} | {'95% CI':<17} | n")
    print("-" * 86)
    for task, res in all_results.items():
        key = res.get("primary_metric")
        val = res.get(key)
        ci = res.get(f"{key}_ci95")
        ci_str = f"[{ci['low'] * 100:.1f}, {ci['high'] * 100:.1f}]" if ci else "n/a"
        val_str = f"{val * 100:.1f}%" if isinstance(val, (int, float)) else "n/a"
        print(f"{task:<16} | {str(res.get('model_name'))[:8]:<8} | {key:<22} | {val_str:>7} | {ci_str:<17} | {res.get('total_samples')}")
    print("=" * 86 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Unified Correctness Evaluation Runner")
    parser.add_argument("--task", type=str, choices=list(TASKS) + ["all"], default="all",
                        help=f"Task to evaluate, or 'all' = {', '.join(DEFAULT_TASKS)}")
    parser.add_argument("--backend", type=str, choices=["gold", "endpoint"], default="gold",
                        help="Generation backend: 'gold' (reference test) or 'endpoint' (live vLLM/FastAPI)")
    parser.add_argument("--endpoint-url", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="URL of the LLM generation endpoint")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Adapter/model to force for every task (default: each task's own adapter; "
                             "'base' or 'Qwen/Qwen2.5-1.5B-Instruct' for the base model)")
    parser.add_argument("--output-json", type=str, default="results/correctness_results.json",
                        help="File path to save JSON evaluation results")
    parser.add_argument("--yes", action="store_true",
                        help="Confirm running model-generated code (code tasks) without an interactive prompt")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse outputs already saved in results/raw_outputs (continue an interrupted run)")
    args = parser.parse_args()

    tasks = DEFAULT_TASKS if args.task == "all" else [args.task]
    all_results = {}
    for t in tasks:
        if not os.path.exists(TASKS[t]["path"]):
            print(f"\n[skip] {t}: {TASKS[t]['path']} not found")
            continue
        all_results[t] = run_evaluation(
            task=t,
            backend=args.backend,
            endpoint_url=args.endpoint_url,
            model_name=args.model_name,
            assume_yes=args.yes,
            resume=args.resume,
        )

    print_summary_table(all_results)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results successfully saved to {args.output_json}")


if __name__ == "__main__":
    main()

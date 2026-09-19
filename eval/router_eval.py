import json
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import confusion_matrix, classification_report
from src.router import get_router

OUT_OF_DOMAIN_QUERIES = [
    "What is the theory of general relativity?",
    "Can you write a poem about autumn leaves falling?",
    "Explain the historical significance of the Roman Empire.",
    "Give me 5 creative ideas for a healthy breakfast.",
    "How does photosynthesis work in plants?",
    "Translate 'Good morning, how are you?' to Spanish.",
    "What are the main differences between classical and operant conditioning?",
    "Summarize the plot of Hamlet in two sentences.",
    "Why is the sky blue during daytime?",
    "Who painted the Mona Lisa?",
    "What are the benefits of cardiovascular exercise?",
    "Can you explain the philosophical concept of existentialism?",
    "What caused the Great Depression in 1929?",
    "Recommend three must-read science fiction novels.",
    "How do airplanes generate lift to fly?",
    "What is the distance between the Earth and the Moon?",
    "Describe the culinary history of pizza in Italy.",
    "Who was the first person to walk on the Moon?",
    "What is the difference between climate and weather?",
    "Give me advice for improving public speaking skills.",
]

def load_task_queries(jsonl_path: str, count: int = 20) -> list[str]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [r["prompt"] for r in records[:count]]

def evaluate_router(threshold: float = 0.55, output_json: str = "results/router_eval.json", strategy: str = "auto") -> dict:
    router = get_router(strategy=strategy, threshold=threshold)

    # 1. Assemble balanced benchmark dataset (80 queries)
    sql_prompts = load_task_queries("data/sql_holdout.jsonl", 20)
    json_prompts = load_task_queries("data/json_holdout.jsonl", 20)
    code_prompts = load_task_queries("data/code_holdout.jsonl", 20)
    base_prompts = OUT_OF_DOMAIN_QUERIES[:20]

    test_samples = (
        [(p, "sql") for p in sql_prompts] +
        [(p, "json") for p in json_prompts] +
        [(p, "code") for p in code_prompts] +
        [(p, "base") for p in base_prompts]
    )

    y_true = []
    y_pred = []
    latencies_ms = []
    detailed_records = []

    print(f"\nEvaluating Semantic Router on {len(test_samples)} held-out queries (threshold={threshold})...")

    for prompt, true_label in test_samples:
        result = router.route_detailed(prompt)
        pred_label = result["route"]
        lat_ms = result["latency_ms"]

        y_true.append(true_label)
        y_pred.append(pred_label)
        latencies_ms.append(lat_ms)

        detailed_records.append({
            "prompt": prompt[:80] + "...",
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": result["confidence"],
            "latency_ms": lat_ms,
        })

    # 2. Compute Metrics
    labels = ["sql", "json", "code", "base"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    p50_lat = float(np.percentile(latencies_ms, 50))
    p95_lat = float(np.percentile(latencies_ms, 95))
    avg_lat = float(np.mean(latencies_ms))

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    # 3. Print Results
    print("\n" + "=" * 65)
    print("ROUTER CONFUSION MATRIX (True \\ Predicted)")
    print("=" * 65)
    header = f"{'':<10}" + "".join([f"{l:>12}" for l in labels])
    print(header)
    print("-" * 65)
    for i, row_label in enumerate(labels):
        row_str = f"{row_label:<10}" + "".join([f"{cm[i][j]:>12}" for j in range(len(labels))])
        print(row_str)
    print("=" * 65)

    print(f"\nOverall Routing Accuracy : {acc * 100:.2f}% ({np.sum(np.diag(cm))}/{len(test_samples)})")
    print(f"P50 Routing Latency      : {p50_lat:.2f} ms")
    print(f"P95 Routing Latency      : {p95_lat:.2f} ms")
    print(f"Average Routing Latency  : {avg_lat:.2f} ms")

    summary_results = {
        "strategy": getattr(router, "strategy_name", strategy),
        "total_queries": len(test_samples),
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "avg_latency_ms": round(avg_lat, 2),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    if output_json:
        os.makedirs(os.path.dirname(output_path := output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2)
        print(f"\nRouter evaluation results saved to {output_json}")

    return summary_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Semantic / Learned Router")
    parser.add_argument("--threshold", type=float, default=0.55, help="Confidence cutoff threshold")
    parser.add_argument("--strategy", type=str, default="auto", help="Router strategy ('centroid', 'learned', or 'auto')")
    parser.add_argument("--output", type=str, default="results/router_eval.json", help="Output JSON path")
    args = parser.parse_args()
    evaluate_router(threshold=args.threshold, output_json=args.output, strategy=args.strategy)

"""
Router evaluation on data/router_testset.jsonl (built by scripts/build_router_testset.py).

Reports accuracy separately for clear-domain and ambiguous prompts, a 4x4 confusion matrix,
per-class precision/recall/F1 and routing latency. No generation is involved.
"""
import json
import os
import sys
import argparse
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import confusion_matrix, classification_report
from src.router import get_router
from eval.stats import bootstrap_ci

LABELS = ["sql", "json", "code", "base"]
DEFAULT_TESTSET = "data/router_testset.jsonl"


def load_testset(path: str, split: str = "all") -> List[Dict]:
    """split: 'test' (reported results), 'calibration' (threshold tuning) or 'all'."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Build it with: python scripts/build_router_testset.py")
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if split != "all":
        rows = [r for r in rows if r.get("split") == split]
        if not rows:
            raise ValueError(f"No rows with split='{split}' in {path}; rebuild the test set.")
    return rows


def summarize(rows: List[Dict]) -> Dict:
    y_true = [r["label"] for r in rows]
    y_pred = [r["predicted"] for r in rows]
    correct = [int(t == p) for t, p in zip(y_true, y_pred)]
    return {
        "n": len(rows),
        "accuracy": round(float(np.mean(correct)), 4) if rows else None,
        "accuracy_ci95": bootstrap_ci(correct),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }


def evaluate_router(strategy: str = "auto", threshold: Optional[float] = None,
                    testset_path: str = DEFAULT_TESTSET, output_json: Optional[str] = None,
                    split: str = "test", model_path: Optional[str] = None) -> Dict:
    kwargs = {"model_path": model_path} if model_path else {}
    router = get_router(strategy=strategy, threshold=threshold, **kwargs)
    strategy_name = getattr(router, "strategy_name", strategy)
    testset = load_testset(testset_path, split)

    for _ in range(3):  # warm up the embedding session
        router.route_detailed("warm up")

    rows = []
    for item in testset:
        res = router.route_detailed(item["prompt"])
        rows.append({
            "id": item.get("id") or item.get("source_id"),
            "prompt": item["prompt"][:120],
            "label": item["label"],
            "category": item["category"],
            "schema_style": item.get("schema_style"),
            "source": item.get("source"),
            "predicted": res["route"],
            "confidence": res["confidence"],
            "scores": res.get("scores", {}),
            "latency_ms": res["latency_ms"],
        })

    latencies = [r["latency_ms"] for r in rows]
    result = {
        "strategy": strategy_name,
        "threshold": getattr(router, "threshold", threshold),
        "model_path": getattr(router, "model_path", None),
        "testset": testset_path,
        "split": split,
        "overall": summarize(rows),
        "clear": summarize([r for r in rows if r["category"] == "clear"]),
        "ambiguous": summarize([r for r in rows if r["category"] == "ambiguous"]),
        "latency_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "mean": round(float(np.mean(latencies)), 2),
        },
        # Clear SQL prompts by how the schema is shown; exposes a CREATE TABLE keyword shortcut.
        "clear_sql_by_schema_style": {
            style: bootstrap_ci([int(r["predicted"] == "sql") for r in rows
                                 if r["category"] == "clear" and r["label"] == "sql" and r["schema_style"] == style])
            for style in sorted({r["schema_style"] for r in rows if r["schema_style"]})
        },
        "errors": [r for r in rows if r["label"] != r["predicted"]],
        "records": rows,
    }

    print_report(result)
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nRouter evaluation saved to {output_json}")
    return result


def print_report(result: Dict):
    print("\n" + "=" * 70)
    print(f"ROUTER EVALUATION  strategy={result['strategy']}  threshold={result['threshold']}")
    print("=" * 70)
    for part in ["overall", "clear", "ambiguous"]:
        s = result[part]
        ci = s.get("accuracy_ci95")
        ci_txt = f"  95% CI [{ci['low'] * 100:.1f}, {ci['high'] * 100:.1f}]" if ci else ""
        print(f"{part:<10} accuracy {s['accuracy'] * 100:6.2f}%  (n={s['n']}){ci_txt}")
    cm = result["overall"]["confusion_matrix"]
    print("\nConfusion matrix (rows = true, cols = predicted)")
    print(f"{'':<8}" + "".join(f"{l:>8}" for l in LABELS))
    for i, l in enumerate(LABELS):
        print(f"{l:<8}" + "".join(f"{cm[i][j]:>8}" for j in range(len(LABELS))))
    rep = result["overall"]["classification_report"]
    print("\nPer class:  precision  recall  f1")
    for l in LABELS:
        print(f"  {l:<8} {rep[l]['precision']:9.3f} {rep[l]['recall']:7.3f} {rep[l]['f1-score']:5.3f}")
    by_style = result.get("clear_sql_by_schema_style") or {}
    if by_style:
        print("\nClear SQL routed to sql, by schema format:")
        for style, s in by_style.items():
            if s:
                print(f"  {style:<8} {s['mean'] * 100:5.1f}%  (n={s['n']})")
    lat = result["latency_ms"]
    print(f"\nRouting latency: p50 {lat['p50']} ms, p95 {lat['p95']} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the semantic / learned router")
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "learned", "centroid"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fallback-to-base threshold (default: the router's own default)")
    parser.add_argument("--testset", type=str, default=DEFAULT_TESTSET)
    parser.add_argument("--split", type=str, default="test", choices=["test", "calibration", "all"],
                        help="Which half of the test set to use (reported results use 'test')")
    parser.add_argument("--model-path", type=str, default=None, help="Learned router .pkl (learned strategy)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON (default: results/router_eval_<strategy>.json)")
    args = parser.parse_args()
    out = args.output or f"results/router_eval_{args.strategy}.json"
    evaluate_router(strategy=args.strategy, threshold=args.threshold, testset_path=args.testset,
                    output_json=out, split=args.split, model_path=args.model_path)

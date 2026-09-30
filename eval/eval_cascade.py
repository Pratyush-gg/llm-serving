"""
Cascade evaluation with REAL generation (no simulated outputs).

For each prompt in the test set:
  - direct:  the router's choice (routing only; its generation is used for latency)
  - cascade: CascadeRouter, whose candidate generations come from a running gateway with the
             adapter forced (so the gateway's own router does not interfere)
Generations are cached per (adapter, prompt), so an adapter is generated at most once per prompt.
Latencies use the generation times reported by the gateway.

Usage:
  python -m src.gateway --engine peft            # in another terminal
  python eval/eval_cascade.py --endpoint http://127.0.0.1:8080 --strategy centroid
"""
import os
import sys
import json
import time
import argparse
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.router import get_router
from src.cascade import CascadeRouter
from eval.router_eval import load_testset, DEFAULT_TESTSET
from eval.stats import bootstrap_ci, paired_delta_ci


class GatewayGenerator:
    """Generates through the gateway with a forced adapter and caches results."""

    def __init__(self, endpoint: str, max_tokens: int, cache_path: Optional[str] = None, resume: bool = False):
        self.url = endpoint.rstrip("/").replace("//localhost", "//127.0.0.1") + "/v1/chat"
        self.max_tokens = max_tokens
        self.session = requests.Session()
        self.cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Generations are appended to cache_path as they happen, so an interrupted run can resume.
        self.cache_file = None
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            if resume and os.path.exists(cache_path):
                with open(cache_path, encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        self.cache[(r["route"], r["prompt"])] = {"text": r["text"], "generation_ms": r["generation_ms"]}
                print(f"Resuming: {len(self.cache)} cached generations from {cache_path}")
            self.cache_file = open(cache_path, "a" if resume else "w", encoding="utf-8")

    def generate(self, route: str, prompt: str) -> Dict[str, Any]:
        key = (route, prompt)
        if key not in self.cache:
            resp = self.session.post(self.url, json={
                "prompt": prompt, "max_tokens": self.max_tokens, "temperature": 0.0,
                "force_adapter": route, "enable_cascade": False,
            }, timeout=300)
            resp.raise_for_status()
            d = resp.json()
            self.cache[key] = {"text": d["response"], "generation_ms": d.get("generation_latency_ms") or 0.0}
            if self.cache_file:
                self.cache_file.write(json.dumps({"route": route, "prompt": prompt, **self.cache[key]},
                                                 ensure_ascii=False) + "\n")
                self.cache_file.flush()
        return self.cache[key]


def evaluate_cascade(endpoint: str, strategy: str = "auto", cascade_threshold: float = 0.70,
                     margin_threshold: float = 0.15, max_tokens: int = 256,
                     testset_path: str = DEFAULT_TESTSET, categories=("ambiguous", "clear"),
                     limit: Optional[int] = None, output_path: Optional[str] = None,
                     split: str = "test", resume: bool = False) -> Dict[str, Any]:
    router = get_router(strategy=strategy)
    strategy_name = getattr(router, "strategy_name", strategy)
    cascade = CascadeRouter(base_router=router, cascade_threshold=cascade_threshold,
                            margin_threshold=margin_threshold)
    gen = GatewayGenerator(endpoint, max_tokens, f"results/raw_outputs/cascade_{split}.jsonl", resume)

    items = [r for r in load_testset(testset_path, split) if r["category"] in categories]
    if limit:
        items = items[:limit]
    print(f"Evaluating {len(items)} prompts | router={strategy_name} | threshold={cascade_threshold} "
          f"| margin={margin_threshold} | max_tokens={max_tokens}")

    records = []
    for i, item in enumerate(items, start=1):
        prompt, label = item["prompt"], item["label"]

        direct = router.route_detailed(prompt)
        direct_gen = gen.generate(direct["route"], prompt)

        generated_routes = []

        def generate_fn(route: str, p: str) -> str:
            generated_routes.append(route)
            return gen.generate(route, p)["text"]

        res = cascade.route_and_generate(prompt, generate_fn)
        cascade_gen_ms = sum(gen.cache[(r, prompt)]["generation_ms"] for r in generated_routes)

        rec = {
            "id": item.get("id") or item.get("source_id"),
            "category": item["category"],
            "label": label,
            "prompt": prompt[:160],
            "direct_route": direct["route"],
            "direct_confidence": direct["confidence"],
            "direct_correct": direct["route"] == label,
            "cascade_route": res.final_adapter,
            "cascade_triggered": res.cascade_triggered,
            "cascade_correct": res.final_adapter == label,
            "direct_latency_ms": round(direct["latency_ms"] + direct_gen["generation_ms"], 2),
            "cascade_latency_ms": round(res.routing_latency_ms + cascade_gen_ms, 2),
            "selection_reason": res.selection_reason,
            "candidates": res.candidates_evaluated,
        }
        records.append(rec)
        mark = "rescue" if (rec["cascade_correct"] and not rec["direct_correct"]) else \
               "HARM" if (rec["direct_correct"] and not rec["cascade_correct"]) else ""
        print(f"  [{i}/{len(items)}] {label:<5} direct={rec['direct_route']:<5} cascade={rec['cascade_route']:<5} "
              f"{'(triggered)' if rec['cascade_triggered'] else '':<12}{mark}")

    def part(rows):
        if not rows:
            return {"n": 0}
        n = len(rows)
        direct = [int(r["direct_correct"]) for r in rows]
        casc = [int(r["cascade_correct"]) for r in rows]
        return {
            "n": n,
            "direct_accuracy": round(sum(direct) / n, 4),
            "cascade_accuracy": round(sum(casc) / n, 4),
            "direct_accuracy_ci95": bootstrap_ci(direct),
            "cascade_accuracy_ci95": bootstrap_ci(casc),
            "cascade_minus_direct_ci95": paired_delta_ci(direct, casc),
            "trigger_rate": round(sum(r["cascade_triggered"] for r in rows) / n, 4),
            "rescues": sum(r["cascade_correct"] and not r["direct_correct"] for r in rows),
            "harms": sum(r["direct_correct"] and not r["cascade_correct"] for r in rows),
            "direct_latency_p50_ms": round(float(np.percentile([r["direct_latency_ms"] for r in rows], 50)), 2),
            "cascade_latency_p50_ms": round(float(np.percentile([r["cascade_latency_ms"] for r in rows], 50)), 2),
            "direct_latency_mean_ms": round(float(np.mean([r["direct_latency_ms"] for r in rows])), 2),
            "cascade_latency_mean_ms": round(float(np.mean([r["cascade_latency_ms"] for r in rows])), 2),
        }

    summary = {
        "mode": "live_generation",
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "router_strategy": strategy_name,
        "cascade_threshold": cascade_threshold,
        "margin_threshold": margin_threshold,
        "max_tokens": max_tokens,
        "testset": testset_path,
        "split": split,
        "overall": part(records),
        "by_category": {c: part([r for r in records if r["category"] == c]) for c in categories},
        "records": records,
    }

    print("\n" + "=" * 70)
    for name, s in [("overall", summary["overall"]), *summary["by_category"].items()]:
        if s["n"]:
            print(f"{name:<10} n={s['n']:<4} direct {s['direct_accuracy'] * 100:5.1f}% -> cascade "
                  f"{s['cascade_accuracy'] * 100:5.1f}% | triggered {s['trigger_rate'] * 100:4.0f}% | "
                  f"rescues {s['rescues']} harms {s['harms']} | p50 latency "
                  f"{s['direct_latency_p50_ms']:.0f} -> {s['cascade_latency_p50_ms']:.0f} ms")
    print("=" * 70)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cascade routing with real generation")
    parser.add_argument("--endpoint", type=str, required=True, help="Running gateway, e.g. http://127.0.0.1:8080")
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "learned", "centroid"])
    parser.add_argument("--threshold", type=float, default=0.70, help="Cascade confidence threshold")
    parser.add_argument("--margin", type=float, default=0.15, help="Cascade top-2 margin threshold")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--testset", type=str, default=DEFAULT_TESTSET)
    parser.add_argument("--categories", type=str, default="ambiguous,clear")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N prompts")
    parser.add_argument("--split", type=str, default="test", choices=["test", "calibration", "all"])
    parser.add_argument("--calibration", type=str, default=None,
                        help="Use thresholds chosen by eval/calibrate_cascade.py (overrides --threshold/--margin)")
    parser.add_argument("--output", type=str, default="results/cascade_eval.json")
    parser.add_argument("--resume", action="store_true", help="Reuse saved generations (continue an interrupted run)")
    args = parser.parse_args()

    threshold, margin = args.threshold, args.margin
    if args.calibration:
        with open(args.calibration, "r", encoding="utf-8") as f:
            chosen = json.load(f)["chosen"]
        threshold, margin = chosen["cascade_threshold"], chosen["margin_threshold"]
        print(f"Using calibrated thresholds from {args.calibration}: threshold={threshold} margin={margin}")

    evaluate_cascade(args.endpoint, args.strategy, threshold, margin, args.max_tokens,
                     args.testset, tuple(args.categories.split(",")), args.limit, args.output, args.split, args.resume)

"""
Calibrate cascade thresholds on the CALIBRATION half of the router test set.

The cascade always compares the router's top-2 adapters, whatever the thresholds are, so each
prompt only needs generations for {direct route, top-1, top-2}. Those are generated once through
the gateway; every (confidence threshold, margin) pair in the grid is then scored offline.

Selection rule: highest accuracy on the calibration half; ties -> lowest trigger rate.
The full grid is saved so a different accuracy/latency trade-off can be chosen.
Reported results must come from eval/eval_cascade.py on the TEST half with these thresholds.

Usage:
  python eval/calibrate_cascade.py --endpoint http://127.0.0.1:8080 --strategy centroid
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.router import get_router
from src.cascade import CascadeRouter
from eval.router_eval import load_testset, DEFAULT_TESTSET
from eval.eval_cascade import GatewayGenerator

CONF_GRID = [0.0] + [round(x, 2) for x in np.arange(0.30, 1.0001, 0.05)] + [1.01]
MARGIN_GRID = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]


class CachedRouter:
    """Returns precomputed routing results so the grid search does not re-embed prompts."""

    def __init__(self, infos: Dict[str, Dict]):
        self.infos = infos

    def route_detailed(self, prompt: str) -> Dict:
        return self.infos[prompt]


def score_grid(items: List[Dict], infos: Dict[str, Dict], gen: GatewayGenerator) -> List[Dict]:
    router = CachedRouter(infos)
    grid = []
    for conf_t in CONF_GRID:
        for margin_t in MARGIN_GRID:
            cascade = CascadeRouter(base_router=router, cascade_threshold=conf_t, margin_threshold=margin_t)
            correct = triggered = 0
            gen_ms = []
            for item in items:
                prompt = item["prompt"]
                used = []

                def generate_fn(route, p):
                    used.append(route)
                    return gen.generate(route, p)["text"]

                res = cascade.route_and_generate(prompt, generate_fn)
                correct += res.final_adapter == item["label"]
                triggered += res.cascade_triggered
                gen_ms.append(sum(gen.cache[(r, prompt)]["generation_ms"] for r in used))
            n = len(items)
            grid.append({
                "cascade_threshold": conf_t,
                "margin_threshold": margin_t,
                "accuracy": round(correct / n, 4),
                "trigger_rate": round(triggered / n, 4),
                "mean_generation_ms": round(float(np.mean(gen_ms)), 1),
            })
    return grid


def main():
    parser = argparse.ArgumentParser(description="Calibrate cascade thresholds on the calibration split")
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "learned", "centroid"])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--testset", type=str, default=DEFAULT_TESTSET)
    parser.add_argument("--resume", action="store_true", help="Reuse saved generations (continue an interrupted run)")
    parser.add_argument("--output", type=str, default=None,
                        help="Default: results/cascade_calibration_<strategy>.json")
    args = parser.parse_args()

    router = get_router(strategy=args.strategy)
    strategy_name = getattr(router, "strategy_name", args.strategy)
    items = load_testset(args.testset, split="calibration")
    gen = GatewayGenerator(args.endpoint, args.max_tokens, "results/raw_outputs/cascade_calibration.jsonl", args.resume)

    print(f"Calibrating on {len(items)} prompts (router={strategy_name}). Generating candidates...")
    infos = {}
    for i, item in enumerate(items, start=1):
        info = router.route_detailed(item["prompt"])
        infos[item["prompt"]] = info
        top2 = [r for r, _ in sorted(info["scores"].items(), key=lambda kv: kv[1], reverse=True)[:2]]
        for route in dict.fromkeys([info["route"], *top2]):
            gen.generate(route, item["prompt"])
        print(f"  [{i}/{len(items)}] {item['label']:<5} routes generated: {', '.join(dict.fromkeys([info['route'], *top2]))}")

    grid = score_grid(items, infos, gen)
    best = sorted(grid, key=lambda g: (-g["accuracy"], g["trigger_rate"], -g["cascade_threshold"]))[0]
    never = next(g for g in grid if g["cascade_threshold"] == 0.0 and g["margin_threshold"] == 0.0)

    result = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "router_strategy": strategy_name,
        "split": "calibration",
        "n": len(items),
        "max_tokens": args.max_tokens,
        "selection_rule": "max accuracy on calibration split; ties -> lowest trigger rate",
        "chosen": best,
        "no_cascade_baseline": never,
        "grid": grid,
    }
    out = args.output or f"results/cascade_calibration_{strategy_name}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print(f"No cascade      : accuracy {never['accuracy'] * 100:.1f}%")
    print(f"Chosen          : threshold={best['cascade_threshold']} margin={best['margin_threshold']} "
          f"-> accuracy {best['accuracy'] * 100:.1f}%, trigger rate {best['trigger_rate'] * 100:.0f}%, "
          f"mean generation {best['mean_generation_ms']:.0f} ms")
    print("=" * 70)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

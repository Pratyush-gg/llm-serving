"""
Live latency benchmark against a running gateway (no simulation).

Sends requests sequentially to /v1/chat and records the per-stage timings the gateway reports
(routing, adapter switch, generation, token counts) plus the client-measured round trip.
Warm-up requests are excluded from the statistics.

Usage:
  python -m src.gateway --engine peft --router-strategy learned   # in another terminal
  python scripts/run_benchmarks.py --endpoint http://localhost:8080 --count 100
"""
import os
import sys
import json
import time
import random
import argparse
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import local_cache  # noqa: F401  (keeps the matplotlib cache inside the repo)

import numpy as np
import requests

# Chart colors: reference categorical slots 1-4 in fixed order, text inks, surface.
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
C_TEXT = "#0b0b0b"
C_TEXT_2 = "#52514e"
C_GRID = "#e4e3df"
C_SURFACE = "#fcfcfb"

BASE_PROMPTS = [
    "Explain the philosophical doctrine of stoicism and its core principles.",
    "What are the main ecological impacts of ocean acidification?",
    "Describe the lifecycle of a high-mass star from nebula to supernova.",
    "Provide a brief historical summary of the Renaissance period in Europe.",
    "How does the immune system develop adaptive immunity against pathogens?",
    "What are good strategies for learning a new language as an adult?",
    "Summarize the causes of the French Revolution.",
    "Why do leaves change color in autumn?",
]


def load_prompts(per_domain: int, seed: int) -> List[str]:
    prompts = []
    for task in ["sql", "json", "code"]:
        with open(f"data/{task}_holdout.jsonl", "r", encoding="utf-8") as f:
            prompts += [json.loads(line)["prompt"] for line in f if line.strip()][:per_domain]
    prompts += (BASE_PROMPTS * (per_domain // len(BASE_PROMPTS) + 1))[:per_domain]
    random.Random(seed).shuffle(prompts)
    return prompts


def stats(values: List[float]) -> Dict[str, float]:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "max": round(float(arr.max()), 2),
    }


def run_benchmarks(endpoint: str, count: int, max_tokens: int, warmup: int, router_strategy: str,
                   cascade: bool, seed: int) -> Dict:
    base_url = endpoint.rstrip("/")
    if "//localhost" in base_url:
        # On Windows, "localhost" tries IPv6 first and adds ~2 s per request against the
        # IPv4-only gateway, which would swamp the client round-trip measurement.
        base_url = base_url.replace("//localhost", "//127.0.0.1")
        print(f"Using {base_url} (avoids the Windows localhost IPv6 fallback delay)")
    health = requests.get(f"{base_url}/health", timeout=10).json()
    print(f"Gateway: engine={health.get('engine')} gpu={health.get('gpu_device')}")

    per_domain = max(1, (count + warmup) // 4 + 1)
    prompts = load_prompts(per_domain, seed)[: count + warmup]

    records = []
    for i, prompt in enumerate(prompts):
        payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
                   "router_strategy": router_strategy, "enable_cascade": cascade}
        t0 = time.perf_counter()
        resp = requests.post(f"{base_url}/v1/chat", json=payload, timeout=300)
        client_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        d = resp.json()

        is_warmup = i < warmup
        gen = d.get("generation_latency_ms")
        other = d["total_latency_ms"] - d["routing_latency_ms"] - (gen or 0) - (d.get("adapter_switch_ms") or 0)
        records.append({
            "warmup": is_warmup,
            "adapter": d["adapter_used"],
            "cascade_triggered": d.get("cascade_triggered", False),
            "client_round_trip_ms": round(client_ms, 2),
            "server_total_ms": d["total_latency_ms"],
            "routing_ms": d["routing_latency_ms"],
            "adapter_switch_ms": d.get("adapter_switch_ms"),
            "generation_ms": gen,
            "other_server_ms": round(other, 2),
            "prompt_tokens": d.get("prompt_tokens"),
            "completion_tokens": d.get("completion_tokens"),
            "tokens_per_second": d.get("tokens_per_second"),
        })
        tag = "warmup" if is_warmup else f"{i - warmup + 1}/{count}"
        print(f"  [{tag}] {d['adapter_used']:<5} total {d['total_latency_ms']:>8.1f} ms "
              f"| gen {gen or 0:>8.1f} ms | {d.get('completion_tokens')} tok")

    measured = [r for r in records if not r["warmup"]]

    def stage_stats(rows):
        return {
            "client_round_trip_ms": stats([r["client_round_trip_ms"] for r in rows]),
            "server_total_ms": stats([r["server_total_ms"] for r in rows]),
            "routing_ms": stats([r["routing_ms"] for r in rows]),
            "adapter_switch_ms": stats([r["adapter_switch_ms"] for r in rows]),
            "generation_ms": stats([r["generation_ms"] for r in rows]),
            "other_server_ms": stats([r["other_server_ms"] for r in rows]),
            "completion_tokens": stats([r["completion_tokens"] for r in rows]),
            "tokens_per_second": stats([r["tokens_per_second"] for r in rows]),
        }

    adapters = sorted({r["adapter"] for r in measured})
    return {
        "mode": "live_measured",
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "endpoint": base_url, "num_requests": len(measured), "warmup_requests": warmup,
            "max_tokens": max_tokens, "temperature": 0.0, "router_strategy": router_strategy,
            "cascade": cascade, "concurrency": 1, "seed": seed,
        },
        "gateway_health": health,
        "overall": stage_stats(measured),
        "by_adapter": {a: stage_stats([r for r in measured if r["adapter"] == a]) for a in adapters},
        "route_distribution": {a: sum(r["adapter"] == a for r in measured) for a in adapters},
        "requests": records,
    }


def generate_latency_chart(results: Dict, output_image: str):
    """Stacked bars of MEAN time per stage for each adapter (means add up; percentiles do not)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [("routing_ms", "Routing"), ("adapter_switch_ms", "Adapter switch"),
              ("generation_ms", "Generation"), ("other_server_ms", "Other server overhead")]
    groups = ["overall"] + list(results["by_adapter"].keys())
    data = {"overall": results["overall"], **results["by_adapter"]}

    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(groups) + 1.8), dpi=200)
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    ys = list(range(len(groups)))[::-1]
    left = np.zeros(len(groups))
    for (key, label), color in zip(stages, STAGE_COLORS):
        vals = np.array([data[g][key].get("mean", 0.0) or 0.0 for g in groups])
        ax.barh(ys, vals, left=left, height=0.55, color=color, label=label,
                edgecolor=C_SURFACE, linewidth=1)
        left += vals
    for y, total, g in zip(ys, left, groups):
        n = data[g]["server_total_ms"]["n"]
        ax.text(total + left.max() * 0.01, y, f"{total:,.0f} ms  (n={n})", va="center",
                fontsize=8.5, color=C_TEXT)

    ax.set_yticks(ys)
    ax.set_yticklabels(["All requests" if g == "overall" else f"{g} adapter" for g in groups],
                       fontsize=9, color=C_TEXT)
    ax.set_xlim(0, left.max() * 1.22)
    ax.set_xlabel("Mean server-side latency per request (ms)", fontsize=9, color=C_TEXT_2)
    ax.tick_params(axis="x", colors=C_TEXT_2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(C_GRID)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8,
              labelcolor=C_TEXT_2)

    cfg = results["config"]
    gpu = results["gateway_health"].get("gpu_device", "?")
    fig.text(0.01, 0.01, f"Measured on {gpu}, engine={results['gateway_health'].get('engine')}, "
             f"max_tokens={cfg['max_tokens']}, sequential requests. Routing and switch are "
             f"small relative to generation.", fontsize=7, color=C_TEXT_2)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(output_image) or ".", exist_ok=True)
    plt.savefig(output_image, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"Latency chart saved -> {output_image}")


def print_summary(results: Dict):
    print("\n" + "=" * 78)
    print(f"LATENCY SUMMARY ({results['config']['num_requests']} measured requests, "
          f"max_tokens={results['config']['max_tokens']})")
    print("=" * 78)
    print(f"{'Stage':<24} | {'mean':>9} | {'p50':>9} | {'p95':>9} | {'p99':>9}")
    print("-" * 78)
    for key, label in [("routing_ms", "Routing"), ("adapter_switch_ms", "Adapter switch"),
                       ("generation_ms", "Generation"), ("other_server_ms", "Other server"),
                       ("server_total_ms", "Server total"), ("client_round_trip_ms", "Client round trip")]:
        s = results["overall"][key]
        if s.get("n"):
            print(f"{label:<24} | {s['mean']:>9.2f} | {s['p50']:>9.2f} | {s['p95']:>9.2f} | {s['p99']:>9.2f}")
        else:
            print(f"{label:<24} | {'not reported':>45}")
    tps = results["overall"]["tokens_per_second"]
    if tps.get("n"):
        print(f"\nDecode throughput: {tps['p50']:.1f} tokens/s (p50)")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Live latency benchmark against a running gateway")
    parser.add_argument("--endpoint", type=str, required=True, help="Gateway base URL, e.g. http://localhost:8080")
    parser.add_argument("--count", type=int, default=100, help="Measured requests (excluding warm-up)")
    parser.add_argument("--warmup", type=int, default=3, help="Warm-up requests excluded from stats")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--router-strategy", type=str, default="auto", choices=["auto", "learned", "centroid"])
    parser.add_argument("--cascade", action="store_true", help="Enable cascade on every request")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default="results/latency_breakdown.json")
    parser.add_argument("--output-image", type=str, default="results/latency_breakdown.png")
    args = parser.parse_args()

    results = run_benchmarks(args.endpoint, args.count, args.max_tokens, args.warmup,
                             args.router_strategy, args.cascade, args.seed)
    print_summary(results)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {args.output_json}")
    generate_latency_chart(results, args.output_image)


if __name__ == "__main__":
    main()

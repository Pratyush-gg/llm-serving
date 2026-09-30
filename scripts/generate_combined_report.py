"""
Build results/combined_benchmark_report.md from the measured result files.

Every figure comes from a results/*.json file produced by a measurement script; the only
estimate (3 separate models' weight memory) is labelled as such. Missing files are reported
as "not run" instead of being filled with placeholder numbers.
"""
import json
import os
from datetime import datetime

RESULTS = os.environ.get("REPORT_RESULTS_DIR", "results")


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


TASK_LABELS = {
    "sql": "SQL (gretel, real data rows)",
    "json": "JSON in-scope (order extraction)",
    "json_paraloq": "JSON out-of-scope (paraloq schemas)",
    "code_humaneval": "Code: HumanEval",
    "code_mbpp": "Code: MBPP (sanitized test)",
}


def ci_str(ci, signed=False):
    if not ci:
        return "n/a"
    f = "{:+.1f}" if signed else "{:.1f}"
    return f"[{f.format(ci['low'] * 100)}, {f.format(ci['high'] * 100)}]"


def quality_section(bt):
    summary = (bt or {}).get("summary") or {}
    rows = {k: v for k, v in summary.items() if isinstance(v, dict) and "baseline_ci95" in v}
    if not rows:
        return "## 1. Task quality\n\n*Not run (no results in the current format).*\n"
    lines = [
        "## 1. Task quality: base model vs. LoRA adapters (natural-language prompts)",
        "",
        "| Test set | Metric | n | Base model | LoRA adapter | Difference (95% CI, paired) |",
        "| :--- | :--- | ---: | :--- | :--- | :--- |",
    ]
    for task, s in rows.items():
        delta = f"{s['delta'] * 100:+.1f} pts {ci_str(s['delta_ci95'], signed=True)}" if s["delta"] is not None else "n/a"
        lines.append(f"| {TASK_LABELS.get(task, task)} | `{s['metric']}` | {s['n']} "
                     f"| {pct(s['baseline'])} {ci_str(s['baseline_ci95'])} "
                     f"| **{pct(s['tuned'])}** {ci_str(s['tuned_ci95'])} | {delta} |")
    lines += [
        "",
        "Brackets are 95% bootstrap confidence intervals. A difference whose interval includes 0 is not",
        "distinguishable from noise at this sample size. Raw model outputs: `results/raw_outputs/`.",
        "",
    ]

    # SQL accuracy by how the schema was shown (the adapter was trained on CREATE TABLE only).
    b_sql = (bt.get("baseline_metrics") or {}).get("sql", {}).get("by_schema_style")
    t_sql = (bt.get("tuned_metrics") or {}).get("sql", {}).get("by_schema_style")
    if b_sql and t_sql:
        lines += ["**SQL execution accuracy by schema format in the prompt**", "",
                  "| Schema format | n | Base model | LoRA adapter |", "| :--- | ---: | :--- | :--- |"]
        for style in ["create", "compact", "prose"]:
            if style in b_sql and style in t_sql:
                b, t = b_sql[style], t_sql[style]
                lines.append(f"| {style} | {t['n']} | {pct(b['mean'])} {ci_str(b)} | {pct(t['mean'])} {ci_str(t)} |")
        lines.append("")
    return "\n".join(lines)


def memory_section(mem):
    if not mem or mem.get("mode") != "measured":
        return "## 2. GPU memory\n\n*Not run (no measured profile).*\n"
    s, env, est = mem["summary"], mem["environment"], mem["estimate_three_separate_models"]
    return "\n".join([
        f"## 2. GPU memory (measured on {env['gpu']}, {env['quantization']})",
        "",
        "| Quantity | MB | Source |",
        "| :--- | ---: | :--- |",
        f"| Base model weights | {s['base_model_weights_mb']:,.0f} | measured |",
        f"| 3 LoRA adapters | {s['three_adapters_mb']:,.0f} | measured |",
        f"| Base + 3 adapters resident | {s['multi_lora_resident_mb']:,.0f} | measured |",
        f"| Peak during generation ({env['max_new_tokens']} new tokens) | {s['multi_lora_peak_during_generation_mb']:,.0f} | measured |",
        f"| 3 separate fine-tuned models, weights only | {est['weights_mb']:,.0f} | **estimate**: {est['method']} |",
        "",
        "![Memory profile](memory_profile.png)",
        "",
    ])


def latency_section(lat):
    if not lat or lat.get("mode") != "live_measured":
        return "## 3. Latency\n\n*Not run (no live measurement).*\n"
    o, cfg, health = lat["overall"], lat["config"], lat["gateway_health"]
    lines = [
        f"## 3. Latency ({cfg['num_requests']} sequential requests, max_tokens={cfg['max_tokens']}, "
        f"engine={health.get('engine')}, {health.get('gpu_device')})",
        "",
        "| Stage | mean | p50 | p95 | p99 |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in [("routing_ms", "Routing"), ("adapter_switch_ms", "Adapter switch"),
                       ("generation_ms", "Generation"), ("other_server_ms", "Other server overhead"),
                       ("server_total_ms", "Server total"), ("client_round_trip_ms", "Client round trip")]:
        st = o[key]
        if st.get("n"):
            lines.append(f"| {label} | {st['mean']:,.1f} ms | {st['p50']:,.1f} ms | {st['p95']:,.1f} ms | {st['p99']:,.1f} ms |")
        else:
            lines.append(f"| {label} | not reported | | | |")
    tps, tok = o["tokens_per_second"], o["completion_tokens"]
    if tps.get("n"):
        lines += ["", f"Decode throughput: {tps['p50']:.1f} tokens/s (p50); "
                      f"generated tokens per request: {tok['p50']:.0f} (p50)."]
    lines += ["", "Percentiles are per stage and do not add up; the chart stacks means.", "",
              "![Latency breakdown](latency_breakdown.png)", ""]
    return "\n".join(lines)


def router_section(evals):
    present = {k: v for k, v in evals.items() if v}
    if not present:
        return "## 4. Router\n\n*Not run.*\n"
    any_eval = next(iter(present.values()))
    lines = [
        f"## 4. Router accuracy (split '{any_eval.get('split', 'all')}' of `{any_eval['testset']}`, "
        f"{any_eval['overall']['n']} prompts)",
        "",
        "| Router | Overall | Clear-domain | Ambiguous | p50 latency |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for name, r in present.items():
        cells = [f"{pct(r[p]['accuracy'])} {ci_str(r[p].get('accuracy_ci95'))}" for p in ["overall", "clear", "ambiguous"]]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {r['latency_ms']['p50']} ms |")
    lines += ["", "Clear-domain prompts come from public datasets not used for adapter training "
                  "(gretel synthetic_text_to_sql, MBPP, Dolly) plus hand-written JSON requests; ambiguous "
                  "prompts are hand-written.", ""]
    return "\n".join(lines)


def cascade_section(cal, ev):
    if not ev or ev.get("mode") != "live_generation":
        return "## 5. Cascade\n\n*Not run.*\n"
    o = ev["overall"]
    lines = [
        f"## 5. Cascade (router={ev['router_strategy']}, split '{ev.get('split', 'all')}', real generation)",
        "",
    ]
    if cal:
        c, never = cal["chosen"], cal["no_cascade_baseline"]
        lines += [f"Thresholds calibrated on the calibration half ({cal['n']} prompts; rule: {cal['selection_rule']}): "
                  f"confidence < {c['cascade_threshold']} or top-2 margin < {c['margin_threshold']}. "
                  f"Calibration accuracy {pct(never['accuracy'])} without cascade -> {pct(c['accuracy'])} with it.", ""]
    lines += [
        "| Subset | n | Direct routing | With cascade | Triggered | Rescues | Harms | p50 latency direct -> cascade |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for name, s in [("all", o), *ev["by_category"].items()]:
        if s.get("n"):
            lines.append(f"| {name} | {s['n']} | {pct(s['direct_accuracy'])} {ci_str(s.get('direct_accuracy_ci95'))} "
                         f"| {pct(s['cascade_accuracy'])} {ci_str(s.get('cascade_accuracy_ci95'))} "
                         f"| {pct(s['trigger_rate'])} | {s['rescues']} | {s['harms']} "
                         f"| {s['direct_latency_p50_ms']:,.0f} -> {s['cascade_latency_p50_ms']:,.0f} ms |")
    lines.append("")
    return "\n".join(lines)


def generate_report():
    parts = [
        "# Combined Benchmark Report",
        "**Routed Multi-Adapter LLM Serving System**",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. All figures are measured unless marked "
        "as an estimate.*",
        "",
        quality_section(load("baseline_vs_tuned.json")),
        memory_section(load("memory_profile.json")),
        latency_section(load("latency_breakdown.json")),
        router_section({
            "Centroid (cosine)": load("router_eval_centroid.json"),
            "Learned v1 (original, retired)": load("router_eval_learned_v1.json"),
            "Learned v2 (default)": load("router_eval_learned.json"),
        }),
        cascade_section(load("cascade_calibration.json"), load("cascade_eval.json")),
    ]
    report = "\n".join(parts)
    path = os.path.join(RESULTS, "combined_benchmark_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nCombined benchmark report generated -> {path}")


if __name__ == "__main__":
    generate_report()

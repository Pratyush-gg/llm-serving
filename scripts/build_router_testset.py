"""Build the router test set (data/router_testset.jsonl) with calibration and test halves."""
import os
import sys
import json
import random
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Keep dataset downloads inside the repo.
os.environ.setdefault("HF_HUB_CACHE", os.path.join(REPO_ROOT, ".model_cache", "hub"))
from src import local_cache  # noqa: F401,E402

from datasets import load_dataset  # noqa: E402

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from schema_formats import STYLES, render_schema  # noqa: E402

HANDWRITTEN_PATH = os.path.join(REPO_ROOT, "data", "router_testset_handwritten.jsonl")
DOLLY_CATEGORIES = {"open_qa", "general_qa", "brainstorming", "creative_writing"}


def public_sql(n, rng):
    """Schema shown in rotating styles (no data rows) so CREATE TABLE is not a giveaway."""
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="test")
    idx = rng.sample(range(len(ds)), n)  # same draw as before, so later samples (code, base) are unchanged
    # Replace schemas SQLite can't parse using a separate RNG, so later samples stay unchanged.
    replacement_rng = random.Random(9001)
    rows = []
    for k, i in enumerate(idx):
        style = STYLES[k % len(STYLES)]
        while True:
            try:
                schema_text = render_schema(ds[i]["sql_context"], style)
                break
            except Exception:
                i = replacement_rng.randrange(len(ds))
                while i in idx:
                    i = replacement_rng.randrange(len(ds))
        rows.append({
            "prompt": f"{schema_text}\n\n{ds[i]['sql_prompt'].strip()}",
            "label": "sql", "category": "clear", "schema_style": style,
            "source": "gretelai/synthetic_text_to_sql:test", "source_id": str(ds[i]["id"]), "license": "Apache-2.0",
        })
    return rows


def public_code(n, rng):
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    idx = rng.sample(range(len(ds)), n)
    return [{
        "prompt": ds[i]["text"].strip(),
        "label": "code", "category": "clear",
        "source": "google-research-datasets/mbpp:full/test", "source_id": str(ds[i]["task_id"]), "license": "CC-BY-4.0",
    } for i in idx]


def public_base(n, rng):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    pool = [i for i in range(len(ds))
            if ds[i]["category"] in DOLLY_CATEGORIES and not ds[i]["context"].strip()
            and 15 <= len(ds[i]["instruction"]) <= 300]
    idx = rng.sample(pool, n)
    return [{
        "prompt": ds[i]["instruction"].strip(),
        "label": "base", "category": "clear",
        "source": f"databricks/databricks-dolly-15k:{ds[i]['category']}", "source_id": str(i), "license": "CC-BY-SA-3.0",
    } for i in idx]


def handwritten():
    if not os.path.exists(HANDWRITTEN_PATH):
        raise SystemExit(f"Missing {HANDWRITTEN_PATH}")
    with open(HANDWRITTEN_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        r.setdefault("source", "handwritten")
        r.setdefault("license", "project")
    return rows


def assign_splits(rows, rng):
    """Stratified 50/50 calibration/test split per (category, label)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["category"], r["label"]), []).append(r)
    for members in groups.values():
        order = list(range(len(members)))
        rng.shuffle(order)
        half = len(members) // 2
        for rank, idx in enumerate(order):
            members[idx]["split"] = "calibration" if rank < half else "test"


def main():
    parser = argparse.ArgumentParser(description="Build the router/cascade test set")
    parser.add_argument("--per-domain", type=int, default=50, help="Public prompts per class (sql, code, base)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=str, default=os.path.join(REPO_ROOT, "data", "router_testset.jsonl"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = public_sql(args.per_domain, rng) + public_code(args.per_domain, rng) + public_base(args.per_domain, rng)
    rows += handwritten()

    # Guard against leakage: no test prompt may appear verbatim in any training file.
    train_prompts = set()
    for task in ["sql", "json", "code"]:
        with open(os.path.join(REPO_ROOT, "data", f"{task}_train.jsonl"), encoding="utf-8") as f:
            train_prompts |= {json.loads(line)["prompt"].strip() for line in f if line.strip()}
    leaked = [r["prompt"] for r in rows if r["prompt"].strip() in train_prompts]
    if leaked:
        raise SystemExit(f"{len(leaked)} test prompts also appear in training data, e.g. {leaked[0][:80]!r}")

    assign_splits(rows, random.Random(args.seed + 1))

    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {}
    for r in rows:
        key = f"{r['category']}/{r['label']}"
        counts.setdefault(key, {"calibration": 0, "test": 0})[r["split"]] += 1
    print(f"Wrote {len(rows)} prompts -> {args.output}")
    print(f"  {'group':<16} {'calibration':>11} {'test':>6}")
    for k in sorted(counts):
        print(f"  {k:<16} {counts[k]['calibration']:>11} {counts[k]['test']:>6}")


if __name__ == "__main__":
    main()

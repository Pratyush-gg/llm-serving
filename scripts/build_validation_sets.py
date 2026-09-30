"""
Build adapter validation sets (data/<task>_val.jsonl) from rows NOT used for train or holdout.

The original data scripts pick rows deterministically. This script replays the same selection,
checks that it reproduces the existing train + holdout files exactly (so we know which rows are
used), and then takes the next unused rows as validation data. The train files are not changed.

  sql  : next 60 valid rows of b-mc2/sql-create-context after the 660 used ones
  json : next 60 unique samples from the same synthetic generator (same seed)
  code : 60 problems from MBPP's validation split (the code generator has only 13 unused tasks)
"""
import os
import sys
import json
import random
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(REPO_ROOT, ".model_cache", "hub"))
from src import local_cache  # noqa: F401,E402

USED = 660  # 600 train + 60 holdout in the original scripts


def read_prompts(path):
    with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as f:
        return [json.loads(line)["prompt"] for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(os.path.join(REPO_ROOT, path), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {len(rows)} validation examples -> {path}")


def verify_prefix(task, regenerated):
    existing = read_prompts(f"data/{task}_train.jsonl") + read_prompts(f"data/{task}_holdout.jsonl")
    if [r["prompt"] for r in regenerated[:USED]] != existing:
        raise SystemExit(f"[{task}] replayed selection does not match the existing train/holdout files; "
                         "refusing to build a validation set that might overlap them.")
    print(f"  [{task}] replay matches the existing {USED} train+holdout rows exactly")


def build_sql(n):
    import prepare_sql_data as p
    from datasets import load_dataset

    ds = load_dataset("b-mc2/sql-create-context", split="train")
    indices = list(range(len(ds)))
    random.seed(p.SEED)
    random.shuffle(indices)
    rows = []
    for idx in indices:
        r = ds[idx]
        schema, question, gold = r["context"].strip(), r["question"].strip(), r["answer"].strip()
        if not p.is_valid_sqlite_schema(schema):
            continue
        rows.append({"prompt": p.format_sql_prompt(schema, question), "schema": schema,
                     "question": question, "gold_sql": gold, "completion": gold})
        if len(rows) >= USED + n:
            break
    verify_prefix("sql", rows)
    write_jsonl("data/sql_val.jsonl", rows[USED:USED + n])


def build_json(n):
    import generate_json_data as g  # seeds the global RNG exactly like the original script

    rows, seen = [], set()
    while len(rows) < USED + n:
        s = g.generate_sample()
        if s["input_text"] not in seen:
            seen.add(s["input_text"])
            rows.append(s)
    verify_prefix("json", rows)
    write_jsonl("data/json_val.jsonl", rows[USED:USED + n])


def build_code(n, seed=7):
    """The code generator has only 13 unused tasks, so code validation uses MBPP's validation
    split (real problems; disjoint from the MBPP test split used for evaluation)."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split="validation")
    idx = random.Random(seed).sample(range(len(ds)), n)
    rows = []
    for i in idx:
        r = ds[i]
        tests = list(r["test_list"])
        rows.append({
            "name": f"mbpp-val-{r['task_id']}",
            "prompt": f"{r['text'].strip()}\nYour code should pass this test:\n{tests[0]}",
            "completion": r["code"].strip(),
            "gold_code": r["code"].strip(),
            "assertions": "\n".join(tests),
            "source": "google-research-datasets/mbpp:full/validation",
        })
    write_jsonl("data/code_val.jsonl", rows)


def main():
    parser = argparse.ArgumentParser(description="Build adapter validation sets from unused rows")
    parser.add_argument("--n", type=int, default=60)
    args = parser.parse_args()
    print("Building validation sets:")
    build_sql(args.n)
    build_json(args.n)
    build_code(args.n)


if __name__ == "__main__":
    main()

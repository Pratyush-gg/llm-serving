"""
Build realistic, natural-language test sets for the LoRA adapters (data/eval/).

  sql_gretel.jsonl       ~300 SELECT questions from gretelai/synthetic_text_to_sql (test split, Apache-2.0).
                         Each example carries its own CREATE + INSERT statements, so queries run on real rows.
                         The prompt shows the schema only (no data rows), in one of three styles
                         (CREATE TABLE / compact / plain English, a third each) - see scripts/schema_formats.py.
                         Kept only if the gold query runs in SQLite and returns a non-empty result.
                         Rows used by the router test set are excluded.
  code_humaneval.jsonl   164 problems from openai/openai_humaneval (MIT) with the official tests.
  code_mbpp.jsonl        257 problems from MBPP sanitized test split (CC BY 4.0), scored on all tests;
                         the prompt shows one test so the model knows the function name.
  json_paraloq.jsonl     80 documents sampled from paraloq/json_data_extraction (Apache-2.0) with
                         schema + gold JSON: an OUT-OF-SCOPE generalization test for the JSON adapter.
                         Only documents whose prompt fits in ~3k tokens are eligible (biases toward shorter docs).

The in-scope JSON set (data/eval/json_inscope.jsonl) is hand-written and reviewed separately.
All downloads go to the repo-local .model_cache/ folder.
"""
import os
import sys
import json
import random
import sqlite3
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(REPO_ROOT, ".model_cache", "hub"))
from src import local_cache  # noqa: F401,E402

from datasets import load_dataset  # noqa: E402

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from schema_formats import STYLES, render_schema  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "data", "eval")


def write_jsonl(name, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {len(rows):>4} examples -> {os.path.relpath(path, REPO_ROOT)}")


def router_testset_ids(source_prefix):
    path = os.path.join(REPO_ROOT, "data", "router_testset.jsonl")
    ids = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("source", "").startswith(source_prefix):
                    ids.add(str(r.get("source_id")))
    return ids


def gold_result(setup_sql, gold_sql):
    try:
        conn = sqlite3.connect(":memory:")
        conn.executescript(setup_sql)
        rows = conn.execute(gold_sql).fetchall()
        conn.close()
    except Exception:
        return None
    return rows


def build_sql(n, rng):
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="test")
    excluded = router_testset_ids("gretelai/synthetic_text_to_sql")
    order = list(range(len(ds)))
    rng.shuffle(order)
    rows = []
    for i in order:
        r = ds[i]
        if str(r["id"]) in excluded:
            continue
        gold = r["sql"].strip().rstrip(";")
        if not gold.upper().startswith(("SELECT", "WITH")):
            continue
        result = gold_result(r["sql_context"], gold)
        if not result or all(v is None for row in result for v in row):
            continue
        # Rotate schema styles so each covers a third; data rows are never shown in the prompt.
        style = STYLES[len(rows) % len(STYLES)]
        rows.append({
            "id": f"gretel-{r['id']}",
            "prompt": f"{render_schema(r['sql_context'], style)}\n\n{r['sql_prompt'].strip()} (The database is SQLite.)",
            "schema_style": style,
            "setup_sql": r["sql_context"],
            "gold_sql": gold,
            "domain": r["domain"],
            "sql_complexity": r["sql_complexity"],
            "source": "gretelai/synthetic_text_to_sql:test", "license": "Apache-2.0",
        })
        if len(rows) >= n:
            break
    return rows


def build_humaneval():
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = []
    for r in ds:
        stub, entry = r["prompt"], r["entry_point"]
        cut = stub.find(f"def {entry}(")
        prelude = stub[:cut] if cut > 0 else ""
        rows.append({
            "id": r["task_id"],
            "name": r["task_id"],
            "prompt": f"Complete the following Python function:\n\n```python\n{stub.rstrip()}\n```",
            "prelude": prelude,  # imports / helper functions defined before the target function
            "assertions": f"{r['test']}\n\ncheck({entry})",
            "gold_code": stub + r["canonical_solution"],
            "source": "openai/openai_humaneval:test", "license": "MIT",
        })
    return rows


def build_mbpp():
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    rows = []
    for r in ds:
        tests = list(r["test_list"])
        rows.append({
            "id": f"mbpp-{r['task_id']}",
            "name": f"mbpp-{r['task_id']}",
            "prompt": f"{r['prompt'].strip()}\nYour code should pass this test:\n{tests[0]}",
            "prelude": "\n".join(r["test_imports"]),
            "assertions": "\n".join(r["test_imports"] + tests),
            "gold_code": r["code"],
            "source": "google-research-datasets/mbpp:sanitized/test", "license": "CC-BY-4.0",
        })
    return rows


MAX_PARALOQ_PROMPT_CHARS = 12_000  # ~3k tokens; some documents exceed the model's context window


def paraloq_prompt(r):
    return (f"Extract the information from this {r['medium']} as JSON that follows this schema.\n\n"
            f"Schema:\n{r['schema']}\n\nDocument:\n{r['text']}")


def build_paraloq(n, rng):
    ds = load_dataset("paraloq/json_data_extraction", split="train")
    eligible = [i for i in range(len(ds)) if len(paraloq_prompt(ds[i])) <= MAX_PARALOQ_PROMPT_CHARS]
    print(f"  paraloq: {len(eligible)}/{len(ds)} documents fit within {MAX_PARALOQ_PROMPT_CHARS} prompt characters")
    idx = rng.sample(eligible, n)
    rows = []
    for i in idx:
        r = ds[i]
        rows.append({
            "id": f"paraloq-{i}",
            "prompt": paraloq_prompt(r),
            "schema": json.loads(r["schema"]),
            "gold_json": json.loads(r["item"]),
            "topic": r["topic"],
            "source": "paraloq/json_data_extraction:train", "license": "Apache-2.0",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build realistic LoRA test sets")
    parser.add_argument("--sql", type=int, default=300)
    parser.add_argument("--paraloq", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    print("Building test sets:")
    write_jsonl("sql_gretel.jsonl", build_sql(args.sql, random.Random(args.seed)))
    write_jsonl("code_humaneval.jsonl", build_humaneval())
    write_jsonl("code_mbpp.jsonl", build_mbpp())
    write_jsonl("json_paraloq.jsonl", build_paraloq(args.paraloq, random.Random(args.seed + 1)))


if __name__ == "__main__":
    main()

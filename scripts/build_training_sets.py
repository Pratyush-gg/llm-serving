"""Build the v2 SQL and code training/validation sets from public data."""
import os
import re
import ast
import sys
import json
import random
import sqlite3
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(REPO_ROOT, ".model_cache", "hub"))
from src import local_cache  # noqa: F401,E402

from datasets import load_dataset  # noqa: E402
from schema_formats import STYLES, render_schema  # noqa: E402

DATA = os.path.join(REPO_ROOT, "data")


def write_jsonl(name, rows):
    with open(os.path.join(DATA, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {len(rows):>5} -> data/{name}")


def load_jsonl(rel):
    with open(os.path.join(DATA, rel), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- SQL -------------------------------------------------------------------------

def runs_with_rows(setup_sql, query):
    try:
        conn = sqlite3.connect(":memory:")
        conn.executescript(setup_sql)
        rows = conn.execute(query).fetchall()
        conn.close()
    except Exception:
        return False
    return bool(rows) and not all(v is None for row in rows for v in row)


def build_sql(n_train, n_val, rng):
    test_questions = {e["prompt"].split("\n\n")[-1].replace(" (The database is SQLite.)", "").strip().lower()
                      for e in load_jsonl("eval/sql_gretel.jsonl")}
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    order = list(range(len(ds)))
    rng.shuffle(order)
    rows = []
    for i in order:
        r = ds[i]
        gold = r["sql"].strip().rstrip(";")
        question = r["sql_prompt"].strip()
        if not gold.upper().startswith(("SELECT", "WITH")) or question.lower() in test_questions:
            continue
        if not runs_with_rows(r["sql_context"], gold):
            continue
        style = STYLES[len(rows) % len(STYLES)]
        rows.append({
            "prompt": f"{render_schema(r['sql_context'], style)}\n\n{question} (The database is SQLite.)",
            "completion": gold,
            "schema_style": style,
            "source": f"gretelai/synthetic_text_to_sql:train:{r['id']}",
        })
        if len(rows) >= n_train + n_val:
            break
    return rows[:n_train], rows[n_train:]


# --- Code ------------------------------------------------------------------------

def ngrams(text, n=10):
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def test_ngrams():
    """10-grams of the test problems themselves (prompt boilerplate removed to avoid false matches)."""
    grams = set()
    for rel in ["eval/code_humaneval.jsonl", "eval/code_mbpp.jsonl"]:
        for e in load_jsonl(rel):
            text = e["prompt"].replace("Complete the following Python function:", "")
            grams |= ngrams(text.split("\nYour code should pass this test:")[0])
    return grams


def first_python_block(response):
    for block in re.findall(r"```(?:python|py)?\s*\n([\s\S]*?)```", response):
        if "def " in block:
            try:
                ast.parse(block)
                return block.strip()
            except SyntaxError:
                continue
    return None


def docstring_stub(code, instruction):
    """HumanEval-style stub (signature + docstring) for the first top-level function."""
    lines = code.splitlines()
    for node in ast.parse(code).body:
        if isinstance(node, ast.FunctionDef):
            if ast.get_docstring(node):
                return "\n".join(lines[node.lineno - 1:node.body[0].end_lineno])
            signature = "\n".join(lines[node.lineno - 1:node.body[0].lineno - 1])
            indent = " " * node.body[0].col_offset
            return f'{signature}\n{indent}"""{instruction.strip()}"""'
    return None


def build_code(n_train, n_val, rng):
    contaminated = test_ngrams()
    clean = lambda text: not (ngrams(text) & contaminated)
    rows = []

    for split in ["train", "prompt"]:
        for r in load_dataset("google-research-datasets/mbpp", "full", split=split):
            prompt = f"{r['text'].strip()}\nYour code should pass this test:\n{r['test_list'][0]}"
            if clean(r["text"]):
                rows.append({"prompt": prompt, "completion": r["code"].strip(), "style": "mbpp",
                             "source": f"mbpp:full/{split}:{r['task_id']}"})

    ds = load_dataset("bigcode/self-oss-instruct-sc2-exec-filter-50k", split="train")
    order = list(range(len(ds)))
    rng.shuffle(order)
    for i in order:
        if len(rows) >= n_train + n_val:
            break
        r = ds[i]
        code = first_python_block(r["response"])
        if not code or not clean(r["instruction"] + "\n" + code):
            continue
        stub = docstring_stub(code, r["instruction"]) if rng.random() < 0.5 else None
        if stub:
            prompt, style = f"Complete the following Python function:\n\n```python\n{stub}\n```", "stub"
        else:
            prompt, style = r["instruction"].strip(), "instruction"
        rows.append({"prompt": prompt, "completion": code, "style": style,
                     "source": f"bigcode/self-oss-instruct-sc2-exec-filter-50k:{r['id']}"})

    rng.shuffle(rows)
    return rows[:n_train], rows[n_train:n_train + n_val]


def main():
    parser = argparse.ArgumentParser(description="Build v2 adapter training sets")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print("SQL:")
    train, val = build_sql(args.n_train, args.n_val, random.Random(args.seed))
    write_jsonl("sql_train_v2.jsonl", train)
    write_jsonl("sql_val_v2.jsonl", val)
    print("Code:")
    train, val = build_code(args.n_train, args.n_val, random.Random(args.seed + 1))
    write_jsonl("code_train_v2.jsonl", train)
    write_jsonl("code_val_v2.jsonl", val)


if __name__ == "__main__":
    main()

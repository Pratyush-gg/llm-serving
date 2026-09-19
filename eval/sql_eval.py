import json
import os
import re
import sqlite3
import argparse
from typing import Callable, Dict, List

def clean_sql(raw_text: str) -> str:
    """Extract raw SQL query from potential markdown code fences or whitespace."""
    text = raw_text.strip()
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # If the model echoes 'SQL Query: ...'
    if "SQL Query:" in text:
        text = text.split("SQL Query:")[-1].strip()
    return text.strip().rstrip(";")

def build_test_db(schema_sql: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema_sql)
    return conn

def run_query(conn: sqlite3.Connection, query: str):
    try:
        cur = conn.execute(query)
        rows = cur.fetchall()
        # Sort rows to ensure order-agnostic comparison unless ORDER BY was specified
        return sorted([tuple(r) for r in rows], key=lambda x: str(x))
    except Exception:
        return None

def evaluate_sql(examples: List[Dict], generate_fn: Callable[[str], str]) -> Dict[str, float]:
    executed = 0
    matched = 0
    n = len(examples)

    for ex in examples:
        conn = build_test_db(ex["schema"])
        raw_gen = generate_fn(ex["prompt"])
        cleaned_gen = clean_sql(raw_gen)

        gen_result = run_query(conn, cleaned_gen)
        gold_result = run_query(conn, clean_sql(ex["gold_sql"]))

        if gen_result is not None:
            executed += 1
            if gold_result is not None and gen_result == gold_result:
                matched += 1

        conn.close()

    return {
        "total_samples": n,
        "execution_rate": round(executed / n, 4) if n else 0.0,
        "exact_match_rate": round(matched / n, 4) if n else 0.0,
        "executed_count": executed,
        "matched_count": matched,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate SQL Generation Correctness")
    parser.add_argument("--data", type=str, default="data/sql_holdout.jsonl",
                        help="Path to SQL held-out evaluation dataset")
    parser.add_argument("--test-gold", action="store_true",
                        help="Sanity test evaluation harness using gold references (should be 100%)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Holdout file not found: {args.data}")

    with open(args.data, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(examples)} SQL evaluation examples from {args.data}")

    if args.test_gold:
        print("Running sanity test using reference gold_sql completions...")
        results = evaluate_sql(examples, generate_fn=lambda p: [ex["gold_sql"] for ex in examples if ex["prompt"] == p][0])
        print(f"SQL Evaluation Results: {json.dumps(results, indent=2)}")
        assert results["exact_match_rate"] == 1.0, "Gold reference queries must achieve 100% exact match!"
        print("Sanity test PASSED: 100% execution and exact match.")

if __name__ == "__main__":
    main()

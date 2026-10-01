import json
import os
import re
import sqlite3
import argparse
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

MAX_VM_STEPS = 2_000_000    # abort runaway queries (e.g. recursive CTEs, huge cross joins)

# Model-generated SQL may only read: no ATTACH, writes or schema changes.
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


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


def normalize_sql(sql: str) -> str:
    """Lenient canonical form for string matching: case, whitespace, quotes and trailing ';'."""
    s = clean_sql(sql).strip().rstrip(";").strip()
    s = s.replace('"', "'")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([(),=<>!])\s*", r"\1", s)
    return s.lower()


def _deny_writes(action, arg1, arg2, db_name, trigger):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _make_step_limiter():
    steps = [0]

    def handler():
        steps[0] += 1000
        return 1 if steps[0] > MAX_VM_STEPS else 0

    return handler


def run_query(conn: sqlite3.Connection, query: str) -> Optional[List[tuple]]:
    """Execute a query; returns rows or None if it fails, is denied, or runs too long."""
    try:
        conn.set_progress_handler(_make_step_limiter(), 1000)
        return [tuple(r) for r in conn.execute(query).fetchall()]
    except Exception:
        return None


def _normalize_value(v):
    return round(v, 6) if isinstance(v, float) else v


def results_match(gold_rows: List[tuple], gen_rows: List[tuple], ordered: bool) -> bool:
    gold = [tuple(_normalize_value(v) for v in r) for r in gold_rows]
    gen = [tuple(_normalize_value(v) for v in r) for r in gen_rows]
    if ordered:
        return gold == gen
    return Counter(gold) == Counter(gen)


def data_execution_check(setup_sql: str, gold_sql: str, gen_sql: str) -> Tuple[bool, bool, bool]:
    """Run both queries on the example's own data (CREATE + INSERT). Returns (gen_executed, match, gold_failed)."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(setup_sql)
        conn.set_authorizer(_deny_writes)
        gold_result = run_query(conn, gold_sql)
        gen_result = run_query(conn, gen_sql)
    finally:
        conn.close()
    if gold_result is None:
        return gen_result is not None, False, True
    if gen_result is None:
        return False, False, False
    return True, results_match(gold_result, gen_result, "ORDER BY" in gold_sql.upper()), False


def evaluate_sql(examples: List[Dict], generate_fn: Callable[[str], str]) -> Dict[str, float]:
    """Execution accuracy: gold and generated queries run on each example's own data."""
    from eval.stats import with_ci

    executed = exec_correct = string_matched = gold_errors = 0
    per_example = []
    n = len(examples)

    for ex in examples:
        gen_sql = clean_sql(generate_fn(ex["prompt"]))
        gold_sql = clean_sql(ex["gold_sql"])
        string_matched += normalize_sql(gen_sql) == normalize_sql(gold_sql)
        ran, correct, gold_failed = data_execution_check(ex["setup_sql"], gold_sql, gen_sql)
        executed += ran
        exec_correct += correct
        gold_errors += gold_failed
        per_example.append(int(correct))

    rate = lambda k: round(k / n, 4) if n else 0.0
    metrics = {
        "total_samples": n,
        "execution_rate": rate(executed),
        "execution_accuracy": rate(exec_correct),
        "string_match_rate": rate(string_matched),
        "executed_count": executed,
        "execution_correct_count": exec_correct,
        "string_match_count": string_matched,
        "gold_error_count": gold_errors,
        "per_example_correct": per_example,
    }
    return with_ci(metrics, "execution_accuracy", per_example)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SQL Generation Correctness")
    parser.add_argument("--data", type=str, default="data/eval/sql_gretel.jsonl",
                        help="Path to SQL evaluation dataset")
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
        gold_lookup = {ex["prompt"]: ex["gold_sql"] for ex in examples}
        results = evaluate_sql(examples, generate_fn=lambda p: gold_lookup[p])
        print(f"SQL Evaluation Results: {json.dumps(results, indent=2)}")
        assert results["execution_accuracy"] == 1.0, "Gold reference queries must achieve 100% execution accuracy!"
        print("Sanity test PASSED: 100% execution accuracy.")

if __name__ == "__main__":
    main()

"""Tests for the realistic evaluation sets, their scorers and the confidence-interval helpers."""
import json
import os

from eval.stats import bootstrap_ci, paired_delta_ci
from eval.sql_eval import evaluate_sql
from eval.json_eval import evaluate_json, evaluate_json_schema
from eval.code_eval import evaluate_code

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(rel):
    with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- confidence intervals -------------------------------------------------------

def test_bootstrap_ci_brackets_the_mean_and_narrows_with_n():
    small = bootstrap_ci([1, 0] * 10)
    large = bootstrap_ci([1, 0] * 200)
    assert small["low"] <= 0.5 <= small["high"]
    assert (large["high"] - large["low"]) < (small["high"] - small["low"])
    assert bootstrap_ci([1] * 30) == {"mean": 1.0, "low": 1.0, "high": 1.0, "n": 30}


def test_paired_delta_ci():
    base = [0, 0, 1, 1] * 25
    tuned = [1, 0, 1, 1] * 25          # +25 points on the same examples
    d = paired_delta_ci(base, tuned)
    assert d["mean"] == 0.25 and d["low"] > 0
    assert paired_delta_ci([1, 0], [1]) is None  # mismatched lengths are rejected


# --- SQL on real data (gretel) ----------------------------------------------------

def test_sql_real_data_gold_is_perfect_and_strict():
    examples = load("data/eval/sql_gretel.jsonl")[:40]
    gold = {ex["prompt"]: ex["gold_sql"] for ex in examples}
    res = evaluate_sql(examples, lambda p: gold[p])
    assert res["execution_accuracy"] == 1.0
    assert len(res["per_example_correct"]) == 40 and res["execution_accuracy_ci95"]["n"] == 40


def test_sql_real_data_rejects_wrong_filter():
    ex = {"prompt": "p", "gold_sql": "SELECT name FROM t WHERE city = 'Paris'",
          "setup_sql": "CREATE TABLE t (name TEXT, city TEXT); INSERT INTO t VALUES ('a','Paris'), ('b','paris');"}
    assert evaluate_sql([ex], lambda p: "SELECT name FROM t WHERE city = 'Paris'")["execution_accuracy"] == 1.0
    # With real data, value casing matters: 'paris' selects a different row.
    assert evaluate_sql([ex], lambda p: "SELECT name FROM t WHERE city = 'paris'")["execution_accuracy"] == 0.0


# --- code: HumanEval prelude ---------------------------------------------------------

def test_code_prelude_supplies_helpers_and_imports():
    ex = {"name": "uses_helper", "prompt": "p",
          "prelude": "from typing import List\n\ndef helper(x):\n    return x * 2\n",
          "assertions": "assert f([1, 2]) == [2, 4]"}
    model_output = "def f(xs: List[int]) -> List[int]:\n    return [helper(x) for x in xs]"
    res = evaluate_code([ex], lambda p: model_output, assume_yes=True, verbose=False)
    assert res["pass_at_1"] == 1.0


def test_humaneval_and_mbpp_gold_pass_sample():
    for rel in ["data/eval/code_humaneval.jsonl", "data/eval/code_mbpp.jsonl"]:
        examples = load(rel)[:10]
        gold = {ex["prompt"]: ex["gold_code"] for ex in examples}
        res = evaluate_code(examples, lambda p: gold[p], assume_yes=True, verbose=False)
        assert res["pass_at_1"] == 1.0, rel


# --- JSON ------------------------------------------------------------------------

def test_json_inscope_exact_match_requires_all_fields():
    ex = {"prompt": "p", "gold_json": {"user": "Priya Nair", "order_id": "44-9812", "amount": 87.40}}
    right = '{"user": "priya  nair", "order_id": "44-9812", "amount": 87.4}'   # case/space-insensitive name
    wrong_amount = '{"user": "Priya Nair", "order_id": "44-9812", "amount": 174.80}'
    assert evaluate_json([ex], lambda p: right)["exact_match_rate"] == 1.0
    res = evaluate_json([ex], lambda p: wrong_amount)
    assert res["schema_valid_rate"] == 1.0 and res["exact_match_rate"] == 0.0
    # Right value as a currency string: counts for exact match, but not as schema-valid.
    res = evaluate_json([ex], lambda p: '{"user": "Priya Nair", "order_id": "44-9812", "amount": "$87.40"}')
    assert res["exact_match_rate"] == 1.0 and res["schema_valid_rate"] == 0.0


def test_json_inscope_gold_is_perfect():
    examples = load("data/eval/json_inscope.jsonl")
    gold = {ex["prompt"]: json.dumps(ex["gold_json"]) for ex in examples}
    assert evaluate_json(examples, lambda p: gold[p])["exact_match_rate"] == 1.0


def test_json_schema_leaf_f1():
    ex = {"prompt": "p", "schema": {"type": "object"},
          "gold_json": {"a": 1, "b": {"c": "X", "d": [1, 2]}}}
    perfect = evaluate_json_schema([ex], lambda p: json.dumps(ex["gold_json"]))
    assert perfect["leaf_f1"] == 1.0 and perfect["exact_match_rate"] == 1.0
    partial = evaluate_json_schema([ex], lambda p: '{"a": 1, "b": {"c": "x"}}')   # 2 of 4 leaves, all correct
    assert partial["leaf_f1"] == round(2 * 1.0 * 0.5 / 1.5, 4)
    broken = evaluate_json_schema([ex], lambda p: "not json")
    assert broken["parse_rate"] == 0.0 and broken["leaf_f1"] == 0.0


def test_json_schema_validity_uses_the_example_schema():
    ex = {"prompt": "p", "gold_json": {"a": 1},
          "schema": {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}}
    assert evaluate_json_schema([ex], lambda p: '{"a": 1}')["schema_valid_rate"] == 1.0
    assert evaluate_json_schema([ex], lambda p: '{"a": "one"}')["schema_valid_rate"] == 0.0
    assert evaluate_json_schema([ex], lambda p: '{"b": 1}')["schema_valid_rate"] == 0.0


def test_run_eval_resume_reuses_saved_outputs(monkeypatch):
    import shutil
    import eval.run_eval as run_eval

    calls = []

    def fake_generator(url, model, max_tokens=256):
        return lambda p: calls.append(p) or '{"user": "x", "order_id": "1", "amount": 1}'

    monkeypatch.setattr(run_eval, "create_endpoint_generator", fake_generator)
    out_dir = os.path.join(REPO_ROOT, ".model_cache", "test_resume")  # inside the repo
    shutil.rmtree(out_dir, ignore_errors=True)
    try:
        run_eval.run_evaluation("json", "endpoint", "http://unused/v1/chat", raw_output_dir=out_dir)
        first = len(calls)
        res = run_eval.run_evaluation("json", "endpoint", "http://unused/v1/chat", raw_output_dir=out_dir, resume=True)
        assert first == 60 and len(calls) == first          # nothing regenerated on resume
        assert res["total_samples"] == 60
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_validation_sets_do_not_overlap_training_or_tests():
    for task, key in [("sql", "question"), ("json", "input_text")]:
        used = {r[key] for f in [f"data/{task}_train.jsonl", f"data/{task}_holdout.jsonl"] for r in load(f)}
        assert not used & {r[key] for r in load(f"data/{task}_val.jsonl")}, task
    code_val = {r["prompt"].splitlines()[0] for r in load("data/code_val.jsonl")}
    mbpp_test = {r["prompt"].splitlines()[0] for r in load("data/eval/code_mbpp.jsonl")}
    assert not code_val & mbpp_test

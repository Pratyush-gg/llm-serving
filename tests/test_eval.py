"""Tests for the SQL / code evaluators and the cascade router's input handling."""
import json
import os
import re

import pytest

from eval.sql_eval import data_execution_check, evaluate_sql, normalize_sql
from eval.code_eval import SANDBOX_ROOT, evaluate_code
from src.cascade import CascadeRouter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(name):
    with open(os.path.join(REPO_ROOT, "data", name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- SQL ----------------------------------------------------------------------

def test_sql_select_star_is_not_rewarded():
    examples = load("eval/sql_gretel.jsonl")

    def select_star(prompt):
        ex = next(e for e in examples if e["prompt"] == prompt)
        return "SELECT * FROM " + re.search(r"CREATE TABLE (\w+)", ex["setup_sql"]).group(1)

    assert evaluate_sql(examples, select_star)["execution_accuracy"] < 0.1


def test_sql_model_output_cannot_write_attach_or_run_away():
    setup = "CREATE TABLE t (a TEXT); INSERT INTO t VALUES ('x');"
    target = os.path.join(SANDBOX_ROOT, "should_never_exist.db")  # inside the repo, gitignored
    for query in [f"ATTACH DATABASE '{target}' AS x", "DROP TABLE t", "INSERT INTO t VALUES ('y')",
                  "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r) SELECT count(*) FROM r"]:
        assert data_execution_check(setup, "SELECT a FROM t", query)[0] is False, query
    assert not os.path.exists(target)


def test_sql_helpers():
    assert normalize_sql('SELECT  a FROM t WHERE b = "X";') == normalize_sql("select a from t where b='x'")


# --- Code ---------------------------------------------------------------------

def test_code_gold_passes_and_sandbox_is_cleaned_up():
    examples = load("code_holdout.jsonl")[:5]
    gold = {ex["prompt"]: ex["gold_code"] for ex in examples}
    res = evaluate_code(examples, lambda p: gold[p], assume_yes=True, verbose=False)
    assert res["pass_at_1"] == 1.0
    assert not os.path.exists(SANDBOX_ROOT) or os.listdir(SANDBOX_ROOT) == []


def test_code_failures_and_timeouts_are_counted():
    examples = [
        {"name": "wrong", "prompt": "a", "assertions": "assert f() == 1"},
        {"name": "hang", "prompt": "b", "assertions": "assert f() == 1"},
    ]
    outputs = {"a": "def f():\n    return 2", "b": "def f():\n    while True: pass"}
    res = evaluate_code(examples, lambda p: outputs[p], timeout_s=1.0, assume_yes=True, verbose=False)
    assert res["passed_count"] == 0
    assert res["error_count"] == 1
    assert res["timeout_count"] == 1


def test_code_eval_refuses_without_confirmation(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    with pytest.raises(SystemExit):
        evaluate_code([{"prompt": "p", "assertions": ""}], lambda p: "raise SystemExit(1)")


# --- Cascade ------------------------------------------------------------------

class FixedRouter:
    strategy_name = "fixed"

    def route_detailed(self, prompt):
        return {"route": "sql", "confidence": 0.95,
                "scores": {"sql": 0.95, "json": 0.03, "code": 0.01, "base": 0.01}, "latency_ms": 0.1}


def test_cascade_rejects_unknown_forced_adapter():
    cascade = CascadeRouter(base_router=FixedRouter())
    with pytest.raises(ValueError):
        cascade.route_and_generate("hi", lambda r, p: "x", force_adapter="nonexistent")


def test_cascade_forced_adapter_is_used():
    cascade = CascadeRouter(base_router=FixedRouter())
    res = cascade.route_and_generate("hi", lambda r, p: f"from {r}", force_adapter="code")
    assert res.final_adapter == "code"
    assert res.response == "from code"

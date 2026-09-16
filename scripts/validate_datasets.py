"""
Dataset Validation & Integrity Check for Day 1.
Verifies line counts, format, train/holdout disjointness,
and functional execution on holdout sets for SQL, JSON, and Code.
"""

import json
import os
import sqlite3
from pydantic import BaseModel, ValidationError

class ExtractionSchema(BaseModel):
    user: str
    order_id: str
    amount: float

def validate_sql():
    print("\n--- Validating SQL Datasets ---")
    train_path = os.path.join("data", "sql_train.jsonl")
    holdout_path = os.path.join("data", "sql_holdout.jsonl")
    
    with open(train_path, "r", encoding="utf-8") as f:
        train = [json.loads(line) for line in f]
    with open(holdout_path, "r", encoding="utf-8") as f:
        holdout = [json.loads(line) for line in f]
        
    print(f"SQL Train count: {len(train)} (expected: 600)")
    print(f"SQL Holdout count: {len(holdout)} (expected: 60)")
    assert len(train) == 600, f"Expected 600 train samples, got {len(train)}"
    assert len(holdout) == 60, f"Expected 60 holdout samples, got {len(holdout)}"
    
    # Check disjointness
    train_prompts = set(s["prompt"] for s in train)
    holdout_prompts = set(s["prompt"] for s in holdout)
    overlap = train_prompts.intersection(holdout_prompts)
    print(f"Train/Holdout prompt overlap: {len(overlap)} (expected: 0)")
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping prompts!"
    
    # Check SQLite execution for holdout
    executed = 0
    for s in holdout:
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(s["schema"])
            cur = conn.execute(s["gold_sql"])
            cur.fetchall()
            executed += 1
        except Exception as e:
            print(f"SQLite error on holdout: {e}\nSchema:\n{s['schema']}\nQuery:\n{s['gold_sql']}")
        finally:
            conn.close()
            
    print(f"SQL Holdout SQLite execution rate: {executed}/{len(holdout)} ({executed/len(holdout)*100:.1f}%)")
    assert executed == len(holdout), "All holdout SQL queries must execute successfully against their schemas"

def validate_json():
    print("\n--- Validating JSON Datasets ---")
    train_path = os.path.join("data", "json_train.jsonl")
    holdout_path = os.path.join("data", "json_holdout.jsonl")
    
    with open(train_path, "r", encoding="utf-8") as f:
        train = [json.loads(line) for line in f]
    with open(holdout_path, "r", encoding="utf-8") as f:
        holdout = [json.loads(line) for line in f]
        
    print(f"JSON Train count: {len(train)} (expected: 600)")
    print(f"JSON Holdout count: {len(holdout)} (expected: 60)")
    assert len(train) == 600, f"Expected 600 train samples, got {len(train)}"
    assert len(holdout) == 60, f"Expected 60 holdout samples, got {len(holdout)}"
    
    # Check disjointness
    train_texts = set(s["input_text"] for s in train)
    holdout_texts = set(s["input_text"] for s in holdout)
    overlap = train_texts.intersection(holdout_texts)
    print(f"Train/Holdout input_text overlap: {len(overlap)} (expected: 0)")
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping texts!"
    
    # Check Pydantic validation on gold_json
    valid_count = 0
    for s in holdout:
        gold = s["gold_json"]
        parsed = ExtractionSchema(**gold)
        assert parsed.user == gold["user"]
        assert parsed.order_id == gold["order_id"]
        assert parsed.amount == gold["amount"]
        valid_count += 1
        
    print(f"JSON Holdout schema validation rate: {valid_count}/{len(holdout)} (100%)")

def validate_code():
    print("\n--- Validating Code Datasets ---")
    train_path = os.path.join("data", "code_train.jsonl")
    holdout_path = os.path.join("data", "code_holdout.jsonl")
    
    with open(train_path, "r", encoding="utf-8") as f:
        train = [json.loads(line) for line in f]
    with open(holdout_path, "r", encoding="utf-8") as f:
        holdout = [json.loads(line) for line in f]
        
    print(f"Code Train count: {len(train)} (expected: 600)")
    print(f"Code Holdout count: {len(holdout)} (expected: 60)")
    assert len(train) == 600, f"Expected 600 train samples, got {len(train)}"
    assert len(holdout) == 60, f"Expected 60 holdout samples, got {len(holdout)}"
    
    # Check disjointness
    train_names = set(s["name"] for s in train)
    holdout_names = set(s["name"] for s in holdout)
    overlap = train_names.intersection(holdout_names)
    print(f"Train/Holdout task overlap: {len(overlap)} (expected: 0)")
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping task names!"
    
    # Check assertions on gold code
    passed = 0
    for s in holdout:
        scope = {}
        exec(s["gold_code"], scope)
        exec(s["assertions"], scope)
        passed += 1
        
    print(f"Code Holdout assertion pass@1 rate: {passed}/{len(holdout)} (100%)")

def main():
    print("========================================")
    print("DAY 1 DATASET INTEGRITY AND QUALITY TEST")
    print("========================================")
    validate_sql()
    validate_json()
    validate_code()
    print("\n>>> ALL DATASET INTEGRITY CHECKS PASSED (100% QUALITY) <<<\n")

if __name__ == "__main__":
    main()

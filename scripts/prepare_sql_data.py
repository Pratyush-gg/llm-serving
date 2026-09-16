"""
Dataset preparation script for SQL adapter.
Pulls from 'b-mc2/sql-create-context', filters for valid SQLite DDL,
and splits into 600 training examples and 60 held-out evaluation examples.
"""

import json
import os
import random
import sqlite3
from datasets import load_dataset

SEED = 42
TRAIN_COUNT = 600
HOLDOUT_COUNT = 60
TOTAL_COUNT = TRAIN_COUNT + HOLDOUT_COUNT

def is_valid_sqlite_schema(ddl: str) -> bool:
    """Validate that the schema DDL can be executed by SQLite."""
    try:
        conn = sqlite3.connect(":memory:")
        conn.executescript(ddl)
        conn.close()
        return True
    except Exception:
        return False

def format_sql_prompt(schema: str, question: str) -> str:
    return (
        f"You are an expert SQL assistant. Given the database schema context below, "
        f"write a SQL query that answers the question.\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        f"SQL Query:"
    )

def main():
    print("Loading 'b-mc2/sql-create-context' from Hugging Face...")
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    
    # Deterministic shuffle
    indices = list(range(len(ds)))
    random.seed(SEED)
    random.shuffle(indices)
    
    selected_samples = []
    print("Filtering and curating valid SQL samples...")
    for idx in indices:
        row = ds[idx]
        schema = row["context"].strip()
        question = row["question"].strip()
        gold_sql = row["answer"].strip()
        
        # Verify schema executes in SQLite
        if not is_valid_sqlite_schema(schema):
            continue
            
        prompt = format_sql_prompt(schema, question)
        
        selected_samples.append({
            "prompt": prompt,
            "schema": schema,
            "question": question,
            "gold_sql": gold_sql,
            "completion": gold_sql
        })
        
        if len(selected_samples) >= TOTAL_COUNT:
            break
            
    if len(selected_samples) < TOTAL_COUNT:
        raise RuntimeError(f"Found only {len(selected_samples)} valid samples, needed {TOTAL_COUNT}")
        
    train_samples = selected_samples[:TRAIN_COUNT]
    holdout_samples = selected_samples[TRAIN_COUNT:]
    
    os.makedirs("data", exist_ok=True)
    
    train_file = os.path.join("data", "sql_train.jsonl")
    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")
            
    holdout_file = os.path.join("data", "sql_holdout.jsonl")
    with open(holdout_file, "w", encoding="utf-8") as f:
        for s in holdout_samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"Generated {len(train_samples)} SQL training records -> {train_file}")
    print(f"Generated {len(holdout_samples)} SQL holdout records -> {holdout_file}")

if __name__ == "__main__":
    main()

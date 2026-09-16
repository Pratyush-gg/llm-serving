"""
JSON Extraction Adapter Correctness Evaluation Suite.
Evaluates generated JSON against Pydantic ExtractionSchema and checks field-level accuracy.
Metrics:
  - schema_valid_rate: Fraction of generated responses that parse as valid JSON conforming to schema.
  - field_accuracy: Accuracy across expected fields (user, order_id, amount) on valid responses.
"""

import json
import os
import re
import argparse
from typing import Callable, Dict, List
from pydantic import BaseModel, ValidationError

class ExtractionSchema(BaseModel):
    user: str
    order_id: str
    amount: float

def clean_json(raw_text: str) -> str:
    """Extract raw JSON string from potential markdown fences or surrounding chatter."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        # Fallback: find outermost { and }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]
    return text.strip()

def evaluate_json(examples: List[Dict], generate_fn: Callable[[str], str]) -> Dict[str, float]:
    valid = 0
    field_correct = 0
    field_total = 0
    n = len(examples)

    for ex in examples:
        raw_gen = generate_fn(ex["prompt"])
        cleaned = clean_json(raw_gen)
        gold = ex["gold_json"]

        try:
            parsed_dict = json.loads(cleaned)
            parsed = ExtractionSchema(**parsed_dict)
            valid += 1

            for field in ["user", "order_id", "amount"]:
                field_total += 1
                val_gen = getattr(parsed, field, None)
                val_gold = gold.get(field)

                if field == "amount" and isinstance(val_gen, (int, float)) and isinstance(val_gold, (int, float)):
                    if abs(float(val_gen) - float(val_gold)) < 1e-3:
                        field_correct += 1
                elif str(val_gen).strip() == str(val_gold).strip():
                    field_correct += 1
        except (json.JSONDecodeError, ValidationError, TypeError):
            # Sample failed schema validation or JSON decoding
            continue

    return {
        "total_samples": n,
        "schema_valid_rate": round(valid / n, 4) if n else 0.0,
        "field_accuracy": round(field_correct / field_total, 4) if field_total else 0.0,
        "valid_count": valid,
        "field_correct_count": field_correct,
        "field_total_count": field_total,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate JSON Extraction Correctness")
    parser.add_argument("--data", type=str, default="data/json_holdout.jsonl",
                        help="Path to JSON held-out evaluation dataset")
    parser.add_argument("--test-gold", action="store_true",
                        help="Sanity test evaluation harness using gold references (should be 100%)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Holdout file not found: {args.data}")

    with open(args.data, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(examples)} JSON evaluation examples from {args.data}")

    if args.test_gold:
        print("Running sanity test using reference gold_json completions...")
        results = evaluate_json(examples, generate_fn=lambda p: json.dumps([ex["gold_json"] for ex in examples if ex["prompt"] == p][0]))
        print(f"JSON Evaluation Results: {json.dumps(results, indent=2)}")
        assert results["schema_valid_rate"] == 1.0, "Gold references must achieve 100% schema validity!"
        assert results["field_accuracy"] == 1.0, "Gold references must achieve 100% field accuracy!"
        print("Sanity test PASSED: 100% schema validity and field accuracy.")

if __name__ == "__main__":
    main()

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

def _norm_text(v) -> str:
    return " ".join(str(v).split()).casefold()


def _to_number(v):
    """Parse amounts like 150, "59.90", "$1,240.75", "129,90 kr" or "R$ 3.200,00"; None if not a number."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = re.sub(r"[^\d.,-]", "", str(v)).strip(".,")  # strip(): "Rs. 2,499" -> "2,499"
    if "," in s and "." in s:  # the separator that comes last is the decimal point
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".") if re.fullmatch(r"-?\d+,\d{2}", s) else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _field_matches(field: str, val_gen, val_gold) -> bool:
    if field == "amount":
        num = _to_number(val_gen)
        return num is not None and abs(num - float(val_gold)) < 1e-3
    return isinstance(val_gen, str) and _norm_text(val_gen) == _norm_text(val_gold)


def evaluate_json(examples: List[Dict], generate_fn: Callable[[str], str]) -> Dict[str, float]:
    """Score {user, order_id, amount} extraction: schema validity, exact match, field accuracy."""
    from eval.stats import with_ci

    valid = exact = field_correct = field_total = 0
    per_valid, per_exact = [], []
    n = len(examples)

    for ex in examples:
        gold = ex["gold_json"]
        try:
            data = json.loads(clean_json(generate_fn(ex["prompt"])))
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            per_valid.append(0)
            per_exact.append(0)
            continue

        matches = [_field_matches(f, data.get(f), gold[f]) for f in ["user", "order_id", "amount"]]
        exact += all(matches)
        per_exact.append(int(all(matches)))
        try:
            ExtractionSchema(**data)
            is_valid = 1
        except (ValidationError, TypeError):
            is_valid = 0
        valid += is_valid
        per_valid.append(is_valid)
        if is_valid:
            field_total += 3
            field_correct += sum(matches)

    metrics = {
        "total_samples": n,
        "schema_valid_rate": round(valid / n, 4) if n else 0.0,
        "exact_match_rate": round(exact / n, 4) if n else 0.0,
        "field_accuracy": round(field_correct / field_total, 4) if field_total else 0.0,
        "valid_count": valid,
        "exact_match_count": exact,
        "field_correct_count": field_correct,
        "field_total_count": field_total,
        "per_example_correct": per_exact,
        "per_example_valid": per_valid,
    }
    with_ci(metrics, "schema_valid_rate", per_valid)
    return with_ci(metrics, "exact_match_rate", per_exact)


def _flatten(obj, prefix="") -> List[tuple]:
    """Flatten JSON into (path, normalized leaf value) pairs."""
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
        return out
    if isinstance(obj, list):
        out = []
        for i, v in enumerate(obj):
            out += _flatten(v, f"{prefix}[{i}]")
        return out
    if isinstance(obj, bool) or obj is None:
        return [(prefix, obj)]
    if isinstance(obj, (int, float)):
        return [(prefix, round(float(obj), 4))]
    return [(prefix, _norm_text(obj))]


def evaluate_json_schema(examples: List[Dict], generate_fn: Callable[[str], str]) -> Dict[str, float]:
    """Score free-form schema extraction: parse rate, schema validity, leaf F1, exact match."""
    from collections import Counter
    from eval.stats import with_ci

    try:
        import jsonschema  # optional dependency
    except ImportError:
        jsonschema = None

    per_parse, per_schema, per_f1, per_exact = [], [], [], []
    for ex in examples:
        raw_gen = generate_fn(ex["prompt"])
        try:
            parsed = json.loads(clean_json(raw_gen))
        except json.JSONDecodeError:
            per_parse.append(0)
            per_schema.append(0 if jsonschema else None)
            per_f1.append(0.0)
            per_exact.append(0)
            continue

        per_parse.append(1)
        if jsonschema:
            try:
                jsonschema.validate(parsed, ex["schema"])
                per_schema.append(1)
            except Exception:
                per_schema.append(0)
        else:
            per_schema.append(None)

        gold_leaves, pred_leaves = Counter(_flatten(ex["gold_json"])), Counter(_flatten(parsed))
        overlap = sum((gold_leaves & pred_leaves).values())
        precision = overlap / sum(pred_leaves.values()) if pred_leaves else 0.0
        recall = overlap / sum(gold_leaves.values()) if gold_leaves else 0.0
        per_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        per_exact.append(int(gold_leaves == pred_leaves))

    n = len(examples)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0
    schema_vals = [v for v in per_schema if v is not None]
    metrics = {
        "total_samples": n,
        "parse_rate": mean(per_parse),
        "schema_valid_rate": mean(schema_vals) if jsonschema else None,
        "leaf_f1": mean(per_f1),
        "exact_match_rate": mean(per_exact),
        "per_example_correct": per_exact,
        "per_example_leaf_f1": [round(x, 4) for x in per_f1],
    }
    with_ci(metrics, "leaf_f1", per_f1)
    with_ci(metrics, "parse_rate", per_parse)
    return metrics

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

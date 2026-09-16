"""
Python Code Generation Adapter Correctness Evaluation Suite.
Evaluates generated Python functions against paired unit assertions in an isolated subprocess.
Metrics:
  - pass_at_1: Fraction of generated functions that pass 100% of paired assertions within timeout.
"""

import json
import os
import re
import sys
import subprocess
import tempfile
import textwrap
import argparse
from typing import Callable, Dict, List

def clean_code(raw_text: str) -> str:
    """Extract clean Python code from potential markdown code fences or conversational wrappers."""
    text = raw_text.strip()
    match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text

def evaluate_code(examples: List[Dict], generate_fn: Callable[[str], str], timeout_s: float = 5.0) -> Dict[str, float]:
    passed = 0
    timeouts = 0
    errors = 0
    n = len(examples)

    for ex in examples:
        raw_gen = generate_fn(ex["prompt"])
        cleaned_fn = clean_code(raw_gen)

        test_script = f"{cleaned_fn.strip()}\n\n# --- Unit Assertions ---\n{ex['assertions'].strip()}\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(test_script)
            path = f.name

        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if result.returncode == 0:
                passed += 1
            else:
                errors += 1
        except subprocess.TimeoutExpired:
            timeouts += 1
        except Exception:
            errors += 1
        finally:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    return {
        "total_samples": n,
        "pass_at_1": round(passed / n, 4) if n else 0.0,
        "passed_count": passed,
        "timeout_count": timeouts,
        "error_count": errors,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Code Generation Correctness (Pass@1)")
    parser.add_argument("--data", type=str, default="data/code_holdout.jsonl",
                        help="Path to Code held-out evaluation dataset")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Subprocess execution timeout in seconds per sample")
    parser.add_argument("--test-gold", action="store_true",
                        help="Sanity test evaluation harness using gold references (should be 100%)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Holdout file not found: {args.data}")

    with open(args.data, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(examples)} Code evaluation examples from {args.data}")

    if args.test_gold:
        print("Running sanity test using reference gold_code completions...")
        results = evaluate_code(examples, generate_fn=lambda p: [ex["gold_code"] for ex in examples if ex["prompt"] == p][0], timeout_s=args.timeout)
        print(f"Code Evaluation Results: {json.dumps(results, indent=2)}")
        assert results["pass_at_1"] == 1.0, "Gold references must achieve 100% pass@1!"
        print("Sanity test PASSED: 100% pass@1.")

if __name__ == "__main__":
    main()

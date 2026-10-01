import json
import os
import re
import sys
import shutil
import subprocess
import tempfile
import argparse
from typing import Callable, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SANDBOX_ROOT = os.path.join(REPO_ROOT, ".eval_sandbox")


def clean_code(raw_text: str) -> str:
    """Extract clean Python code from potential markdown code fences or conversational wrappers."""
    text = raw_text.strip()
    match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text


def confirm_code_execution(num_programs: int, timeout_s: float, assume_yes: bool) -> None:
    """Explain what is about to run and ask for confirmation. Raises SystemExit if declined."""
    print("\n" + "-" * 72)
    print("CODE EVALUATION: this step RUNS Python code written by the model.")
    print(f"  Programs     : {num_programs} (one per test case, each followed by its unit assertions)")
    print(f"  Where        : a temporary folder inside {SANDBOX_ROOT}")
    print("                 (created now, deleted when evaluation finishes)")
    print(f"  Limits       : {timeout_s:g}s timeout each; isolated Python (-I); minimal environment;")
    print("                 TEMP and home folders point into the sandbox folder")
    print("  Caution      : this is basic protection, not a security boundary. A malicious program")
    print("                 could still reach files your user account can access.")
    print("-" * 72)

    if assume_yes:
        print("--yes was given: continuing without asking.\n")
        return
    if not sys.stdin or not sys.stdin.isatty():
        raise SystemExit("Refusing to run model-generated code without confirmation. "
                         "Re-run interactively, or pass --yes to confirm explicitly.")
    try:
        answer = input("Run these programs? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        raise SystemExit("Code evaluation cancelled by user. Nothing was executed.")
    print()


def _sandbox_env(run_dir: str) -> Dict[str, str]:
    env = {
        "TEMP": run_dir,
        "TMP": run_dir,
        "TMPDIR": run_dir,
        "HOME": run_dir,
        "USERPROFILE": run_dir,
        "PYTHONIOENCODING": "utf-8",
    }
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]  # required for Python to start on Windows
    return env


def _failure_reason(stderr: str) -> str:
    lines = [l for l in stderr.strip().splitlines() if l.strip()]
    return lines[-1][:100] if lines else "non-zero exit"


def evaluate_code(
    examples: List[Dict],
    generate_fn: Callable[[str], str],
    timeout_s: float = 5.0,
    assume_yes: bool = False,
    verbose: bool = True,
) -> Dict[str, float]:
    passed = 0
    timeouts = 0
    errors = 0
    n = len(examples)

    confirm_code_execution(n, timeout_s, assume_yes)
    per_example = []

    os.makedirs(SANDBOX_ROOT, exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix="run_", dir=SANDBOX_ROOT)
    env = _sandbox_env(run_dir)
    if verbose:
        print(f"Created sandbox folder: {run_dir}")

    try:
        for i, ex in enumerate(examples, start=1):
            raw_gen = generate_fn(ex["prompt"])
            cleaned_fn = clean_code(raw_gen)
            # Imports/helpers from the problem statement run first, as in the official harnesses.
            prelude = ex.get("prelude", "").strip()
            test_script = (f"{prelude}\n\n" if prelude else "") + \
                f"{cleaned_fn.strip()}\n\n# --- Unit Assertions ---\n{ex['assertions'].strip()}\n"

            path = os.path.join(run_dir, f"sample_{i:03d}.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(test_script)

            label = ex.get("name", f"sample_{i}")
            passed_before = passed
            try:
                result = subprocess.run(
                    [sys.executable, "-I", "-B", path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    cwd=run_dir,
                    env=env,
                )
                if result.returncode == 0:
                    passed += 1
                    status = "PASS"
                else:
                    errors += 1
                    status = f"FAIL ({_failure_reason(result.stderr)})"
            except subprocess.TimeoutExpired:
                timeouts += 1
                status = f"TIMEOUT (>{timeout_s:g}s)"
            except Exception as e:
                errors += 1
                status = f"ERROR ({type(e).__name__})"

            per_example.append(int(passed > passed_before))
            if verbose:
                print(f"  [{i}/{n}] {label}: {status}")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        if verbose:
            print(f"Deleted sandbox folder: {run_dir}")

    from eval.stats import with_ci
    metrics = {
        "total_samples": n,
        "pass_at_1": round(passed / n, 4) if n else 0.0,
        "passed_count": passed,
        "timeout_count": timeouts,
        "error_count": errors,
        "per_example_correct": per_example,
    }
    return with_ci(metrics, "pass_at_1", per_example)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Code Generation Correctness (Pass@1)")
    parser.add_argument("--data", type=str, default="data/code_holdout.jsonl",
                        help="Path to Code held-out evaluation dataset")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Subprocess execution timeout in seconds per sample")
    parser.add_argument("--test-gold", action="store_true",
                        help="Sanity test evaluation harness using gold references (should be 100%)")
    parser.add_argument("--yes", action="store_true",
                        help="Confirm running code without an interactive prompt")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Holdout file not found: {args.data}")

    with open(args.data, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(examples)} Code evaluation examples from {args.data}")

    if args.test_gold:
        print("Running sanity test using reference gold_code completions...")
        gold_lookup = {ex["prompt"]: ex["gold_code"] for ex in examples}
        results = evaluate_code(examples, generate_fn=lambda p: gold_lookup[p],
                                timeout_s=args.timeout, assume_yes=args.yes)
        print(f"Code Evaluation Results: {json.dumps(results, indent=2)}")
        assert results["pass_at_1"] == 1.0, "Gold references must achieve 100% pass@1!"
        print("Sanity test PASSED: 100% pass@1.")

if __name__ == "__main__":
    main()

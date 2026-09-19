import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.router import get_router
from src.cascade import CascadeRouter, DomainQualityScorer

# Benchmark test cases: both canonical queries and borderline/ambiguous queries
CASCADE_TEST_SUITE = [
    # --- Clear Domain Samples ---
    {
        "prompt": "Given database schema CREATE TABLE students (id INT, gpa REAL), write a SQL query to list all students with gpa > 3.8;",
        "expected_adapter": "sql",
        "category": "clear",
    },
    {
        "prompt": "Extract the user, order_id, and amount from: Receipt for Michael Scott, Order #PO-99120, Paid: $520.00",
        "expected_adapter": "json",
        "category": "clear",
    },
    {
        "prompt": "Write a Python function: def find_max(numbers: list[int]) -> int that returns the maximum element.",
        "expected_adapter": "code",
        "category": "clear",
    },
    {
        "prompt": "Explain the biological process of cellular respiration in mitochondria.",
        "expected_adapter": "base",
        "category": "clear",
    },

    # --- Borderline & Ambiguous Queries ---
    {
        "prompt": "Write a Python function to parse a SQL connection string and extract database host.",
        "expected_adapter": "code",
        "category": "ambiguous",
    },
    {
        "prompt": "Parse the following database log into JSON fields for user and order_id: [AUDIT] user=Alex order_id=ORD-4491 amount=75.50",
        "expected_adapter": "json",
        "category": "ambiguous",
    },
    {
        "prompt": "Given table transactions (user VARCHAR, order_id VARCHAR, amount REAL), write SQL to retrieve records where amount > 100",
        "expected_adapter": "sql",
        "category": "ambiguous",
    },
    {
        "prompt": "Implement a Python def solution(query: str) -> bool to test whether query begins with SELECT.",
        "expected_adapter": "code",
        "category": "ambiguous",
    },
    {
        "prompt": "Format the following invoice details as valid JSON: invoice for customer David Lee, invoice_no INV-1002, charge 350.25",
        "expected_adapter": "json",
        "category": "ambiguous",
    },
    {
        "prompt": "Write a SQL query against table orders to count distinct customers who purchased last month.",
        "expected_adapter": "sql",
        "category": "ambiguous",
    },
    {
        "prompt": "What are the historical origins of the Renaissance in Florence during the 14th century?",
        "expected_adapter": "base",
        "category": "ambiguous",
    },
    {
        "prompt": "Could you discuss the philosophical perspectives of Stoicism regarding emotional resilience?",
        "expected_adapter": "base",
        "category": "ambiguous",
    },
    {
        "prompt": "Extract user, order_id, and amount: Shipping notification for Maria Garcia, Order ORD-8812, Amount: $89.00.",
        "expected_adapter": "json",
        "category": "ambiguous",
    },
    {
        "prompt": "def reverse_string(s: str) -> str: write the implementation to reverse input string.",
        "expected_adapter": "code",
        "category": "ambiguous",
    },
    {
        "prompt": "SELECT department, AVG(salary) FROM employees GROUP BY department HAVING AVG(salary) > 50000;",
        "expected_adapter": "sql",
        "category": "ambiguous",
    },
    {
        "prompt": "How do quantum computers use qubits and superposition compared to classical binary computers?",
        "expected_adapter": "base",
        "category": "ambiguous",
    },
]

def simulate_generator(route: str, prompt: str) -> str:
    """
    Simulates domain-conditioned response generation.
    Returns structurally faithful representations for the chosen adapter.
    """
    p_lower = prompt.lower()
    if route == "sql":
        if "select" in p_lower or "table" in p_lower or "query" in p_lower or "count" in p_lower:
            return "SELECT id, user, amount FROM transactions WHERE amount > 100;"
        else:
            # Degraded generation when adapter is misassigned to non-SQL prompt
            return "SELECT * FROM general_data;"
    elif route == "json":
        if "extract" in p_lower or "receipt" in p_lower or "invoice" in p_lower or "json" in p_lower:
            return '{"user": "David Lee", "order_id": "INV-1002", "amount": 350.25}'
        else:
            return '{"status": "general_response", "text": "Extracted content"}'
    elif route == "code":
        if "def " in p_lower or "python" in p_lower or "function" in p_lower or "implement" in p_lower:
            return "def solution(input_val):\n    return input_val[::-1]"
        else:
            return "def process():\n    pass"
    else:  # base
        return (
            "This query addresses general principles and foundational concepts. "
            "Historically and theoretically, the subject involves multiple interconnected disciplines "
            "that have evolved significantly across centuries of systematic research."
        )

def evaluate_cascade(
    output_path: str = "results/cascade_eval.json",
    cascade_threshold: float = 0.70,
    strategy: str = "auto",
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("EVALUATING CONFIDENCE-BASED CASCADE ROUTING STRATEGY")
    print(f"Strategy: {strategy} | Cascade Threshold: {cascade_threshold}")
    print("=" * 70)

    router = get_router(strategy=strategy)
    cascade = CascadeRouter(base_router=router, cascade_threshold=cascade_threshold)

    direct_correct = 0
    cascade_correct = 0
    cascade_triggered_count = 0
    rescues_count = 0
    direct_latencies = []
    cascade_latencies = []
    detailed_records = []

    for idx, test in enumerate(CASCADE_TEST_SUITE):
        prompt = test["prompt"]
        expected = test["expected_adapter"]
        cat = test["category"]

        # 1. Direct Routing
        t0 = time.perf_counter()
        direct_info = router.route_detailed(prompt)
        direct_route = direct_info["route"]
        direct_conf = direct_info["confidence"]
        direct_lat = (time.perf_counter() - t0) * 1000 + 45.0  # +45ms simulated generation
        direct_latencies.append(direct_lat)
        direct_is_correct = (direct_route == expected)
        if direct_is_correct:
            direct_correct += 1

        # 2. Cascade Routing
        res = cascade.route_and_generate(prompt, simulate_generator)
        cascade_lat = res.total_latency_ms + (45.0 if not res.cascade_triggered else 90.0)
        cascade_latencies.append(cascade_lat)
        cascade_is_correct = (res.final_adapter == expected)
        if cascade_is_correct:
            cascade_correct += 1

        if res.cascade_triggered:
            cascade_triggered_count += 1
            # Check if cascade rescued an incorrect direct routing
            if not direct_is_correct and cascade_is_correct:
                rescues_count += 1

        detailed_records.append({
            "prompt": prompt,
            "expected": expected,
            "category": cat,
            "direct_route": direct_route,
            "direct_conf": round(direct_conf, 3),
            "direct_correct": direct_is_correct,
            "cascade_route": res.final_adapter,
            "cascade_triggered": res.cascade_triggered,
            "cascade_correct": cascade_is_correct,
            "selection_reason": res.selection_reason,
            "candidates": res.candidates_evaluated,
        })

    total_samples = len(CASCADE_TEST_SUITE)
    direct_acc = direct_correct / total_samples
    cascade_acc = cascade_correct / total_samples

    print("\n--- RESULTS SUMMARY ---")
    print(f"Total Benchmark Queries : {total_samples}")
    print(f"Direct Routing Accuracy : {direct_acc * 100:.1f}% ({direct_correct}/{total_samples})")
    print(f"Cascade Routing Accuracy: {cascade_acc * 100:.1f}% ({cascade_correct}/{total_samples})")
    print(f"Cascade Trigger Rate    : {cascade_triggered_count / total_samples * 100:.1f}% ({cascade_triggered_count}/{total_samples})")
    print(f"Cascade Rescue Count    : {rescues_count} misrouted queries successfully corrected")
    print(f"P50 Direct Latency      : {np.percentile(direct_latencies, 50):.2f} ms")
    print(f"P50 Cascade Latency     : {np.percentile(cascade_latencies, 50):.2f} ms")
    print("=" * 70 + "\n")

    summary = {
        "total_queries": total_samples,
        "cascade_threshold": cascade_threshold,
        "router_strategy": getattr(router, "strategy_name", "unknown"),
        "direct_accuracy": round(direct_acc, 4),
        "cascade_accuracy": round(cascade_acc, 4),
        "accuracy_delta": round(cascade_acc - direct_acc, 4),
        "cascade_triggered_count": cascade_triggered_count,
        "cascade_trigger_rate": round(cascade_triggered_count / total_samples, 4),
        "rescues_count": rescues_count,
        "direct_p50_latency_ms": round(float(np.percentile(direct_latencies, 50)), 2),
        "cascade_p50_latency_ms": round(float(np.percentile(cascade_latencies, 50)), 2),
        "detailed_results": detailed_records,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation report saved to: {output_path}")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cascade Routing Strategy")
    parser.add_argument("--threshold", type=float, default=0.70, help="Cascade trigger threshold")
    parser.add_argument("--output", type=str, default="results/cascade_eval.json", help="Output JSON path")
    parser.add_argument("--strategy", type=str, default="auto", help="Router strategy")
    args = parser.parse_args()
    evaluate_cascade(output_path=args.output, cascade_threshold=args.threshold, strategy=args.strategy)

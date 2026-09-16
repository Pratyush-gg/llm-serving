"""
Automated Smoke Test Suite for Serving Gateway.
Tests health endpoint and validates semantic routing across:
  - SQL query -> sql-adapter
  - JSON extraction -> json-adapter
  - Code generation -> code-adapter
  - Out-of-domain query -> base model fallback
"""

import sys
import json
import time
import argparse
import requests

SMOKE_TESTS = [
    {
        "domain": "SQL",
        "prompt": "Given the database schema CREATE TABLE employees (id INT, salary INT), write a query to find the maximum salary.",
        "expected_adapter": "sql",
    },
    {
        "domain": "JSON",
        "prompt": "Extract user, order_id, and amount from: Order Confirmation for Sarah Connor, order #TXN-88219, total charge $149.95.",
        "expected_adapter": "json",
    },
    {
        "domain": "Code",
        "prompt": "Write a Python function to calculate the greatest common divisor (GCD) of two integers a and b.",
        "expected_adapter": "code",
    },
    {
        "domain": "Base (General)",
        "prompt": "Explain the historical significance of the Magna Carta in three concise bullet points.",
        "expected_adapter": "base",
    },
]

def run_smoke_test(base_url: str = "http://localhost:8080") -> bool:
    chat_url = f"{base_url.rstrip('/')}/v1/chat"
    health_url = f"{base_url.rstrip('/')}/health"

    print("\n" + "=" * 70)
    print(f"RUNNING SERVING GATEWAY SMOKE TEST AGAINST: {base_url}")
    print("=" * 70)

    # 1. Health Check
    print("\n[Step 1] Checking Gateway Health Endpoint...")
    try:
        r_health = requests.get(health_url, timeout=5.0)
        if r_health.status_code == 200:
            print(f"[OK] Health Check OK: {r_health.json()}")
        else:
            print(f"[FAIL] Health Check returned status {r_health.status_code}: {r_health.text}")
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Could not connect to gateway at {health_url}: {e}")
        return False

    # 2. Test Routing on All 4 Query Domains
    print("\n[Step 2] Sending Test Queries Across 4 Routing Domains...")
    all_passed = True
    results_table = []

    for test in SMOKE_TESTS:
        prompt = test["prompt"]
        expected = test["expected_adapter"]
        domain = test["domain"]

        t0 = time.perf_counter()
        try:
            resp = requests.post(chat_url, json={"prompt": prompt, "max_tokens": 128}, timeout=30.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if resp.status_code == 200:
                data = resp.json()
                adapter_used = data.get("adapter_used")
                routing_ms = data.get("routing_latency_ms", 0.0)
                confidence = data.get("router_confidence", 0.0)
                response_text = data.get("response", "").strip().replace("\n", " ")[:45]

                match = (adapter_used == expected)
                if not match:
                    all_passed = False

                results_table.append({
                    "domain": domain,
                    "expected": expected,
                    "actual": adapter_used,
                    "confidence": f"{confidence:.3f}",
                    "route_ms": f"{routing_ms:.1f}ms",
                    "total_ms": f"{elapsed_ms:.1f}ms",
                    "status": "PASS" if match else "FAIL",
                    "snippet": response_text,
                })
            else:
                all_passed = False
                results_table.append({
                    "domain": domain,
                    "expected": expected,
                    "actual": f"HTTP {resp.status_code}",
                    "confidence": "-",
                    "route_ms": "-",
                    "total_ms": f"{elapsed_ms:.1f}ms",
                    "status": "ERROR",
                    "snippet": resp.text[:40],
                })
        except Exception as e:
            all_passed = False
            results_table.append({
                "domain": domain,
                "expected": expected,
                "actual": "EXCEPTION",
                "confidence": "-",
                "route_ms": "-",
                "total_ms": "-",
                "status": "ERROR",
                "snippet": str(e)[:40],
            })

    # Print Formatted Results Table
    print("\n" + "-" * 75)
    print(f"{'Domain':<12} | {'Expected':<8} | {'Routed To':<8} | {'Conf':<6} | {'Route':<7} | {'Total':<7} | {'Status':<6}")
    print("-" * 75)
    for r in results_table:
        print(f"{r['domain']:<12} | {r['expected']:<8} | {r['actual']:<8} | {r['confidence']:<6} | {r['route_ms']:<7} | {r['total_ms']:<7} | {r['status']:<6}")
    print("-" * 75)

    if all_passed:
        print("\n>>> ALL SMOKE TESTS PASSED SUCCESSFULLY! <<<\n")
    else:
        print("\n>>> SMOKE TEST FAILED: One or more routing outcomes did not match. <<<\n")

    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke Test Serving Gateway")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="Gateway URL")
    args = parser.parse_args()

    success = run_smoke_test(base_url=args.url)
    sys.exit(0 if success else 1)

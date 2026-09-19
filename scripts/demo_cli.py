import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.router import get_router

DEMO_PRESETS = [
    ("SQL Query", "Given schema CREATE TABLE users (id INT, created_at DATE, active BOOL), write a SQL query: SELECT count(*) FROM users WHERE created_at > '2026-01-01' AND active = 1;"),
    ("JSON Extraction", "Extract the user, order_id, and amount from the following text as a JSON object: Order Confirmation for Sarah Jenkins, order #TXN-98421 totaling $149.50."),
    ("Python Code", "Write a Python function with implementation: def is_anagram(s1: str, s2: str) -> bool that checks whether s1 is an anagram of s2."),
    ("Base (Fallback)", "Explain the difference between supervised, unsupervised, and reinforcement learning in machine learning."),
]

def run_interactive_demo():
    print("=" * 75)
    print("ROUTED MULTI-ADAPTER LLM SERVING SYSTEM - LIVE DEMO")
    print("=" * 75)
    print("Powered by BAAI/bge-small-en-v1.5 Semantic Router + vLLM Multi-LoRA Engine")
    print("Threshold: 0.60 | Supported Adapters: sql-adapter, json-adapter, code-adapter, base")
    print("=" * 75)

    router = get_router(threshold=0.60)

    # 1. Run Presets
    print("\n--- Running Automated Preset Demonstrations ---")
    for category, prompt in DEMO_PRESETS:
        res = router.route_detailed(prompt)
        print(f"\n[Category: {category}]")
        print(f"Prompt: \"{prompt}\"")
        print(f"-> Selected Route  : {res['route'].upper()}")
        print(f"-> Router Conf     : {res['confidence']:.4f}")
        print(f"-> Routing Latency : {res['latency_ms']:.2f} ms")
        print(f"-> All Scores      : {res['scores']}")
        time.sleep(0.3)

    print("\n" + "=" * 75)
    print("--- Interactive Mode: Enter Your Own Prompts (Type 'exit' to quit) ---")
    print("=" * 75)

    while True:
        try:
            user_input = input("\nEnter prompt > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting live demo. Thank you!")
                break

            res = router.route_detailed(user_input)
            print(f"-> Selected Route  : {res['route'].upper()}")
            print(f"-> Router Conf     : {res['confidence']:.4f}")
            print(f"-> Routing Latency : {res['latency_ms']:.2f} ms")
            print(f"-> All Scores      : {res['scores']}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting live demo.")
            break

if __name__ == "__main__":
    run_interactive_demo()

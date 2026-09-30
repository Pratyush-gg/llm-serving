"""
Run the full measured benchmark suite on the local GPU, in order:

  1. VRAM profile (alone on the GPU)
  2. start the gateway (PEFT engine)
  3. adapter quality eval, then base-model baseline eval (raw outputs saved)
  4. live latency benchmark
  5. router eval on the TEST half of the router test set (centroid, learned v1, learned v2)
  6. cascade: calibrate thresholds on the CALIBRATION half, evaluate on the TEST half
  7. stop the gateway, build the combined report

Before running: plug in, set Windows power mode to "Best performance", close heavy apps.
The code eval runs model-written programs in .eval_sandbox/ and asks first unless --yes is given.

Usage:
  python scripts/run_all_benchmarks.py            # interactive (asks before running model code)
  python scripts/run_all_benchmarks.py --yes      # approve running model-generated code up front
  python scripts/run_all_benchmarks.py --yes --resume   # continue after stopping (Ctrl+C) mid-run
"""
import os
import sys
import time
import argparse
import subprocess

import requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(REPO_ROOT, ".model_cache", "logs")
PY = sys.executable


def banner(text: str):
    print("\n" + "#" * 78 + f"\n# {text}\n" + "#" * 78, flush=True)


def run(step: str, args, yes_flag: bool = False, assume_yes: bool = False):
    cmd = [PY, *args] + (["--yes"] if yes_flag and assume_yes else [])
    banner(step)
    print("$ " + " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    print(f"[done in {time.time() - t0:.0f}s] {step}", flush=True)


def port_in_use(port: int) -> bool:
    try:
        requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def start_gateway(port: int):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"gateway_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log = open(log_path, "w", encoding="utf-8")
    banner(f"Starting gateway on port {port} (log: {os.path.relpath(log_path, REPO_ROOT)})")
    proc = subprocess.Popen([PY, "-m", "src.gateway", "--engine", "peft", "--port", str(port)],
                            cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Gateway exited early (code {proc.returncode}); see {log_path}")
        if port_in_use(port):
            print("Gateway is up.", flush=True)
            return proc, log
        time.sleep(2)
    proc.terminate()
    raise RuntimeError(f"Gateway did not become healthy within 10 minutes; see {log_path}")


def stop_gateway(proc, log):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Gateway stopped.", flush=True)
    if log:
        log.close()


def main():
    parser = argparse.ArgumentParser(description="Run the full measured benchmark suite")
    parser.add_argument("--yes", action="store_true", help="Approve running model-generated code (code eval)")
    parser.add_argument("--no-pause", action="store_true", help="Do not wait for Enter after the checklist")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--latency-requests", type=int, default=100)
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run: reuse generations saved in results/raw_outputs "
                             "(short steps - VRAM, latency, routers - simply rerun)")
    args = parser.parse_args()
    resume = ["--resume"] if args.resume else []

    endpoint = f"http://127.0.0.1:{args.port}"
    banner("Checklist before measuring")
    print("  - laptop plugged in, Windows power mode set to 'Best performance'")
    print("  - browsers, games and other GPU/CPU-heavy apps closed")
    print("  - the laptop will not sleep for ~4-5 hours (rough estimate: 861 test examples x 2 models,")
    print("    plus latency, router and cascade runs)")
    print(f"  - code eval: {'approved via --yes' if args.yes else 'will ask for confirmation when it starts'}")
    if not args.no_pause:
        input("\nPress Enter to start (Ctrl+C to abort)... ")

    if port_in_use(args.port):
        raise SystemExit(f"Something is already serving on port {args.port}; stop it first.")

    t_start = time.time()
    run("1/7 VRAM profile", ["scripts/profile_memory.py"])

    proc, log = start_gateway(args.port)
    try:
        run("3a/7 Adapter quality eval (sql / json / code adapters)",
            ["-m", "eval.run_eval", "--task", "all", "--backend", "endpoint",
             "--endpoint-url", f"{endpoint}/v1/chat", "--output-json", "results/correctness_results.json", *resume],
            yes_flag=True, assume_yes=args.yes)
        run("3b/7 Baseline eval (base model, no adapter)",
            ["-m", "eval.baseline_eval", "--endpoint-url", f"{endpoint}/v1/chat",
             "--output", "results/baseline_vs_tuned.json", *resume],
            yes_flag=True, assume_yes=args.yes)
        run("4/7 Live latency benchmark",
            ["scripts/run_benchmarks.py", "--endpoint", endpoint, "--count", str(args.latency_requests)])
        run("5a/7 Router eval: centroid", ["eval/router_eval.py", "--strategy", "centroid",
                                           "--output", "results/router_eval_centroid.json"])
        run("5b/7 Router eval: learned (v2, default)", ["eval/router_eval.py", "--strategy", "learned",
                                                         "--output", "results/router_eval_learned.json"])
        run("6a/7 Cascade calibration (calibration half)",
            ["eval/calibrate_cascade.py", "--endpoint", endpoint, "--strategy", "auto",
             "--output", "results/cascade_calibration.json", *resume])
        run("6b/7 Cascade eval (test half, calibrated thresholds)",
            ["eval/eval_cascade.py", "--endpoint", endpoint, "--strategy", "auto",
             "--calibration", "results/cascade_calibration.json", "--output", "results/cascade_eval.json", *resume])
    finally:
        stop_gateway(proc, log)

    run("7/7 Combined report", ["scripts/generate_combined_report.py"])
    banner(f"All benchmarks finished in {(time.time() - t_start) / 60:.0f} minutes")


if __name__ == "__main__":
    main()

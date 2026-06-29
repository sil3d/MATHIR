"""
Orchestrator for the v8.5.0 lifecycle benchmarks.

Runs the full suite in sequence:
  1. micro_bench.py   — memory-only throughput (~5 min)
  2. ai_cognitive_bench.py — AI-driven end-to-end (~duration minutes)

Usage:
  python run_all.py --duration 20
  python run_all.py --duration 20 --micro-count 1000 --experiences 50
  python run_all.py --skip-micro --duration 30
  python run_all.py --skip-ai
"""
import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


BENCH_DIR = Path(__file__).resolve().parent


def run_step(name: str, cmd: list, results_dir: Path) -> dict:
    print(f"\n{'='*70}")
    print(f"  STEP: {name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*70}\n")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(BENCH_DIR))
    wall = time.perf_counter() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL ({proc.returncode})"
    print(f"\n  [{name}] {status} in {wall:.1f}s")
    return {"name": name, "cmd": cmd, "wall_s": round(wall, 1), "returncode": proc.returncode}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=20,
                   help="AI cognitive bench duration in minutes (default 20)")
    p.add_argument("--micro-count", type=int, default=1000,
                   help="Number of memories for micro-bench (default 1000)")
    p.add_argument("--experiences", type=int, default=50,
                   help="Number of LLM-generated experiences (default 50)")
    p.add_argument("--questions", type=int, default=20,
                   help="Number of Q&A pairs (default 20)")
    p.add_argument("--skip-micro", action="store_true")
    p.add_argument("--skip-ai", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    args = p.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {out_dir}")

    summary = {"timestamp": timestamp, "config": vars(args), "steps": []}

    if not args.skip_micro:
        micro_out = out_dir / "micro.json"
        summary["steps"].append(run_step(
            "micro_bench",
            [sys.executable, str(BENCH_DIR / "micro_bench.py"),
             "--count", str(args.micro_count),
             "--out", str(micro_out)],
            out_dir,
        ))

    if not args.skip_ai:
        ai_out = out_dir / "ai_cognitive.json"
        summary["steps"].append(run_step(
            "ai_cognitive_bench",
            [sys.executable, str(BENCH_DIR / "ai_cognitive_bench.py"),
             "--experiences", str(args.experiences),
             "--questions", str(args.questions),
             "--duration", str(args.duration),
             "--out", str(ai_out)],
            out_dir,
        ))

    # Write summary
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*70}")
    print(f"  ALL DONE — summary at {summary_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

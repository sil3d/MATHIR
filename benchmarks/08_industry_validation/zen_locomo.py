#!/usr/bin/env python3
"""
Run LoCoMo benchmark via OpenCode Zen (no TPM limit like Groq).

Tests: mimo-v2.5-free (answer) + deepseek-v4-flash-free (judge)
Then: minimax-m3 (answer) + minimax-m3 (judge) for comparison.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import _env  # noqa
except ImportError:
    pass

zen_key = os.environ.get("OPENCODE_ZEN_KEY", "")
if not zen_key:
    print("ERROR: OPENCODE_ZEN_KEY not set in benchmarks/.env")
    sys.exit(1)

os.environ["MATHIR_LLM_BACKEND"] = "api"
os.environ["MATHIR_API_BASE"] = "https://opencode.ai/zen/v1"
os.environ["MATHIR_API_KEY"] = zen_key

# Phase 1: free models
ANSWER_MODEL = os.environ.get("ZEN_ANSWER_MODEL", "mimo-v2.5-free")
JUDGE_MODEL = os.environ.get("ZEN_JUDGE_MODEL", "deepseek-v4-flash-free")

os.environ["MATHIR_API_MODEL"] = ANSWER_MODEL
os.environ["MATHIR_BENCHMARK_ANSWER_MODEL"] = ANSWER_MODEL
os.environ["MATHIR_BENCHMARK_JUDGE_MODEL"] = JUDGE_MODEL
os.environ["MATHIR_BENCHMARK_ANSWER_MAX_TOKENS"] = "4096"
os.environ["MATHIR_BENCHMARK_JUDGE_MAX_TOKENS"] = "1024"

print(f"Answer model: {ANSWER_MODEL}")
print(f"Judge model:  {JUDGE_MODEL}")
print(f"API base:     https://opencode.ai/zen/v1")
print()

sys.argv = [
    "run_locomo.py",
    "--conversations", "2",
    "--categories", "1,2,3,4",
    "--k", "10",
    "--output", str(Path(__file__).resolve().parent.parent / "06_results" / "current" / "zen_locomo_results.json"),
]

from run_locomo import main
main()

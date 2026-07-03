#!/usr/bin/env python3
"""
Run LoCoMo benchmark with Llama 3.3 70B via Groq.

Reuses run_locomo.py's infrastructure but forces Groq as the LLM backend
by setting the right env vars before importing llm_client.
"""
import os
import sys
from pathlib import Path

# Force Groq as the LLM backend
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import _env  # noqa - loads .env
except ImportError:
    pass

groq_key = os.environ.get("GROQ_API_KEY", "")
if not groq_key:
    print("ERROR: GROQ_API_KEY not set in benchmarks/.env")
    sys.exit(1)

os.environ["MATHIR_LLM_BACKEND"] = "api"
os.environ["MATHIR_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["MATHIR_API_KEY"] = groq_key
os.environ["MATHIR_API_MODEL"] = "llama-3.3-70b-versatile"
os.environ["MATHIR_BENCHMARK_ANSWER_MODEL"] = "llama-3.3-70b-versatile"
# gpt-oss-120b as judge: no thinking tokens = lower TPM usage
os.environ["MATHIR_BENCHMARK_JUDGE_MODEL"] = "openai/gpt-oss-120b"

# Groq free tier has 12K TPM limit — reduce k to 5 to fit context
sys.argv = [
    "run_locomo.py",
    "--conversations", "2",
    "--categories", "1,2,3,4",
    "--k", "5",
    "--output", str(Path(__file__).resolve().parent.parent / "06_results" / "current" / "groq_locomo_results.json"),
]

# Groq free TPM is 12K — llama thinking inflates tokens 10x
os.environ["MATHIR_BENCHMARK_ANSWER_MAX_TOKENS"] = "1024"
os.environ["MATHIR_BENCHMARK_JUDGE_MAX_TOKENS"] = "256"
os.environ["MATHIR_BENCHMARK_CONTEXT_MAX_CHARS"] = "3000"

# Import and run
from run_locomo import main
main()

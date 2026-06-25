# MATHIR Benchmarks

## Directory Structure

```
benchmarks/
├── README.md
├── 01_cross_llm_benchmark/    ← Test if Claude and GPT share the same memories
│   └── benchmark.py
├── 02_memory_risks/           ← Detect data leaks and biases
│   └── memory_risks.py
├── 03_vector_search_benchmarks/ ← Test vector search speed
│   ├── benchmark_beir.py
│   ├── multi_dataset_efficient.py
│   └── test_*.py
├── 04_provider_benchmarks/    ← Test with Ollama, OpenRouter, etc.
│   ├── ollama_one_by_one.py
│   └── openrouter_*.py
├── 05_test_data/              ← Test data (BEIR)
│   ├── beir_data/
│   └── controlled_emb_cache/
├── 06_results/                ← Benchmark results
│   ├── current/
│   └── reports/
├── 07_utilities/              ← Utility scripts
│   └── __init__.py
└── 99_deprecated/             ← Old scripts (outdated, ignore these)
```

## What is this?

MATHIR is a memory system for LLMs. These benchmarks test if it works.

## Why does it matter?

| Test | What does it test? | Why is it useful? |
|------|-------------------|-------------------|
| **Cross-LLM** | Can Claude and GPT understand the same memories? | Proves MATHIR works with any LLM |
| **Risks** | Do personal data leak between domains? | Prevents sensitive data leaks |
| **Speed** | Is search fast enough? | Important for real-time apps |

## How to use?

```bash
# 1. Start MATHIR daemon
python /path/to/MATHIR/bin/mathir_server.py

# 2. Run cross-LLM benchmark
cd /path/to/MATHIR/benchmarks
python 01_cross_llm_benchmark/benchmark.py --providers google nvidia minimax
```

## Results

Results are saved in `06_results/current/`:
- `beir/` → Vector search results
- `memory_tiers/` → Memory tier results
- `stress_tests/` → Stress test results
- `gpu_vec_benchmark.json` → GPU benchmarks

## Deprecated

The `99_deprecated/` folder contains old scripts:
- 384-dim scripts (old MiniLM model)
- FAISS comparisons (we proved numpy is faster)
- Demo scripts (not real benchmarks)
- Old results (not comparable to current)

**Do not use these files** — kept for history only.

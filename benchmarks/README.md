# MATHIR Benchmarks

## Directory Structure

```
benchmarks/
├── README.md
├── 01_cross_llm_benchmark/    ← Cross-LLM memory benchmark (NEW)
│   └── benchmark.py           ← Main cross-provider test
├── 02_memory_risks/           ← Memory risk mitigation (NEW)
│   └── memory_risks.py        ← Leakage, sycophancy, PII detection
├── 03_vector_search_benchmarks/ ← Active vector search benchmarks (1024-dim bge-large)
│   ├── benchmark_beir.py      ← BEIR SciFact benchmark
│   ├── multi_dataset_efficient.py ← Multi-dataset BEIR
│   ├── test_episodic_memory_2hour_stress.py
│   ├── test_episodic_memory_online_learning.py
│   ├── test_immunological_2hour_stress.py
│   ├── test_immunological_anomaly_detection.py
│   ├── test_kl_router_accuracy.py
│   ├── test_kl_router_2hour_stress.py
│   ├── test_working_memory_2hour_stress.py
│   ├── test_working_memory_context.py
│   └── test_integration_2hour_stress.py
├── 04_provider_benchmarks/    ← Provider-specific benchmarks (require external services)
│   ├── ollama_one_by_one.py   ← Requires Ollama running
│   ├── openrouter_multiprovider_benchmark.py ← Requires OpenRouter API
│   ├── openrouter_free_model_probe.py ← Requires OpenRouter API
│   └── sync_report.py         ← Report sync utility
├── 05_test_data/              ← BEIR datasets + embedding caches
│   ├── beir_data/
│   └── controlled_emb_cache/
├── 06_results/                ← All results
│   ├── current/               ← Current results (1024-dim bge-large)
│   │   ├── beir/
│   │   ├── memory_tiers/
│   │   ├── stress_tests/
│   │   ├── gpu_vec_benchmark.json
│   │   └── MATHIR_FINAL_REPORT.html
│   └── reports/               ← HTML reports
├── 07_utilities/              ← Utility scripts
│   └── __init__.py
└── 99_deprecated/             ← Outdated scripts + results (gitignored)
    ├── cross_provider_self_correction_test.py ← 384-dim MiniLM
    ├── universal_recall_demo.py ← 384-dim MiniLM
    ├── mathir_vs_faiss_full.py ← FAISS comparison (numpy beats FAISS)
    ├── benchmark_unified.py   ← FAISS baseline comparison
    ├── latin_demo.py          ← Demo, not benchmark
    ├── latin_e2e.py           ← Demo, not benchmark
    ├── ollama_test.py         ← Local Ollama test
    ├── test_fts.py            ← FTS test
    ├── test_warmup_transition.py ← Warmup test
    ├── test_cold_start_*.py   ← Cold start tests
    ├── test_fifo_vs_lirs.py   ← FIFO vs LIRS eviction
    ├── cross_provider_self_correction_results.json ← 384-dim results
    ├── immunological_results.json ← 384-dim results
    ├── router_accuracy_results.json ← 272-dim results
    ├── universal_recall_results.json ← 384-dim results
    ├── ollama_one_by_one.json ← Ollama results
    ├── openrouter_*.json      ← OpenRouter results
    ├── fifo_vs_lirs_results.json ← Eviction results
    ├── controlled_results.json ← Old BEIR results
    ├── multi_dataset_efficient_results.json ← Old BEIR results
    └── router_stress_results.json ← Old stress results
```

## What Is This?

Cross-LLM memory benchmark that tests MATHIR's unique value: **persistent memory across LLM provider switches**.

Based on 3 research papers:
- **Rosetta Memory** (arxiv 2606.07711) — Cross-LLM memory adaptation
- **PersistBench** (arxiv 2602.01146) — Memory risks (leakage, sycophancy)
- **STATE-Bench** (Microsoft 2026) — Agent memory evaluation

## Why Existing Benchmarks Are Wrong for MATHIR

| Benchmark | Tests | Wrong for MATHIR because |
|-----------|-------|-------------------------|
| BEIR SciFact | Vector search quality | Tests retrieval, not cross-LLM persistence |
| Needle-in-a-Haystack | Context window ability | Tests model, not external memory |
| MRCR v2 | Multi-fact tracking | Single-session, not persistent |
| MTEB | Embedding quality | Doesn't test LLM switching |

## What This Benchmark Tests

| Test | What it measures | Why it matters |
|------|-----------------|----------------|
| **Write Phase** | LLM saves memories to MATHIR | Does the write work across providers? |
| **Cross-Model Recall** | LLM B recalls what LLM A wrote | Core value: provider-agnostic memory |
| **Semantic Drift** | Do two LLMs interpret memories the same way? | Memory fidelity across providers |
| **Risk Mitigation** | Does MATHIR block leakage/sycophancy? | Safety: memories don't leak across domains |
| **Provider Chain** | Write A → Recall B → Re-save C → Recall D | Multi-hop cross-LLM continuity |

## Supported Providers

| Provider | Model | API Key Env Var |
|----------|-------|-----------------|
| Google AI Studio | gemini-2.5-pro | `GOOGLE_AI_STUDIO_KEY` |
| MiniMax | MiniMax-Text-01 | `MINIMAX_API_KEY` |
| NVIDIA NIM | llama-4-maverick | `NVIDIA_API_KEY` |
| OpenCode Zen | zen-mini | `OPENCODE_ZEN_KEY` |

## Setup

```bash
# 1. Start MATHIR daemon
python /path/to/MATHIR/bin/mathir_daemon.py

# 2. Set API keys
export GOOGLE_AI_STUDIO_KEY="your-key"
export MINIMAX_API_KEY="your-key"
export NVIDIA_API_KEY="your-key"
export OPENCODE_ZEN_KEY="your-key"

# 3. Run cross-LLM benchmark
cd /path/to/MATHIR/benchmarks
python 01_cross_llm_benchmark/benchmark.py --providers google nvidia minimax
```

## Risk Mitigation (from PersistBench)

PersistBench found:
- **53% cross-domain leakage** — memories leak between domains
- **>90% sycophancy** — memories bias LLMs incorrectly

MATHIR implements:
1. **Domain isolation** — memories tagged by domain, cross-domain blocked
2. **Sycophancy detection** — biased memories flagged
3. **PII detection** — sensitive data blocked from retrieval
4. **Risk scoring** — each memory gets a risk score

## Output

Results saved to `benchmark_results.json`:
```json
{
  "overall_score": 0.82,
  "passed": 12,
  "failed": 3,
  "leaderboard": {
    "Google AI Studio → NVIDIA NIM": {"score": 0.875, "passed": true},
    "NVIDIA NIM → Google AI Studio": {"score": 0.750, "passed": true}
  }
}
```

## Deprecated

The `99_deprecated/` directory contains outdated scripts and results:
- **384-dim scripts**: Used old MiniLM model, now replaced by 1024-dim bge-large
- **FAISS comparisons**: We proved numpy beats FAISS at MATHIR scale (N<10K)
- **Demo scripts**: Not actual benchmarks, just demonstrations
- **Old results**: From previous model versions, not comparable to current

These are kept for historical reference but should NOT be used for current evaluation.

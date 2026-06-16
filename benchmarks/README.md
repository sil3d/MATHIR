# MATHIR Benchmarks

## Directory Structure

```
benchmarks/
├── README.md                     ← You are here
├── cross-llm/                    ← Cross-LLM memory benchmark
│   └── benchmark.py              ← Main cross-provider test
├── risks/                        ← Memory risk mitigation
│   └── memory_risks.py           ← Leakage, sycophancy, PII detection
├── scripts/                      ← Existing benchmark scripts
│   ├── benchmark_beir.py
│   ├── benchmark_unified.py
│   └── ...
├── data/                         ← BEIR datasets + embedding caches
│   ├── beir_data/
│   └── controlled_emb_cache/
└── results/                      ← All results (consolidated)
    ├── final/                    ← Final results
    │   ├── beir/
    │   ├── memory_tiers/
    │   ├── stress_tests/
    │   └── gpu_vec_benchmark.json
    ├── reports/                  ← HTML reports
    │   └── MATHIR_FINAL_REPORT.html
    └── *.json                    ← Old results
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
python cross-llm/benchmark.py --providers google nvidia minimax
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

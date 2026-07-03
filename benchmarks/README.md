# MATHIR Benchmarks

## Directory Structure

```
benchmarks/
├── README.md
├── 01_cross_llm_benchmark/    <- Cross-LLM memory sharing
├── 02_memory_risks/           <- Data leak and bias detection
├── 03_vector_search_benchmarks/ <- Vector search quality + speed
├── 04_provider_benchmarks/    <- Ollama, OpenRouter, etc.
├── 05_test_data/              <- Test data (BEIR, fluid mechanics)
├── 06_results/current/        <- All benchmark results (see README inside)
├── 07_utilities/              <- Utility scripts
├── 08_industry_validation/    <- LoCoMo, multi-agent, reranking benchmarks
└── 99_deprecated/             <- Old scripts (ignore)
```

## Benchmark Suites (v8.6.0)

| Benchmark | Script | What it proves |
|---|---|---|
| **Multi-agent shared memory** | `08_industry_validation/multi_agent_bench.py` | Free models: 0% → 53% with MATHIR |
| **LoCoMo (Groq)** | `08_industry_validation/run_locomo.py` | 51.2% on conversational QA |
| **LoCoMo (Zen free)** | `08_industry_validation/zen_locomo.py` | 38.8% with free models |
| **Cross-encoder reranking** | `08_industry_validation/rerank_benchmark.py` | +20pp hit@10 on NL queries |
| **e5-small vs e5-large** | `08_industry_validation/e5_comparison.py` | e5-small + rerank > e5-large alone |
| **INT8 quantization** | Built-in (mathir_vec.py auto-migration) | 4x compression, 0% recall loss |
| **Cross-LLM** | `01_cross_llm_benchmark/benchmark.py` | Claude + GPT share memories |
| **Risks** | `02_memory_risks/memory_risks.py` | No PII leaks between domains |

## Key Results (2026-07-03)

```
Benchmark                Score       vs. Baseline    Key Insight
-------------------------------------------------------------------
Multi-agent + MATHIR     53% avg     +53pp vs 0%     Memory makes dumb models smart
Temporal retrieval       78%         +78pp vs 0%     MATHIR excels at time-based recall
Cross-encoder rerank     52.9%       +7.8pp          Cheap model + rerank > expensive model
INT8 quantization        10/10       0% loss         4x compression, zero degradation
e5-small + rerank        52.9%       > e5-large 51%  47x cheaper, better results
LoCoMo (Groq 70B)       51.2%       --              Competitive with published baselines
```

Full detailed report: [06_results/current/README.md](06_results/current/README.md)

## How to Run

```bash
# 1. Start MATHIR daemon
mathir-server &

# 2. Set API keys
cp .env.example .env
# Edit .env with your keys (GROQ_API_KEY, MINIMAX_API_KEY, ZEN_API_KEY)

# 3. Run multi-agent benchmark
cd benchmarks/08_industry_validation
python multi_agent_bench.py

# 4. Run LoCoMo
python run_locomo.py
```

## Models Used

| Model | Provider | Role |
|---|---|---|
| mimo-v2.5-free | OpenCode Zen | Worker agent |
| nemotron-3-ultra-free | OpenCode Zen | Worker agent |
| north-mini-code-free | OpenCode Zen | Worker agent |
| deepseek-v4-flash-free | OpenCode Zen | Worker/orchestrator |
| MiniMax-M3 | MiniMax (`api.minimax.io/v1`) | Judge |
| llama-3.3-70b-versatile | Groq | Answer + judge |

## Deprecated

The `99_deprecated/` folder contains old scripts kept for history only.

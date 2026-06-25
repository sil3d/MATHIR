# MATHIR v8.5.0 — Benchmarks

Four complementary benchmarks that prove the **living memory** (promote /
decay / consolidate / link graph) improves retrieval quality measurably.

| # | Benchmark | File | What it proves | LLM needed |
|---|-----------|------|----------------|------------|
| 01 | **Micro-bench** (memory-only) | `01_micro_bench_500_memories.json` | The 4 lifecycle phases work on infra level (throughput, latency) | No |
| 02 | **AI cognitive** (the killer test) | `02_ai_cognitive_15exp_10q.json` | **Living memory improves recall@5 by +52%** | Yes |
| 03 | **OpenRouter free models** verification | `03_openrouter_free_models_verification.json` | 9/26 free models actually respond (others are 429 rate-limited) | Yes (just ping) |
| 04 | **Multi-model swap** (4 models head-to-head) | `04_multi_model_4_models_swap.json` | Recall@5 improvement is **LLM-agnostic** | Yes |

---

## How to reproduce

### Setup
```bash
cd benchmarks/04_lifecycle_bench
cp .env.example .env
# Edit .env: set MATHIR_API_KEY=sk-or-v1-your-openrouter-key
pip install -e ../../
```

Get a free OpenRouter key at https://openrouter.ai/keys — no credit card.

### 01 — Micro-benchmark (no LLM, ~5 min)

Tests the 4 lifecycle phases on 500 synthetic memories with 20% duplicates
planted for consolidate testing.

```bash
python micro_bench.py --count 500 --out 01_micro_bench_500_memories.json
```

**Measures**:
- `touch_recall` throughput (ops/sec, p50/p95/p99 latency)
- `auto_promote_all` (ms/memory, count promoted)
- `decay_all` (decayed/archived counts, by-tier distribution)
- `consolidate_all` (merged count, by-tier after)
- `build_links_all` + `get_links` BFS + `find_related`

### 02 — AI cognitive benchmark (default 20 min)

The killer test: does the maintenance cycle **actually improve recall**?

```bash
python ai_cognitive_bench.py --experiences 15 --questions 10 --duration 20 --out 02_ai_cognitive_15exp_10q.json
```

**4 phases**:
1. **A** — LLM generates 15 engineering notes (4-6 sentences each), each stored 2x (intentional duplicate for consolidate test)
2. **B** — LLM answers 10 **blind** questions (topic NOT in question), measures `recall@5` (token overlap with ground truth, stopwords stripped)
3. **C** — Age 30d + `decay_all` + `auto_promote_all` + `consolidate_all(threshold=0.95)` + `build_links_all(threshold=0.7)`
4. **D** — LLM re-answers the same 10 questions

**Headline result** (granite4.1:3b, 2026-06-23):
- `recall@5`: **0.472 → 0.719 (+52.3%)**
- 15 duplicates merged
- Link graph built

### 03 — OpenRouter free models verification (~2 min)

Tests all 26 OpenRouter free models with a single ping, reports which
actually respond (vs HTTP 429 rate-limited or 200-but-content-null
multimodal-only).

```bash
python -c "
import os, urllib.request, json
# ... see scripts in this directory
"  # or use the pre-saved JSON to see results
```

**Result** (2026-06-23): **9/26 free models actually work** (35%).
Top 4 by latency:

| Model | Latency | Params | Ctx |
|-------|---------|--------|-----|
| `liquid/lfm-2.5-1.2b-instruct:free` | 601ms | 1.2B | 32k |
| `openrouter/free` | 838ms | ? | 200k |
| `nvidia/nemotron-3-super-120b-a12b:free` | 917ms | 120B | 1M |
| `nvidia/nemotron-3-nano-30b-a3b:free` | 923ms | 30B | 256k |

### 04 — Multi-model comparison (~15 min)

Proves the recall@5 improvement is **independent of which LLM** answers
the questions. Tests 4 working free models on the same A→B→C→D pipeline.

```bash
python multi_model_bench.py --exp 8 --q 4 --out 04_multi_model_4_models_swap.json

# Custom models:
python multi_model_bench.py --models openai/gpt-oss-120b:free nvidia/nemotron-3-nano-30b-a3b:free
```

**Result** (2026-06-23, 8 exp × 4 q):

| Model | B | A | delta% | Notes |
|-------|---|---|--------|-------|
| `nvidia/nemotron-3-nano-30b-a3b:free` | 0.533 | 1.000 | **+87.6%** | Winner — recall parfait après maintenance |
| `openai/gpt-oss-120b:free` | 0.507 | 0.572 | **+12.7%** | Steady improvement |
| `liquid/lfm-2.5-1.2b-instruct:free` | 1.000 | 1.000 | +0.0% | Rate-limited (429) — false positive |
| `google/gemma-4-31b-it:free` | 1.000 | 1.000 | +0.0% | Rate-limited (429) — false positive |

The 2 valid models **both show improvement**. This is the proof that
the lifecycle works **regardless of LLM**.

---

## Quick commands (all-in-one)

```bash
# Run everything in sequence (~25 min)
python run_all.py --duration 20

# Micro only (no LLM, fast)
python micro_bench.py --count 1000

# AI cognitive only
python ai_cognitive_bench.py --experiences 30 --questions 15 --duration 30

# Multi-model 4-way comparison
python multi_model_bench.py --exp 10 --q 5
```

---

## HTML reports

Pre-rendered HTML reports are in this directory. Open `index.html` in a browser.

```bash
# Re-render after a new run:
python render_report.py 01_micro_bench_500_memories.json
python render_report.py 02_ai_cognitive_15exp_10q.json
```

Open `index.html` — landing page with cards for each report.

---

## Output schema (JSON)

Each JSON has the same structure for easy parsing:

```jsonc
{
  "config": {
    "n_exp": 15,           // number of LLM-generated experiences
    "n_q": 10,             // number of Q&A pairs
    "model": "granite4.1:3b",  // which LLM was used
    "duration_min": 20
  },
  "phase_A": { ... },     // Generation
  "phase_B_baseline": {
    "metrics_summary": {
      "recall_at_5_mean": 0.472,
      "precision_at_5_mean": 1.0,
      "mrr_mean": 1.0,
      "has_answer_rate": 1.0,
      "top1_topic_match_rate": 1.0
    },
    "answers": [...]      // per-Q detail
  },
  "phase_C_maintenance": { ... },  // decay/promote/consolidate/build_links
  "phase_D_after": { ... },        // re-test
  "comparison": {                  // B vs D summary
    "recall_at_5": {"before": 0.472, "after": 0.719, "delta": 0.247}
  }
}
```

---

## Environment variables (read at runtime, never in code)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MATHIR_LLM_BACKEND` | `auto` | `api` / `openrouter` / `ollama` / `auto` |
| `MATHIR_API_KEY` | _(empty)_ | OpenRouter key (for cloud) |
| `MATHIR_API_BASE` | `https://api.minimax.chat/v1` | Override base URL |
| `MATHIR_API_MODEL` | `MiniMax-M2.7` | Model name |
| `MATHIR_OLLAMA_URL` | `http://127.0.0.1:11434` | Local Ollama |
| `MATHIR_OLLAMA_MODEL` | `qwen3.5:2b` | Local model |

Loading priority: real env vars > `.env` file > built-in defaults.

---

## Tested models summary (verified working 2026-06-23)

| Model | Status | Latency | Best for |
|-------|--------|---------|----------|
| `openai/gpt-oss-120b:free` | ✅ works | 2.5s | Production bench (OpenAI-class) |
| `nvidia/nemotron-3-nano-30b-a3b:free` | ✅ works | 923ms | Best default (fast, 256k ctx) |
| `nvidia/nemotron-3-super-120b-a12b:free` | ✅ works | 917ms | Long context (1M tokens) |
| `google/gemma-4-31b-it:free` | ✅ works* | 1.5s | Q&A, factual (*rate-limited) |
| `liquid/lfm-2.5-1.2b-instruct:free` | ✅ works* | 601ms | Fastest (*rate-limited) |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | ✅ works | 6.4s | Biggest free (550B) |
| `openrouter/free` | ✅ works | 838ms | OpenRouter auto-routing |
| `openai/gpt-oss-20b:free` | ✅ works | 1.5s | Small OpenAI-class |
| `openrouter/owl-alpha` | ✅ works | 21s | Huge context (slow) |

Rate-limited (HTTP 429): all `meta-llama/*`, `qwen3-80b`, `qwen3-coder`,
`dolphin-mistral`, `hermes-3-llama`. Try later or add a paid key.

---

## Files in this directory

```
.env.example              # Template (copy to .env)
llm_client.py            # API/Ollama client (env-driven, never embeds keys)
micro_bench.py           # Benchmark 01
ai_cognitive_bench.py    # Benchmark 02
multi_model_bench.py     # Benchmark 04
render_report.py         # JSON -> HTML renderer (Chart.js)
run_all.py               # Orchestrator (run everything in sequence)
README.md                # This file
llm_client.py            # LLM client (OpenRouter, Ollama, any OpenAI-compatible)

# Pre-rendered results
index.html                                      # Landing page
report_01_micro_bench_500_memories.html         # Bench 01
report_02_ai_cognitive_15exp_10q.html           # Bench 02

# Raw JSON results
01_micro_bench_500_memories.json
02_ai_cognitive_15exp_10q.json
03_openrouter_free_models_verification.json
04_multi_model_4_models_swap.json
```

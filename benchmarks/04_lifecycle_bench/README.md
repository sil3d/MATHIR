# MATHIR v8.4.0 Lifecycle Benchmarks

Two complementary benchmarks that exercise the new **living memory** phases
(promote / decay / consolidate / link graph) shipped in v8.4.0.

## What's in here

| File | Purpose | Duration | LLM required |
|------|---------|----------|--------------|
| `micro_bench.py` | Memory-only throughput + correctness | ~5 min | No |
| `ai_cognitive_bench.py` | End-to-end cognitive quality (before/after maintenance) | 20 min default | Yes (API or Ollama) |
| `llm_client.py` | Unified LLM client (API + Ollama, env-driven) | — | — |
| `run_all.py` | Orchestrator that runs both | 25 min total | For `ai_cognitive` only |

## Quick start

```bash
# Micro-bench only (no LLM, fast)
python micro_bench.py --count 1000

# AI cognitive bench (20 min, requires LLM)
python ai_cognitive_bench.py --duration 20

# Full suite
python run_all.py --duration 20
```

## Environment

The LLM client reads these at call time — **never** hardcoded in the repo.
Loading priority (first non-empty wins):
  1. Real environment variables (`$env:` in PowerShell, `export` in bash)
  2. `.env` file in `benchmarks/04_lifecycle_bench/` (auto-loaded if present)
  3. Built-in defaults

| Variable | Default | Purpose |
|----------|---------|---------|
| `MATHIR_LLM_BACKEND` | `auto` | `api` / `openrouter` / `ollama` / `auto` |
| `MATHIR_API_KEY` | _(empty)_ | Set this to use API/OpenRouter |
| `MATHIR_API_BASE` | `https://api.minimax.chat/v1` | API base URL (auto-overridden for OpenRouter) |
| `MATHIR_API_MODEL` | `MiniMax-M2.7` | Model name |
| `MATHIR_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama base |
| `MATHIR_OLLAMA_MODEL` | `qwen3.5:2b` | Ollama model |
| `MATHIR_OPENROUTER_REFERER` | _(unset)_ | Optional. Site URL for OpenRouter ranking |
| `MATHIR_OPENROUTER_TITLE` | _(unset)_ | Optional. Site title for OpenRouter ranking |

## Setup with `.env` (recommended for repeated runs)

```bash
# 1. Copy the template
cp benchmarks/04_lifecycle_bench/.env.example benchmarks/04_lifecycle_bench/.env

# 2. Edit .env — set your real key
#    (NEVER commit .env — it's in .gitignore)
notepad benchmarks/04_lifecycle_bench/.env

# 3. Run — the client auto-loads .env
python benchmarks/04_lifecycle_bench/run_all.py --duration 20
```

`.env` is gitignored at the repo root. Your keys stay local.

## LLM providers

### Option 1: Local Ollama (no API key, slower, weaker)
```powershell
# Nothing to set — falls back automatically
python run_all.py --duration 20
```

### Option 2: OpenRouter (free tier, recommended) ⭐
Get a free API key at https://openrouter.ai/keys, then either:

**A. .env (recommended for repeated runs):**
```bash
# In benchmarks/04_lifecycle_bench/.env
MATHIR_LLM_BACKEND=openrouter
MATHIR_API_KEY=sk-or-v1-your-key
MATHIR_API_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

**B. Or inline env vars:**
```powershell
$env:MATHIR_API_KEY = "sk-or-v1-your-key"
$env:MATHIR_API_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
python run_all.py --duration 20
```

The client auto-detects `sk-or-` keys and routes to OpenRouter.

**Other recommended free models** (paste any into `.env`):
```bash
MATHIR_API_MODEL=qwen/qwen3-next-80b-a3b-instruct:free    # 80B, 262k ctx
MATHIR_API_MODEL=openai/gpt-oss-120b:free                 # 120B GPT-class
MATHIR_API_MODEL=google/gemma-4-31b-it:free               # 31B, 262k ctx
MATHIR_API_MODEL=meta-llama/llama-3.2-3b-instruct:free    # 3B, fast smoke tests
MATHIR_API_MODEL=nvidia/nemotron-3-nano-30b-a3b:free      # 30B
```

List all curated free models: `python llm_client.py list-free`

### Option 3: Any OpenAI-compatible API
```bash
# In .env
MATHIR_LLM_BACKEND=api
MATHIR_API_KEY=sk-...
MATHIR_API_BASE=https://api.your-provider.com/v1
MATHIR_API_MODEL=your-model-name
```

### CLI flags (override `.env`)
```bash
python ai_cognitive_bench.py --provider openrouter \
  --model meta-llama/llama-3.3-70b-instruct:free --duration 20
```

## What each benchmark measures

### `micro_bench.py` — memory infrastructure

For N memories (default 1000):

| Phase | Metric |
|-------|--------|
| Seed | memories/s throughput |
| `touch_recall` | ops/s, p50/p95/p99 latency |
| `auto_promote_all` | ms/memory, count promoted |
| `decay_all` | decayed / archived counts, by-tier distribution |
| `consolidate_all` | merged count, by-tier after merge |
| `build_links_all` | links created, links/s |
| `get_links` BFS | avg links per node, p95 latency |
| `find_related` | vector+graph merged results per query |

### `ai_cognitive_bench.py` — the killer benchmark

This is the test that matters. It answers:
> **Does the maintenance cycle (decay + promote + consolidate + build_links)
> actually improve recall quality, or just shuffle data?**

Four phases:

1. **A — Generation**: LLM creates N "engineering notes" (e.g. "JWT token
   expiration edge case"), each stored in memory. Also stores an intentional
   duplicate per note to give `consolidate` something to merge.

2. **B — Baseline**: LLM answers K questions about the topics. Measure
   `recall@5`, `precision@5`, `MRR`, `has_answer` (token overlap with ground
   truth ≥ 20%).

3. **C — Aging + maintenance**:
   - Force `last_recalled_at = now - 30d` (simulate 30 days of inactivity)
   - Run `decay_all(threshold_days=30)` → archive stale memories
   - Run `auto_promote_all()` → promote mature ones
   - Run `consolidate_all(threshold=0.95, dry_run=False)` → merge duplicates
   - Run `build_links_all(threshold=0.7)` → build link graph

4. **D — Re-test**: LLM re-answers the **same** K questions. Same metrics.

**The headline result**: how much did recall quality change after the
maintenance cycle ran on a stale + duplicated + fragmented memory?

```
=== COMPARISON ===
  recall@5:    0.42 -> 0.51  (Δ +0.09)
  precision@5: 0.35 -> 0.62  (Δ +0.27)
  has_answer:  78%   -> 92%   (Δ +14%)
```

This is the proof that **living memory is measurably better than passive
storage** — and it's what separates MATHIR from every other memory layer
for LLMs.

## Output format

Each script writes a JSON file. Top-level keys:

```jsonc
{
  "config": { "count": 1000, ... },
  "seed_wall_s": 1.2,
  "db_size_mb": 4.5,
  "touch_recall": { "ops_per_sec": 8000, "latency_ms": {...} },
  "promote": { "scanned": 1000, "promoted": 0 },
  "decay": { "decayed": 0, "archived": 0, "by_tier": {...} },
  "consolidate": { "merged": 0, "candidates": 200, "by_tier": {...} },
  "link_graph": { "build": {...}, "bfs_get_links": {...}, "find_related": {...} }
}
```

For `ai_cognitive_bench.py`:

```jsonc
{
  "config": {...},
  "phase_A": { "experiences_generated": 50, "memories_stored": 100, "wall_s": 120 },
  "phase_B_baseline": { "questions_answered": 20, "metrics_summary": {...}, "answers": [...] },
  "phase_C_maintenance": { "decay": {...}, "promote": {...}, "consolidate": {...}, "build_links": {...} },
  "phase_D_after": { "questions_answered": 20, "metrics_summary": {...}, "answers": [...] },
  "comparison": { "recall_at_5": {"before": 0.4, "after": 0.5, "delta": 0.1}, ... }
}
```

## Resource notes

- `micro_bench.py` uses synthetic random vectors — no embedder loaded, runs in seconds.
- `ai_cognitive_bench.py` loads the project's `MiniLM-L12` embedder (~250MB RAM,
  GPU optional). The LLM call is the bottleneck.
- **Do not run both `llama-server` and `ollama` simultaneously** on the same
  GPU — pick one. `LLMClient` chooses API > Ollama > (fail) automatically.

## CI usage

```yaml
# Fast smoke test (no LLM, no GPU)
- run: python benchmarks/04_lifecycle_bench/micro_bench.py --count 100

# Full bench on nightly schedule
- run: python benchmarks/04_lifecycle_bench/run_all.py --duration 20
  env:
    MATHIR_API_KEY: ${{ secrets.MINIMAX_API_KEY }}
```

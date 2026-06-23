# Changelog

All notable changes to MATHIR are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

> **v1–v7.6 are historical** — kept per project policy ("laisse les deprecated pour qu'ils comprennent que je reviens de loin"). Source code for these versions lives under [`_deprecated/`](_deprecated/). Each version below is summarized in 1–3 lines; see the git history or the `_deprecated/` folder for the full original entries.

---

## [8.4.1] — 2026-06-23 — DYNAMIC INJECTION + SYNC

**Dev-loop release.** v8.4.1 ships two new developer tools (`mathir_inject.py`, `mathir_sync.py`), 5 target-specific injection templates, and fixes the install/import path story so the package is reproducible from a clean clone.

### Dynamic Injection (NEW)

- **`mathir_inject.py`** — Multi-target dynamic injection of the MATHIR block into any `.md` file in `agents/`, `commands/`, `skills/`, `skills-global/`, or `docs/`. Reads the right template from `<target_dir>/_MATHIR_INJECT.md` (falls back to root or `agents` template). Idempotent via content-equivalent detection (`_normalize()`). Flags: `--target {agents,commands,skills,skills-global,docs,all}`, `--apply`, `--check`, `--file PATH`, `--list`, `--explain`.
- **`mathir_sync.py`** — Safe source-to-config sync from `<repo_root>/mathir_mcp/` into `~/.config/opencode/`. **SAFE BY DEFAULT** — only copies NEW files, never overwrites existing without `--update-existing`. Syncs `mathir_lib/*.py`, `brain/*.py`, `config/*.json`, `docs/*.md`, `GLOBAL_INSTRUCTIONS.md`, install/scripts, and all 5 `_MATHIR_INJECT.md` templates. Flags: `--dry-run`, `--only`, `--update-existing`, `--no-inject`, `--force`, `--explain`. Auto-runs `mathir_inject.py --apply` after a successful `--apply`.
- **5 target templates** in `mathir_mcp/opencode/<target>/_MATHIR_INJECT.md` (one per target: `agents`=full, `commands`=short, `skills`=minimal, `skills-global`=minimal, `docs`=reference — 4 unique templates, `skills-global` shares the `skills` template). Editing the template propagates to all files of that target in one command.
- **`--explain` mode** on both scripts — self-documenting, no external README needed.
- Slash commands: `/mathir_inject [check|apply] [target] [name]` and `/mathir_sync [check|apply] [filter]`.

### Bug Fixes

| Severity | Issue | Fix |
|---|---|---|
| CRITICAL | Stale `.pth` file in `mathir_lib/mathir_mcp.egg-info/` broke `pip install -e .` reproducibility | Cleared egg-info, content re-pinned |
| CRITICAL | Namespace shadow: importing `mathir_mcp.X` resolved to the package, not the module | Added `mathir_mcp/__init__.py` as package marker (exports `__version__` + `__all__`; submodules are reached via `mathir_mcp.mathir_lib.*`, `mathir_mcp.brain.*`, `mathir_mcp.mathir_dropin.*`) |
| HIGH | `pyproject.toml` entry points pointed at flat paths | Rewrote as `mathir_mcp.mathir_lib.<tool>:main` (nested) |
| HIGH | 9 self-referencing imports in `__main__.py` (`from mathir_mcp import X` → unbound) | Rewrote all to `from mathir_mcp.mathir_lib.X import Y` |
| HIGH | MCP server config pointed at wrong path | Now: `bin/mathir_mcp_server.py` with `PYTHONPATH=bin` |
| MEDIUM | Hardcoded repo path in `mathir_mcp_server.py:1190` (install-help message) | Replaced with `<repo_root>` placeholder + auto-detect |
| MEDIUM | `get_embedder()` had no way to force ONNX | Added `MATHIR_USE_ONNX=1` env var support |
| LOW | Stale cross-refs in `docs/AGENT.md` (11) and `docs/DASHBOARD_GUIDE.md` (12) | Updated to current paths |

### Tests

- **173/173 tests pass** (unchanged from v8.4.0)
- Swarm verified: `@coder`, `@refactor`, `@security`, `@debugger`, `@make`, `@check` — all green
- Database state: 2 active DBs, 100% embeddings, lifecycle active
  - Mycerise project: 368 memories
  - MATHIR root project: 137 memories

### Raspberry Pi / Jetson

- `raspberry_jetson/` bumped to v8.4.1 (all files synced)
- Install flow: `install.ps1` now calls `mathir_inject.py --check` on completion

### Files Changed

- `mathir_mcp/__init__.py` (new) — package marker, exports `__version__`; submodules reached via `mathir_mcp.mathir_lib`, `mathir_mcp.brain`, `mathir_mcp.mathir_dropin`
- `mathir_mcp/opencode/{agents,commands,skills,skills-global,docs}/_MATHIR_INJECT.md` (new, 5 templates)
- `mathir_mcp/__main__.py` — 9 import paths fixed
- `mathir_mcp/pyproject.toml` — entry points nested
- `mathir_mcp/mathir_lib/mathir_mcp_server.py` — line 1190 placeholder (install-help message), `get_embedder()` ONNX env
- `mathir_mcp/docs/AGENT.md` — 11 references updated
- `mathir_mcp/docs/DASHBOARD_GUIDE.md` — 12 references updated
- `raspberry_jetson/` — all files bumped to v8.4.1

---

## [8.4.0] — 2026-06-23 — 🧠 LIVING MEMORY

**The breakthrough release.** MATHIR is no longer a write-only memory disk. It now manages its own memory lifecycle through 4 cognitive phases inspired by hippocampal-cortical consolidation (CLS theory, McClelland, McNaughton & O'Reilly 1995) and Ebbinghaus forgetting curves.

### Memory Lifecycle — 4 Phases

#### Phase 1: Promote (tier transitions)
- `touch_recall(memory_id)` — increments `recall_count`, stamps `last_recalled_at`, boosts stability (Ebbinghaus)
- `promote(memory_id, force=False)` — moves memory up the tier ladder via rules:
  - `working_memory` → `episodic`: `recall_count >= 3` AND `age >= 1d`
  - `episodic` → `semantic`: `recall_count >= 10` AND `age >= 7d`
  - `semantic` → `procedural`: `priority >= 8` AND label starts with `how-to:` or `recipe:`
- `auto_promote_all()` — scans all memories, promotes eligible ones

#### Phase 2: Decay (Ebbinghaus forgetting)
- `boost_on_recall(memory_id)` — `stability += 0.1`, capped at 1.0
- `get_decay_candidates(threshold_days=30)` — ordered list of stale memories
- `decay_all(threshold_days, archive_floor=0.05)` — 5%/30d linear decay, archives when `stability < 0.05`
- Archived memories keep their `memory_id` (soft delete, audit trail)

#### Phase 3: Consolidate (semantic merge)
- `find_duplicates(threshold=0.95)` — pairs with cosine > threshold
- `consolidate_pair(id_strong, id_weak)` — transactional merge with audit trail in `metadata.merged_from[]`
- `consolidate_all(threshold, dry_run=True)` — orchestrator with dry-run support

#### Phase 4: Link Graph (spreading activation)
- New table `memory_links(source_id, target_id, weight, created_at)` + indexes
- `add_link(source, target, weight=1.0)` — bidirectional graph edges
- `get_links(memory_id, depth=1, decay=0.5)` — BFS with per-hop decay (Collins & Loftus 1975)
- `build_links_all(threshold=0.7)` — creates symmetric links for similar pairs
- `find_related(memory_id, max_hops=2)` — vector + graph combined, tags source as `vector`/`link`/`both`

### MCP / Daemon Integration
- **7 new MCP tools** registered in `mathir_mcp_server.py`:
  - `memory_promote`, `memory_auto_promote`, `memory_decay`, `memory_consolidate`
  - `memory_link`, `memory_get_links`, `memory_build_links`
- **7 new daemon RPC methods** in `mathir_daemon.py`
- `memory_recall` now auto-calls `touch_recall()` on every result — stability grows on use
- Handlers wired in `_METHOD_HANDLERS` dispatch table

### Schema
- New column `memories.last_recalled_at REAL DEFAULT 0` (idempotent `ALTER TABLE` migration)
- New table `memory_links(source_id, target_id, weight, created_at)` with `idx_links_source` + `idx_links_target`
- Both schema branches (new `content`-column / legacy `modality`-BLOB) fully supported

### Tests
- **26 new pytest tests** in `mathir_mcp/dev/test_lifecycle.py`
  - `TestPromote` (9 tests): force transitions, rule checks, auto-promote, touch_recall
  - `TestDecay` (6 tests): boost, decay, archive, new-schema skip
  - `TestConsolidate` (4 tests): find_duplicates, merge, dry-run, real-run
  - `TestLinkGraph` (7 tests): add_link, get_links BFS with decay, build_links_all, find_related
- **173/173 tests pass** (was 147)

### Assets
- `docs/assets/logo.png` (1024×1024) — neural core with 8 pathways
- `docs/assets/architecture.png` (1600×900) — 5-layer system topology
- `docs/assets/PROMPTS.md` — ready-to-paste prompts for AI image generators

### Bug Fixes
- `mathir_mcp_server.py`: Python boolean syntax (`True`/`False`, not JSON `true`/`false`)
- `mathir_mcp_server.py`: extra closing brace in TOOLS list (line 484)
- `mathir_vec.py`: `stats()` was querying non-existent columns — now uses `json_extract()` on metadata

### Live Verification (2026-06-23)
```text
stats: 29 memories, by_tier={episodic:14, semantic:9, working:6}
promote: episodic → semantic (force=True)
recall: 3 results, touched=3
build_links: 246 links created from 29 memories
consolidate: 3 candidates at threshold 0.9 (dry_run)
```

---

## [8.3.0] — 2026-06-19

### HybridSearch — Direct SQLite Backend (FIX)

- **Bug fix**: `memory_hybrid_search` returned 0 results because it used a separate empty `_hybrid.db` file
- **Root cause**: VecMemory was created in the daemon's main thread, then used in handler threads → `ProgrammingError: SQLite objects created in a thread can only be used in that same thread`
- **Fix**: Hybrid handler now creates its own SQLite connection with `check_same_thread=False`, reads directly from the main vector DB
- **Schema auto-detection**: Handler detects old schema (`modality_text`) vs new schema (`content`) automatically
- **Performance**: ~60ms per hybrid search (was timeout/unusable before)

### Daemon Thread Safety (FIX)

- **Bug fix**: 3rd request always timed out (daemon hung after 2 successful requests)
- **Root cause**: `from mathir_mcp_server import get_embedder_dim` inside handler methods created a local variable that shadowed the global function → `UnboundLocalError` on `ping` → crash in `handle_client` while-True loop
- **Fix**: All `get_project_db_path`, `get_project_name`, `get_embedder` moved to top-level imports (line 50)
- **VecMemory**: Added `check_same_thread=False` to `sqlite3.connect()` for cross-thread access
- **Stress test**: 50/50 requests (20 saves + 20 pings + 10 recalls), 0 errors

### Embedding Model

- **paraphrase-multilingual-MiniLM-L12-v2** (384d) is now the confirmed production model
- All 4 MATHIR databases migrated from 1024d (bge-large-en-v1.5) to 384d
- 239MB VRAM fp16, 0.929 cosine FR↔EN, 50+ languages

### Changes

- `mathir_daemon.py` — Hybrid handler uses direct SQLite, global imports, `_get_vec_mem` cache
- `mathir_vec.py` — `check_same_thread=False` for cross-thread SQLite access
- `mathir_search.py` — HybridSearch BM25 + RRF fusion (unchanged, was already correct)

---

## [8.2.0] — 2026-06-16

### Daemon Push (NEW)

- **Proactive memory delivery** — daemon pushes relevant memories without explicit recall requests
- **Push modes**:
  - `--auto`: Daemon analyzes context, returns text ready for system prompt injection
  - `--json`: Returns structured `{memories: [...]}` with metadata
  - (default): Returns human-readable `[block/agent] label: content` format
- **Cache system** — LRU cache with TTL (300s) prevents redundant embedding computations
- **Use cases**:
  - Auto-inject relevant context before each LLM call
  - Proactive memory suggestions during conversations
  - Background context enrichment for long sessions

### Push Architecture

```
Client → push "context" → Daemon analyzes context → Extracts queries → Searches memory → Returns ranked memories
```

### Commands Added

```bash
# Push (proactive memory delivery)
python ~/.config/opencode/mcp/mathir_lib/mathir_client.py push "contexte ici" --auto
python ~/.config/opencode/mcp/mathir_lib/mathir_client.py push "contexte ici" --json
python ~/.config/opencode/mcp/mathir_lib/mathir_client.py push "contexte ici"

# Cache stats (via daemon)
python ~/.config/opencode/mcp/mathir_lib/mathir_client.py push "" --json 2>&1 | head -1
```

### Documentation Updates

- `README.md` — New Daemon Push section with architecture diagram
- `GLOBAL_INSTRUCTIONS.md` — Added push commands to MEMORY PROTOCOL and AGENT MEMORY BLOCKS sections

---

## [8.0.0] — 2026-06-15

### HybridSearch Auto-Scaling Backend (NEW)

- **`mathir_search.py`** — New unified vector search class replacing `VectorSearch`
  - Auto-scales: numpy brute-force (N < 5000) → USearch HNSW (N ≥ 5000)
  - SQLite WAL metadata store always-on (thread-safe, crash-safe)
  - USearch HNSW tuned for 1024d: `connectivity=32`, `expansion_add=256`, `expansion_search=128`
  - Auto-persists USearch index to disk (`mathir_indexes/`), rebuilds on load
  - Thread-safe with `RLock` on all mutating operations
  - Agent-filtered search with over-fetch + post-filter
  - `store()`, `store_batch()`, `search()`, `delete()`, `count()`, `stats()`, `save()`, `close()`

### USearch Integration

- **`usearch`** — HNSW library with memory-mapped indexes for fast ANN search
  - Cosine metric, L2-normalized vectors
  - Memory-mapped persistence — index survives process restarts
  - Auto-builds from SQLite metadata on first load (crash-recovery)
  - `save()` persists to `mathir_indexes/mathir_{dim}d.usearch`

### sqlite-vec WAL Optimization

- **`mathir_vec_optimized.py`** — High-performance sqlite-vec backend
  - WAL mode + single connection (no pool overhead for sequential access)
  - LRU dict cache (512 entries, 120s TTL) for repeated queries
  - PRAGMA tuning: `cache_size=-8000`, `temp_store=MEMORY`, `mmap_size=256MB`
  - `store()`, `store_batch()`, `search()`, `delete()`, `get_all()`, `stats()`, `close()`

### BEIR Benchmark Results

- **SciFact** (5183 docs, 1109 queries):

| Backend | Latency | Recall@10 | nDCG@10 |
|---|:---:|:---:|:---:|
| **Numpy** | **0.83ms** | **0.8592** | — |
| USearch | 5.33ms | 0.8526 | — |
| sqlite-vec | 37.95ms | 0.8592 | — |

- Numpy backend is fastest at MATHIR scale (< 5000 memories)
- USearch outperforms sqlite-vec for large-scale workloads
- Fixed stale USearch index persistence bug
- Fixed USearch HNSW recall with tuned params

### Security Fixes (8)

| Severity | Issue | Fix |
|---|---|---|
| CRITICAL | RCE via `torch.load` | `weights_only=True` enforced |
| HIGH | SQL injection in vec0 DDL | Validated dim as `int > 0` |
| HIGH | `assert`-based validation | Replaced with `ValueError` |
| MEDIUM | Path traversal on `db_path`/`index_dir` | `Path.resolve()` canonicalization |
| MEDIUM | Path traversal on torch save/load | `Path.resolve()` + whitelist |
| MEDIUM | Race condition in USearch search | Lock moved around index access |
| MEDIUM | No thread safety in VecMemory | Added `threading.Lock()` |
| LOW | Partial locking in `_MetadataStore` | Full RLock on all mutations |

### Code Reduction (-46%)

| File | Before | After | Reduction |
|---|:---:|:---:|:---:|
| `mathir_search.py` | 559 | 321 | -43% |
| `mathir_gpu_vec.py` | 511 | 275 | -46% |
| `mathir_vec_optimized.py` | 476 | 240 | -50% |
| **Total** | **1546** | **836** | **-46%** |

### Documentation Updates

- `README.md` — HybridSearch section, updated performance numbers, architecture diagram, vector search benchmarks
- `03_MASTER_QA_GUIDE.md` — Architecture diagram, BEIR benchmarks, deployment options, quick reference
- `04_DEV_INTEGRATION_GUIDE.md` — Full HybridSearch chapter with architecture, benchmarks, quick start, advanced usage
- `stress_test/static/changelog.html` — Interactive changelog with before/after diffs

### Bug Fixes

- `benchmark_unified.py` — Stale `VectorSearch` import → `HybridSearch`
- `benchmark_beir.py` — Dead `recall_at_k` function fixed
- `mathir_search.py` — Docstring `VectorSearch` → `HybridSearch`
- Stale `VectorSearch` references: 0 remaining

---

## [7.8.0] — 2026-06-15

### GPU Embedding Engine (NEW)

- **Default model changed**: `BAAI/bge-large-en-v1.5` (1024d, CUDA) replaces `all-MiniLM-L6-v2` (384d)
- **Persistent daemon**: `mathir_daemon.py` keeps model loaded in RAM, serves via TCP port 7338
- **Fast client**: `mathir_client.py` connects to daemon — no Python startup per call
- **onnxruntime-gpu 1.26.0**: CUDAExecutionProvider + TensorrtExecutionProvider available
- **Model saved locally**: `~/.config/opencode/models/bge-large-v1.5/`

### Benchmarks (RTX 4060 Laptop GPU)

| Model | Dims | Save | Recall | Device |
|---|:---:|:---:|:---:|---|
| **bge-large-en-v1.5** | **1024** | **43ms** | **25ms** | CUDA |
| MiniLM-L6-v2 | 384 | 22ms | 53ms | CUDA |
| nomic-embed-text-v1.5 | 768 | ~21ms | ~20ms | CUDA |
| Octen INT8 (ONNX) | 1024 | ~5000ms | ~2700ms | CPU |
| Octen INT8 (ONNX+CUDA) | 1024 | ~776ms | — | Partial GPU |

### MCP Documentation

- New `mcp/` folder with comprehensive integration guides
- `DIMENSIONS.md` — Embedding dimension explained
- `MODEL_COMPARISON.md` — All models compared
- `GPU_SETUP.md` — GPU acceleration setup
- `DAEMON.md` — Daemon architecture
- `INTEGRATION.md` — Platform integration guides

### Bug Fixes

- Removed `backend="onnx"` from SentenceTransformer (caused silent CPU fallback, 200x slowdown)
- Fixed `EMBEDDING_DIM` default from 1024 to match actual model dimensions
- VecMemory auto-recreates vec0 table on dimension mismatch

---

## [7.7.3] — 2026-06-15

### ONNX Embedding Provider (NEW)

- **`mathir_lib/providers/onnx.py`** — New `ONNXProvider` using ONNX Runtime
  - Supports INT8 quantized models (5.2 MB vs 80 MB for MiniLM)
  - L2-normalized output (cosine-ready, unlike HuggingFace)
  - Configurable execution provider: `CPUExecutionProvider` or `DmlExecutionProvider` (GPU via DirectML)
  - Mean pooling + L2 normalize pipeline
  - Auto-detects embedding dim from `config.json` (default 1024)
- **`mathir_lib/providers/onnx_embedder.py`** — Standalone embedder wrapper
- **`mathir_lib/providers/__init__.py`** — Registered `onnx` in factory
- **`mathir_lib/config.py`** — Added `onnx` config section (`model_dir`, `provider`)
- **`mcp_server.py`** — Easy-to-plug MCP server with 4 tools (`memory_save`, `memory_recall`, `memory_stats`, `provider_info`)

### Benchmarks

| Provider | Model | Dim | Size | Batch (5q+8d) | Single | Normalized |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **ONNX (Octen INT8)** | `Octen-Embedding-0.6B-INT8` | **1024** | **5.2 MB** | 203 ms | 18.8 ms | ✅ |
| HuggingFace | `all-MiniLM-L6-v2` | 384 | 80 MB | 27 ms | 5.2 ms | ❌ |
| HuggingFace | `Qwen/Qwen2.5-7B-Instruct` | 3584 | 14 GB | ~30 ms (GPU) | ~10 ms (GPU) | ❌ |

**Quality:** Octen INT8 produces similarity scores in `[0.42, 0.98]` (L2-normalized), while MiniLM raw outputs span `[-2.53, 34.34]` (unnormalized, requires post-processing).

### Documentation

- Updated `README.md` with provider comparison table and ONNX section
- `benchmark_onnx.py` — Reproducible benchmark script
- `examples/onnx_usage.py` — Usage examples
- `mcp_server.py` — Drop-in MCP server for Claude/OpenCode

---

## [7.7.2] — 2026-06-11

### Thread Safety

- **Full RLock audit** — All 8 memory modules in `mathir_lib/memory/` now use `threading.RLock` on mutating methods (`store`, `forget`, `reset`)
  - `working.py`, `episodic.py`, `semantic.py`, `immunological.py` — already had RLock
  - `ensemble_episodic.py`, `hybrid_episodic.py`, `raw_episodic.py` — **added** RLock
- **mathir_dropin** — `store.py` and `memory.py` already thread-safe (confirmed during audit)
- **Thread-safe under concurrent stress** — stress test runs 4-tier memory + BM25 + cross-encoder concurrently without data races

### Stress Test Fixes

- **CPU metric fixed** — Was always 0% due to `psutil.cpu_percent(interval=None)` delta-based measurement from sleeping thread. Replaced with manual `cpu_times()` delta (`process.cpu_times().user + .system` / `time.monotonic()`). Process-wide, thread-independent.
- **GPU metric fixed** — Was reporting global VRAM (all processes, 241–328 MB). Now uses `torch.cuda.memory_allocated()` for MATHIR-only tensors (36–53 MB).
- **Clean slate on restart** — `start()` now deletes `stress_memory.db` + WAL/SHM files before reinit. Old data no longer persists between runs.
- **Start after Stop fixed** — `start()` now recreates `ThreadPoolExecutor` after `stop()` kills it. Root cause of "Start doesn't work after Stop".
- **Config deep merge** — `start()` preserves `health_thresholds` when merging frontend config.

### Dashboard

- **System Health Bar** — 3-color bar (green/blue/red) scoring CPU, GPU, Recall, Errors, DB Write. Thresholds server-driven via WebSocket `health_config` event.
- **REST /api/metrics fixed** — Now returns all fields including `cpu_percent`, `peak_ram_mb`, `throughput`, `db_write_latency`, `uptime`.
- **Frontend error handling** — `startTest()` handles `already_running` response properly.
- **Changelog page** — Full architecture documentation at `/changelog` with before/after code diffs, benchmarks, file reference. Accessible via "Changelog" button in dashboard header.

---

## [7.7.1] — 2026-06-06

**Major UI + memory overhaul.** Full changelog in `_deprecated/CHANGELOG_v7.7.1.md`.

- **Memory**: `SimpleMemory` (FTS5-only, zero deps) + `search_context()` (recall+last, dedup) + 31/31 audit checks.
- **UI**: SVG icons (no emoji), localStorage chat history, OpenCV backend camera, 6 views, dark theme, `/playground.html` multi-session chat with model switching / image drag / hold-to-talk.
- **API fixes**: 6 broken routes repaired; new `/api/system/*`, `/api/models/*`, `/api/memory/delete`, `/api/accuracy/test`.
- **System prompt**: ~1100 → ~126 tokens; rigid template removed.
- **Other**: `localhost`→`127.0.0.1`, auto-load LFM2.5-VL, full-conversation storage with skip-trivial filter, file markers `[IMAGE/AUDIO ATTACHED]`, hardcoded values eliminated (lang, routes, paths), workspace cleanup, LSTM refs removed, LaTeX paper + AGENT.md + benchmark methodology added.

---</newString>

## [7.7.0] — 2026-06-06

**Vision & audio testing UI** — Flask backend + 17 API routes, Chat/Camera/Models/Memory/Settings views; CLI tools (`model_manager.py`, `setup_binaries.py`, `download_models.py`); LFM2.5-VL-1.6B (1.2 GB) + LFM2.5-Audio-1.5B (1.0 GB); add any HF GGUF via UI/CLI; no hardcoded paths (`config.json`, `ui_config.json`, `system_context.json`).

## [7.6.0] — 2026-06-06

**Universal Bridge (UNIBRI)** — cross-provider recall (OpenAI ↔ Ollama ↔ Cohere), cross-lingual (EN/FR/DE/ES/ZH), Latin name handling (taxonomic, diacritics, Roman numerals). 137/137 tests, 11 theorems (Broder, J-L, Wedin, Cormack).

## [7.5.0] — 2026-06-06

**Real BEIR benchmarks** — FAISS dense-only = SOTA on SciFact (0.7441 nDCG@10); ArguAna 0.6613; LIRS 100% recovery; KL router 100% tier-routing accuracy; immunological 100% anomaly detection.

## [7.2.0] — 2026-06-06

**Latency optimization** — LRU result cache (10K entries, 80-85% hit), 3ms warm path (vs 500ms cold), adaptive re-ranking, ONNX cross-encoder.

## [7.1.0] — 2026-06-06

**Retrieval research** — 4 approaches (raw / BM25 / hybrid RRF / hybrid+CE); 130 new tests, 0 regressions; key finding: dense-only = SOTA for scientific retrieval.

## [7.0.0] — 2026-06-06

**Doctoral-grade memory** — 8 algorithms (Ebbinghaus, SparseCoding, Variational, CrossAttention, Hyperbolic, InfoNCE, NeuralODE, Mahalanobis), 6 theorems with proofs, 9.3× compression (1,088,000 → 116,976 bytes), 49/49 tests, 100% V6-compatible.

## [6.0.0] — 2026-06-06

**MATHIRPlugin (LLM-agnostic API)** — `MATHIRPlugin` for any LLM/embedding dim; 4-tier memory + KL router + TurboQuant. 12/12 tests.

## [5.0.0] — 2026-01

**KL router + immunological memory** — KL-divergence constrained router; immunological anomaly detection. 21 bug fixes in V5.1.

## [4.0.0] — 2026-01

**Manifold-Constrained Hyper-Connections** — mHC integration (DeepSeek paper), Sinkhorn-Knopp projection, Lyapunov-based adaptive omega.

## [1.0.0–3.0.0] — 2026-01

**Core architecture** — CNN + MLP vision encoder, 3-tier memory (Working/Episodic/Semantic), basic RL training loop. Pre-history; full source in `_deprecated/legacy_v1_v3/`.

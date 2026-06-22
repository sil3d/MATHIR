# Changelog

All notable changes to MATHIR are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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

### Memory System

- **SimpleMemory** — New FTS5-only memory class (`mathir_dropin/simple.py`), zero external dependencies
- **get_last(n)** — Always include last N memories for context
- **search_context()** — One-call method for LLM context injection (recall + last, deduplicated)
- **DB preservation** — `setup_memory()` no longer deletes DB on restart
- **Thread safety** — Concurrent access via WAL mode + per-operation connections
- **31/31 audit checks** — Architecture, store, recall, edge cases, concurrency

### UI Overhaul

- **SVG icons** — Replaced all emoji with clean SVG icons
- **Chat history** — Persisted in localStorage (survives page reload)
- **Backend camera** — OpenCV via API (not browser getUserMedia)
- **6 views** — Chat, Camera, Models, Memory, Accuracy, Settings
- **Dark theme** — 8px grid, responsive layout

### API Fixes

- Fixed 6 broken frontend→backend routes (were returning 404)
- `/api/system/context` + `/api/system/info`
- `/api/models/switch`, `/api/models/toggle`, `/api/models/add-from-hf`
- `/api/accuracy/test` (was `/api/accuracy/run`)
- Audio via `/api/chat` with audio field

### System Prompt

- Reduced from ~1100 tokens to ~126 tokens
- Removed rigid 4-section template forcing
- Model responds naturally instead of following template

### Other

- Fixed `localhost` hardcoded → `127.0.0.1`
- Model auto-loads at startup (LFM2.5-VL-1.6B by default)
- Updated README.md for GitHub presentation
- Created AGENT.md (agent guide)
- LaTeX research paper for scientific review (`docs/MATHIR_Research_Paper.tex`)
- Benchmark methodology documented (dataset, queries, metrics, hardware, sources)
- Added `simple_memory_demo.py` example (zero deps)
- Cleaned workspace (old files archived, temp dirs removed)
- Removed LSTM references from docs (kept as historical citations only)
- Updated docs/28_HOW_TRAINING_WORKS.md (modern training workflow)
- Updated docs/02_MASTER_REFERENCE.md, docs/03_MASTER_QA_GUIDE.md
- Excluded large files from git (GGUF, DLL, binaries)

### Memory Management

- **`/api/memory/delete`** — Delete by ID or clear all memories
- **Settings view** — Create, view, delete memories in MATHIR Memory section
- **Playground** — Memory panel with delete buttons
- **Full conversation storage** — Stores complete Q&A (not truncated to 200 chars)
- **Skip trivial messages** — "hi", "ok", "thanks" not stored
- **File markers** — `[IMAGE ATTACHED]` / `[AUDIO ATTACHED]` in memory

### Chat Playground

- **`playground.html`** — New standalone chat UI at `/playground.html`
- **Multi-session** — Create new chats, switch between, delete
- **Model load modal** — See all models, capabilities, switch mid-chat
- **Image drag & drop** — Attach images directly
- **Camera integration** — Start/stop backend camera
- **Hold-to-talk** — Audio recording
- **Export chat** — Save conversation as .txt

### MATHIR Status

- **Status indicator** — Green/red dot showing MATHIR connection status
- **Memory count** — Shows "MATHIR: connected (N memories)"
- **Auto-refresh** — Status checked every 15 seconds

### No Hardcoded Values

- **Language** — `lang` HTML attribute set from `ui_config.json` (not hardcoded `en`)
- **All routes** — Frontend calls match backend endpoints
- **All paths** — Relative to config files

---

## [7.7.0] — 2026-06-06

### Vision & Audio Testing UI

- Complete web UI in `vision_testing/` for testing vision/audio models
- Flask backend with 17 API routes
- Web UI with Chat, Camera, Models, Memory, Settings views
- CLI tools: `model_manager.py`, `setup_binaries.py`, `download_models.py`

### Models

- LFM2.5-VL-1.6B-GGUF (vision-language) — 1.2 GB Q4_0
- LFM2.5-Audio-1.5B-GGUF (audio understanding) — 1.0 GB Q4_0
- Add ANY HF GGUF model via UI or CLI

### Configuration

- `config.json` — All model paths (no hardcoded)
- `ui_config.json` — UI settings (port, camera, audio)
- `system_context.json` — System prompt for models

---

## [7.6.0] — 2026-06-06

### Universal Bridge (UNIBRI)

- Cross-provider recall (OpenAI ↔ Ollama ↔ Cohere)
- Cross-lingual recall (EN ↔ FR ↔ DE ↔ ES ↔ ZH)
- Latin name handling (taxonomic, diacritics, Roman numerals, abbreviations)
- 137/137 tests pass
- 11 mathematical theorems (Broder, Johnson-Lindenstrauss, Wedin, Cormack)

---

## [7.5.0] — 2026-06-06

### Real BEIR Benchmarks

- Dense-only FAISS = SOTA on SciFact (0.7441 nDCG@10)
- ArguAna complete (0.6613 nDCG@10)
- LIRS eviction: 100% recovery after stress
- KL router: 100% tier-routing accuracy
- Immunological: 100% anomaly detection

---

## [7.2.0] — 2026-06-06

### Latency Optimization

- LRU result cache (10K entries, 80-85% hit rate)
- 3ms warm path latency (vs 500ms cold)
- Adaptive re-ranking
- ONNX cross-encoder support

---

## [7.1.0] — 2026-06-06

### Retrieval Research

- 4 retrieval approaches (A: raw, B: BM25, C: hybrid RRF, D: hybrid+CE)
- 130 new tests, 0 regressions
- Key finding: dense-only = SOTA for scientific retrieval

---

## [7.0.0] — 2026-06-06

### Doctoral-Grade Memory

- 8 new algorithms (Ebbinghaus, SparseCoding, Variational, CrossAttention, Hyperbolic, InfoNCE, NeuralODE, Mahalanobis)
- 6 novel theorems with full proofs
- 9.3× compression (1,088,000 → 116,976 bytes)
- 49/49 unit tests pass
- 100% backward compatible with V6

---

## [6.0.0] — 2026-06-06

### MATHIRPlugin (LLM-Agnostic API)

- `MATHIRPlugin` class — works with any LLM, any embedding dimension
- 4-tier memory (Working, Episodic, Semantic, Immunological)
- KL-constrained router
- TurboQuant compression
- 12/12 tests pass

---

## [5.0.0] — 2026-01

### KL Router + Immunological Memory

- KL-divergence constrained router (prevents collapse)
- Immunological memory (anomaly detection)
- 21 bug fixes (V5.1)

---

## [4.0.0] — 2026-01

### Manifold-Constrained Hyper-Connections

- mHC integration (DeepSeek paper)
- Sinkhorn-Knopp projection
- Lyapunov-based adaptive omega

---

## [1.0.0–3.0.0] — 2026-01

### Core Architecture

- CNN + MLP vision encoder
- 3-tier memory (Working, Episodic, Semantic)
- Basic RL training loop

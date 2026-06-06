# Changelog

All notable changes to MATHIR are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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

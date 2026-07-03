# MATHIR Changelog

## [8.6.1] — 2026-07-03 — PORTABLE PATHS + CROSS-PLATFORM INSTALL FIX

### Fixed
- **Eliminated ALL hardcoded machine-specific paths** from the entire codebase
  - Removed `C:\Users\<username>\...` and `D:\SECRET_PROJECT\...` references (39 occurrences across 8 files)
  - Removed ALL `~/.config/opencode/bin/` legacy paths (~50 occurrences across 18 files)
- **Migrated install location** from legacy `~/.config/opencode/bin/` to canonical `~/.config/MATHIR/mathir_mcp/`
- **Daemon launch bug** — `auto_start.bat` was launching `mathir_daemon.py` (raw TCP socket) instead of `mathir_server.py` (HTTP/Flask). Clients make HTTP requests, so the TCP daemon accepted connections but never responded. Fixed in all auto-start scripts.
- **VBS launcher** (`auto_start_vbs.vbs`) — replaced hardcoded path with `WScript.Shell.ExpandEnvironmentStrings("%USERPROFILE%")` for portable resolution
- **Python scripts** (`migrate_mathir_schema.py`, `test_batch_recall.py`) — replaced hardcoded paths with `os.path.expanduser("~")` based resolution
- **install_smart.py** macOS plist generation — plist paths now use `~/.config/MATHIR/` instead of `~/.config/opencode/`
- Added `.claude/` to `.gitignore` (contains machine-specific hook paths, must not be committed)

### Changed
- All cross-platform scripts (`.bat`, `.ps1`, `.sh`, `.service`, `.plist`) now resolve paths from `%USERPROFILE%` / `$HOME` / `~`
- All install guides (Windows, Linux, macOS) updated to `~/.config/MATHIR/mathir_mcp/` paths
- All agent/command/docs templates (opencode + mimocode) updated to new canonical paths
- `GLOBAL_INSTRUCTIONS.md` deployed path updated

### Files touched (18 files)
- `bin/auto_start.bat`, `bin/auto_start.sh`, `bin/auto_start_vbs.vbs`
- `bin/auto_start_helpers.ps1`, `bin/auto_start_healthcheck.ps1`
- `bin/start_daemon_background.ps1`, `bin/setup-autostart.ps1`
- `bin/mathir-daemon.service`, `bin/com.mathir.daemon.plist`
- `dev/migrate_mathir_schema.py`, `dev/test_batch_recall.py`
- `INSTALL_FOR_AGENT/INSTALL_WINDOWS.md`, `INSTALL_FOR_AGENT/INSTALL_LINUX.md`, `INSTALL_FOR_AGENT/INSTALL_MACOS.md`
- `INSTALL_FOR_DEV/README.md`, `INSTALL_FOR_DEV/install_smart.py`
- `GLOBAL_INSTRUCTIONS.md`
- All `opencode_templates/` and `mimocode_templates/` inject files

## [8.6.0] — 2026-07-03 — INT8 QUANTIZATION + MULTI-AGENT BENCHMARK

### Added
- **INT8 scalar quantization** for embedding storage (algo #22) — 4x compression, 0% recall loss
  - `_quantize_int8()` / `_serialize_embedding()` — float32 → int8 at store time
  - `vec_int8(X'...')` SQL function for sqlite-vec INT8 table operations
  - Automatic FLOAT→INT8 migration on first DB access (transparent, zero-downtime)
  - Migrated 410 production DBs: 1.9 GB → 825 MB (2.3x average reduction)
  - Proven 10/10 recall@10 overlap vs float32 (zero ranking degradation)
- **Cross-encoder reranking** (algo #21) — `CrossEncoderReranker` class
  - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, lazy-loaded)
  - Integrated into `HybridSearch.hybrid_search(rerank=True)` — fetches k*3, reranks to k
  - +20pp hit@10 on natural-language queries (50% → 70%)
  - `rerank` parameter added to MCP `memory_hybrid_search` tool and HTTP API
- **Multi-agent shared memory benchmark** (`benchmarks/08_industry_validation/multi_agent_bench.py`)
  - Tests whether MATHIR can make "dumb" (free-tier) models intelligent via shared memory
  - 3 phases: baseline (no memory) → MATHIR-assisted → multi-agent collaboration
  - Result: **0% → 53% average accuracy** with MATHIR (+70pp for best agent)
  - Supports OpenCode Zen free models + MiniMax native API + Groq
- **LoCoMo benchmark** on Groq (Llama 3.3 70B) and OpenCode Zen (mimo/deepseek/nemotron)
  - Groq partial: 51.2% (21/41 judged, TPM-limited)
  - Zen (mimo+deepseek free): 38.8% (67/152 judged)
- **e5-small vs e5-large-v2** embedding comparison on fluid mechanics corpus
  - e5-small + rerank beats e5-large (52.9% vs 51.0%) at 47x less encoding cost
- 4 new cross-encoder reranking tests (98 total tests passing)

### Fixed
- `find_duplicates()` — adapted to INT8: queries use `vec_int8(X'...')` for MATCH
- `mathir_server.py` hybrid search — adapted vector query to INT8 format
- Groq LLM client — added 413 retry (TPM rate limit masquerades as "Request too large")
- Groq LLM client — added `User-Agent` header (fixes Cloudflare 403)
- LoCoMo context truncation — `MATHIR_BENCHMARK_CONTEXT_MAX_CHARS` env var

### Changed
- `vec_memories` table schema: `FLOAT[dim]` → `INT8[dim]` (new DBs)
- `_serialize_embedding()` returns hex string (was bytes blob)
- `_deserialize_embedding()` decodes int8 → float32 (was float32 → float32)
- Test `test_search_include_embeddings_returns_raw_vectors`: asserts cosine > 0.99 (was atol=1e-5)

## [8.5.1] — 2026-07-01 — BUG FIXES + AUDIT LOG

### Added
- `memory_audit` table + `_log_audit()` — real audit trail for save/delete/promote/decay/link
- `stability` column on memories table (Ebbinghaus decay persistence)
- Anomaly threshold recalibrated from 2.0 → 25.0 using production data
- `reset_anomaly_state()` function for full anomaly detector reset

### Fixed
- Embedder "meta tensor" crash: root-caused to PyTorch 2.6 `weights_only=True` default
- Anomaly state reset: clearing DB row alone doesn't reset a running daemon's cached detector

## [8.5.0] — 2026-06-25 — FASTMCP REWRITE + AUTO-INJECTION + MULTI-SESSION

### Changed
- MCP server rewritten using FastMCP 3.4.2 (replaces hand-rolled JSON-RPC stdio loop)
- MCP server v3 = thin HTTP proxy to daemon (port 7338) — NO local embedder loading
- 20 MCP tools (2 auto-injection + 10 basic + 7 lifecycle + 1 health check)
- Multi-session safe: multiple OpenCode sessions share ONE daemon embedder (no CUDA conflicts)
- Unified Flask+Waitress server (mathir_server.py) replaces TCP daemon + http.server
- Auto-injection plugin (mathir-auto-inject.ts) injects memories into system prompt
- `/api/context` endpoint for plugin auto-injection
- `memory_session_start` + `memory_context` tools for session context
- Registry-based DB resolution (checks registry → projects dir → CWD → legacy)
- `127.0.0.1` instead of `localhost` (Windows IPv6 resolution delay)
- Dependencies: added `fastmcp>=3.4.0`, removed `aiohttp`, `pyzmq` (no longer needed)

### Fixed
- PyTorch 2.6 meta tensor crash ("Cannot copy out of meta tensor; no data!")
- Multi-session CUDA crash — root cause was 2+ MCP servers each loading embedder on GPU
- Missing `import threading` in MCP server (crash on startup)
- Hardcoded path in `get_project_db_path`
- NULL embeddings (17 memories saved via direct sqlite had NULL vectors)

### Security
- Input length caps: content 100KB, query 5KB, label 200B, agent 100B
- Memory ID validation regex

## [8.4.0] — 2026-06-23 — LIVING MEMORY

### Added
- 7 lifecycle tools: memory_promote, memory_auto_promote, memory_decay, memory_consolidate, memory_link, memory_get_links, memory_build_links
- 4-phase memory lifecycle: promote → decay → consolidate → link graph
- Spreading activation via link graph (Collins & Loftus 1975)
- Ebbinghaus decay (5%/30d) for unused memories
- Immunological tier with Mahalanobis anomaly detection
- HTML dashboard reports with Chart.js (dark theme)
- ONBOARDING.md, INSTALL_FOR_AGENTS.md

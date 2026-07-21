# MATHIR Changelog

## [8.9.5] — 2026-07-21 — AUTONOMOUS MAINTENANCE + HEADLESS GOD-MODE WORKERS

### Added
- **Autonomous "sleep" maintenance thread** (`_maintenance_loop` in `mathir_server.py`) — a background daemon thread that periodically runs `run_maintenance()` (decay/promote/dedupe/link-build) on every DB currently cached in `_vec_cache`. Previously `run_maintenance()` existed but nothing ever invoked it automatically — lifecycle transitions only happened if a human or agent explicitly called it. Config-driven via a new `"maintenance"` block in `mathir.json` (`enabled`, `interval_hours`, `do_decay`, `do_promote`, `do_dedupe`, `do_links`), with `MATHIR_MAINTENANCE_ENABLED`/`MATHIR_MAINTENANCE_INTERVAL_HOURS` env overrides.
- **Headless, on-demand god-mode workers** — `bin/god/god_mode_start.py` / `god_mode_stop.py` launch/kill a headless coding-agent CLI (opencode, mimo, claude, openclaude, codex, gemini, aider, cursor-agent, copilot) as a detached background process, human-triggered only, never autostarted. `bin/god/god_worker_daemon.py` is the actual execution loop: polls `/api/god/poll`, claims a task, spawns the target CLI with its documented headless flags, streams output live, retries on a silent no-op (process exits 0 but never called `memory_save`) or a hang past its timeout, then acks the result. Closes the gap where `god_bridge.py` only notified and needed a human to manually drive execution.
- **`bin/god/god_mode_report.py`** — deterministic, LLM-independent text report that reads a project's SQLite DB directly (bypassing `/api/memories`'s project-resolution quirks) and groups results by task, correctly handling multi-target fan-out. Following a real incident (2026-07-21: 3 workers' answers never reached the human because the orchestrator's own relay failed), `mathir_god_orchestre()`'s built-in prompt guidance now mandates running this after every dispatch round instead of relying on the orchestrator LLM's memory.
- **`/api/god/ack` daemon route** — flips a claimed task's label in place (`pending`→`running`→`completed`/`failed`) instead of creating a duplicate memory per state change, fixing `god_poll` re-serving the same stale pending task forever.
- **Atomic task claiming** — `/api/god/poll` now wraps its SELECT+UPDATE in `BEGIN IMMEDIATE`/commit, preventing two parallel pollers sharing an agent name from double-claiming the same task.
- **`memory_save` `file_path` param** — persisted into `metadata.file_path`; new `/api/memory/by_path` route does a real `json_extract(metadata, '$.file_path') LIKE ?` SQL filter (replacing the old semantic-recall-then-post-filter approach, which silently missed structurally-tagged-but-not-semantically-similar memories).
- **`memory_dashboard`** now calls a genuinely distinct `/api/memory/dashboard` route (recent activity, guardrail roster, save trend) instead of secretly proxying `/api/memory/stats` verbatim.
- **Self-healing + worker-silence guidance for orchestrators** — new prompt sections requiring `mathir_god_orchestre()` to fix bugs it hits in MATHIR's own code directly (syncing to the deployed daemon path and restarting it, per the existing sync guardrail), with third-party tool bugs only proposed as GitHub issues pending explicit human approval.

### Fixed
- **`created_at_unix` decay-eligibility bug** — was hardcoded to literal SQL `NULL`, which via `COALESCE(last_recalled_at, created_at_unix, 0)` permanently excluded every never-recalled memory from decay eligibility. Fixed to `CAST(strftime('%s', created_at) AS REAL)`.
- **Guardrail saves no longer misfire the anomaly detector** — `memory_save`'s anomaly check is now skipped for `block_type == 'guardrail'` saves, since new guardrails almost always describe novel problems (exactly what the Mahalanobis detector is tuned to flag), which previously caused real guardrail saves to get silently reclassified to `tier=immunological` and vanish from the always-injected GUARDRAILS block.
- **`memory_recall_quality` lexical grounding check** — added a requirement that at least one ≥4-char query token literally appear in the top result's text before trusting a "high" (≥0.7 cosine) quality verdict, after a verified case where a nonsense query scored 0.839 purely from embedding-space coincidence.

### Changed
- **`build_links_all` threshold raised 0.7 → 0.88** — 0.7 against the real `multilingual-e5-small` embedding model produced an almost-complete graph (442,890 links from 666 memories), useless as a "related memories" signal.
- **`god_bridge.py` env vars renamed** `MYCERISE_STATE_DIR`/`MYCERISE_LOG_FILE` → `MATHIR_STATE_DIR`/`MATHIR_LOG_FILE` — leftover naming from before MATHIR was extracted into its own standalone project; defaults still resolve under `$XDG_CONFIG_HOME`, no functional behavior change beyond the var name itself.
- **`memory_consolidate(dry_run=True)` output bounded** — new `max_results` param (default 50) caps preview events to a compact `{memory_id_a, memory_id_b, snippet_a, snippet_b, similarity}` shape (100-char snippet cap), replacing the old verbose per-pair shape that hit 228K+ chars on ~700 memories.
- **`/api/memory/export` is now file-based** — writes the full export to `<DATA_DIR>/exports/export_{project_slug}_{timestamp}.json` on disk and returns just the file path + count, instead of inlining potentially huge JSON (previously hit 137,102 chars at 755 memories).

---

## [8.9.4] — 2026-07-16 — SELF-HEALING DAEMON + UNIVERSAL INJECTION PROXY

### Fixed
- **Windows healthcheck watchdog required admin, so it silently never installed.** `setup-autostart.ps1`'s 5-minute self-heal task needed `-InstallHealthcheck` + `-RunLevel Highest`, which most non-admin installs don't have — a daemon that died mid-session (crash, sleep/resume) stayed dead until next logon. The healthcheck doesn't actually need elevation; it's installed by default now (`-SkipHealthcheck` to opt out), and `install_smart.py`'s Windows autostart path registers it automatically.
- **`claude_code_hook.py` (UserPromptSubmit auto-injection hook) existed but was dead code** — never wired into any `settings.json`, and even when invoked manually it read `hook_input["message"]` while Claude Code's real payload field is `"prompt"`, so it silently injected nothing. Fixed the field name and wired it into `~/.claude/settings.json`; `install_smart.py` now does this automatically for Claude Code installs.
- **Daemon-side prompt-injection sanitizer (`_sanitize_for_prompt`) was a no-op.** `s.replace(tok, tok.strip())` does nothing when `tok` has no surrounding whitespace — true for every token in the list (`</mathir-`, `{{MATHIR_CONTEXT}}`, `<|`, `### `). Confirmed live: a query/memory containing `</mathir-auto-injection>` passed through unstripped into `/api/context`'s response, which the hook wraps in exactly that tag — a stored-memory or query-based prompt-injection breakout, live in production since the hook above was wired up. Fixed, then unified with `mathir_proxy.py`'s (correct) independent copy into one shared `mathir_sanitize.py` so the two can't drift apart again.
- **`mathir_proxy.py`'s OpenAI-route default target had a double-`/v1` bug** (`https://api.openai.com/v1` + route path `/v1/chat/completions` → `.../v1/v1/chat/completions`) — would have 404'd on every real OpenAI-format request the proxy ever forwarded. Never caught because the proxy had never been run end-to-end before this session. Default target is now `https://api.openai.com`.

### Added
- **`mathir_proxy.py` now speaks Anthropic's native `/v1/messages`** (previously OpenAI-compatible `/v1/chat/completions` only), augmenting the top-level `system` field — this is the endpoint Claude Code itself calls, so without this route the proxy could not help the tool most people actually use.
- **Multi-upstream routing**: an optional `X-Mathir-Upstream` request header lets one proxy process serve requests bound for different providers (Anthropic, OpenAI, OpenRouter, a local model, ...) instead of needing one process per upstream. Validated against an allowlist (~30 known providers + `*.openai.azure.com` / `*.bedrock-runtime.amazonaws.com` suffix matching + loopback for any local model server) — falls back to the process's configured default on any unlisted host, never fails open.
- **`mathir-proxy.service` (systemd) and `com.mathir.proxy.plist` (launchd)**, mirroring the existing daemon units, so the proxy gets the same `Restart=on-failure` / `KeepAlive` self-healing on Linux/macOS. Windows healthcheck now checks both port 7338 (daemon) and 7339 (proxy).

### Removed
- **`mathir_mcp/brain/`** — a stale, unimported fork of `mathir_watchdog.py`, `mathir_prime.py`, `mathir_inject_proxy.py`, and `mathir_brain.py`, superseded by the `mathir_lib/` versions (confirmed by diff + mtime) and already flagged as legacy dead weight in `tests/test_module_tree.py`. Removed the now-stale `brain -> bin` entry from `mathir_sync.py`'s sync manifest.

## [8.9.3] — 2026-07-15 — CROSS-PLATFORM AUTO-START FIX (hardcoded Python paths)

### Fixed
- **`bin/auto_start.bat` (Windows)** — was hardcoded to `%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe`, which silently failed the daemon launch on any machine using a different Python (e.g. Miniconda — the case that surfaced this bug). Now resolves dynamically: `where python` on PATH → `py` launcher → common install locations (Miniconda, Anaconda, WindowsApps, `Programs\PythonXXX`).
- **`bin/mathir-daemon.service` (Linux/systemd)** — `ExecStart` hardcoded `/usr/bin/python3`, which doesn't exist on conda-only or minimal setups. Now resolves `python3`/`python` via `PATH` at start time (`ExecStart=/bin/sh -c 'exec "$(command -v python3 || command -v python)" ...'`), with a fallback `PATH` covering `~/.local/bin`, `~/miniconda3/bin`, `~/anaconda3/bin`.
- **`bin/com.mathir.daemon.plist` (macOS/launchd)** — default interpreter changed from hardcoded `/usr/bin/python3` (removed on newer macOS, never used by Homebrew/conda) to `/usr/bin/env python3` (PATH resolution at launch), with a documented `PATH` fallback covering `/opt/homebrew/bin`.
- **`INSTALL_FOR_DEV/install_smart.py`** (`_setup_autostart_macos`) — the actual code path that renders the deployed plist. Resolution order changed to: venv python (unchanged) → `shutil.which("python3")` on the installer's own PATH (new — covers Homebrew/conda/pyenv) → `/usr/bin/env python3` fallback (new). Previously defaulted straight to the hardcoded `/usr/bin/python3` if no venv was found.
- `auto_start.sh` (Linux/macOS) and `bin/mathir_daemon.py` (the HTTP-shim launcher) were already correct — no change needed there.

Synced to `~/.config/MATHIR/mathir_mcp/` (verified byte-identical post-sync).

---

## [8.9.2] — 2026-07-05 — GOD-MODE CLIENT BRIDGE

### Added
- **`bin/god/god_bridge.py`** — cross-platform polling daemon for god-mode (3 modes: `worker` / `orchestrator` / `observer`), stdlib-only, no new deps. Beeps + logs when new `god:*` events arrive.
- **`bin/god/god_poll.ps1`** — Windows PowerShell one-shot poller.
- **`bin/god/god_poll.sh`** — POSIX bash one-shot poller.
- **`bin/god/PROTOCOL.md`** — full label taxonomy + message flow.
- **`bin/god/README.md`** — usage + env vars (`MATHIR_DAEMON_URL`, `MYCERISE_STATE_DIR`, `MYCERISE_LOG_FILE`).
- **`docs/GOD_MODE.md`** — new "Client-side tooling (`bin/god/`)" section explaining modes, quick start, cross-platform.
- **Top-level `README.md`** — added client bridge link + bumped to v8.9.1.
- **`mathir_mcp/README.md`** — new "God-mode orchestration" section.

### Changed
- Daemon `/health` reports `version: 8.9.1` (was `8.5.1`) after pip editable install refresh.
- `__version__` + `pyproject.toml`: `8.5.1` → `8.9.1`.
- `docs/GOD_MODE.md` extended with client-side tooling section (was server-only).
- Default tool count docs: `26 tools` → `27 tools` (mathir_god_agent + mathir_god_orchestre were missing from the count).

### Synced across all 3 MATHIR install locations
- `D:\SECRET_PROJECT\MATHIR\mathir_mcp\` (github source / truth)
- `~/.config/MATHIR/mathir_mcp/` (egg install used by Claude / MiMo / OpenCode — `bin/god/` populated, `.mathir/` DB preserved, 780 memories intact)
- `~/.config/mimocode/tools/mathir_mcp/` (v8.4.0 dev copy — 12 drifted files re-synced, `bin/god/` populated)

### Hygiene
- 4 stale `god:task:*:pending` entries purged (had completed siblings).
- `~/.config/MATHIR/mathir_mcp-8.5.1.dist-info` (stale pip editable-install metadata) refreshed to `8.9.1` via `pip install --force-reinstall --no-deps -e ...`.
- Mycerise `scripts/god-mode/` + `scripts/mathir/` deleted (were polluting namespace, never belonged in the consumer project).

---

## [8.9.1] — 2026-07-05 — DOC CORRECTION (27 tools canonical)

### Fixed
- **Canonical tool count corrected: 26 → 27** — verified against live MCP tool list.
  - The `audit_immunological` tool was missing from the v8.9.0 count.
  - All docs, templates, READMEs, and architecture diagrams now consistently say **27 MCP tools**.
- Canonical breakdown: 2 auto-injection + 10 basic + 7 lifecycle + 3 advanced + 1 guardrail + 1 immunological + 1 health + 2 god mode = **27**.
- `block_type` declarations everywhere now list all 6 tiers: `working_memory | episodic | semantic | procedural | guardrail | immunological`.
- `LIVING MEMORY (5 TIERS)` headers → `(6 TIERS)` in all 150+ agent/skill files.
- Guardrail row added to all memory-tier tables (was missing in skill/agent injections).
- Version titles updated to v8.9.0 (was stale v8.5.0/v8.6.0/v8.7.0 in some deployments).
- Templates `opencode_templates/README.md`: "19 MCP tools" → "27 MCP tools", "5-tier model" → "6-tier model".

### Synced deployments
- `~/.config/MATHIR/` (mathir_mcp full copy)
- `~/.config/opencode/` (32 agents + 82 skills + 82 skills-global re-injected)
- `~/.config/mimocode/` (32 agents re-injected)
- `~/.claude/CLAUDE.md`

---

## [8.9.0] — 2026-07-05 — GUARDRAIL TIER

### Added
- **`guardrail` tier** — 6th memory tier, push-based. Always auto-injected into every `/api/context`, `memory_session_start`, `memory_context` response.
- **`memory_list_guardrails`** MCP tool — list all active guardrails
- **`/api/memory/guardrails`** daemon route — GET/POST guardrail listing
- **`list_guardrails()` / `count_guardrails()`** methods in VecMemory
- Guardrails are immune to decay and promotion (terminal tier)
- Minimum priority enforced at 8, max 50 per project

### Changed
- Total MCP tools: 27 (corrected in 8.9.1; was miscounted as 26 at release)
- Total tiers: 6 (was 5)

---

## [8.8.0] — 2026-07-04 — GOD MODE — MULTI-AGENT ORCHESTRATION

### Added
- **`mathir_god_agent`** MCP tool — worker self-identification + task polling. Call with no args → agent self-assesses honestly (name, capabilities, strengths, weaknesses). Call with profile → register + poll.
- **`mathir_god_orchestre`** MCP tool — orchestrator discovers workers with full profiles, decomposes directives, assigns tasks by worker strength.
- **`mathir_god.py`** core module — `GodProtocol` (label encoding), `TaskGraph` (DAG + cycle detection), `WorkerRegistry` (capability lookup), `WorktreeManager` (git worktree lifecycle)
- **`/api/god/poll`** daemon route — optimized SQL polling for pending tasks
- **`/api/god/agents`** daemon route — list registered workers with introductions
- **Built-in helpers** — `name="help"` and `directive="help"` return full usage guides
- **LIKE injection prevention** — `%` and `_` escaped in god/poll queries
- 40 new tests in `tests/test_god.py`
- `docs/GOD_MODE.md` — full architecture and usage guide

### Changed
- Total MCP tools: 25 (was 23)

---

## [8.7.0] — 2026-07-03 — 3-LAYER AUTO-CACHE

### Added
- **3-layer auto-cache system** (`mathir_cache.py`) for significant performance boost:
  - **L1 Embedding Cache** — LRU (1024 entries) on `_encode_query()`/`_encode_passage()`. Same text → instant lookup (~60ms → <1ms). Deterministic, never expires.
  - **L2 Recall Cache** — TTL-based (256 entries, 60s TTL) on `/api/memory/recall` results. Deduplicates identical queries across agents. Invalidated on any write (save/delete/promote/consolidate).
  - **L3 Session Cache** — Pre-warmed top-20 memories per project (5 min TTL) on `/api/context`. Session start and context calls return instantly on repeat.
- **`/api/cache/stats`** endpoint — hit/miss counters and hit ratio for all 3 layers
- **`cache` field** in recall responses — `"hit"` or `"miss"` for observability
- **Write-through invalidation** — `invalidate_on_write()` called on save, delete, promote, auto_promote, consolidate (non-dry-run)
- 24 new cache tests (LRU eviction, TTL expiry, invalidation, stats, integration)

### Changed
- `_encode_query()` / `_encode_passage()` now check L1 cache before calling `embedder.encode()`
- `/api/memory/recall` checks L2 cache before running vector search
- `/api/context` checks L3 cache before running search
- Total tests: 122 (was 98)

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
- Added `.claude/` and `.mimocode/` to `.gitignore` (local state, must not be committed)
- **DB routing** — `_resolve_db()` was creating DBs in global `~/.config/MATHIR/data/projects/` even when a project cwd was provided. Now creates `.mathir/mathir.db` inside the project directory for new projects.
- **DB routing backward-compat** — existing 417 databases in global config are still found: routing checks local `.mathir/` first, then global, only creates local for genuinely new projects.
- **Case-sensitivity** — standardized `~/.config/MATHIR` (uppercase) everywhere. The lowercase `~/.config/mathir` variant would create a separate directory on Linux/macOS.
- Removed legacy `~/.config/opencode` fallback from `mathir_paths.py`

### Changed
- All cross-platform scripts (`.bat`, `.ps1`, `.sh`, `.service`, `.plist`) now resolve paths from `%USERPROFILE%` / `$HOME` / `~`
- All install guides (Windows, Linux, macOS) updated to `~/.config/MATHIR/mathir_mcp/` paths
- All agent/command/docs templates (opencode + mimocode) updated to new canonical paths
- `GLOBAL_INSTRUCTIONS.md` deployed path updated
- DB routing priority: local `.mathir/` → global `~/.config/MATHIR/data/` → create local

### Files touched (24 files)
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

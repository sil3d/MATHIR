# MATHIR Changelog

## [8.5.0] — 2026-06-25 — ⚡ FASTMCP REWRITE + AUTO-INJECTION + MULTI-SESSION

### Changed
- MCP server rewritten using FastMCP 3.4.2 (replaces hand-rolled JSON-RPC stdio loop)
- MCP server v3 = thin HTTP proxy to daemon (port 7338) — NO local embedder loading
- 19 MCP tools (2 auto-injection + 10 basic + 7 lifecycle)
- Multi-session safe: multiple OpenCode sessions share ONE daemon embedder (no CUDA conflicts)
- Unified Flask+Waitress server (mathir_server.py) replaces TCP daemon + http.server
- Auto-injection plugin (mathir-auto-inject.ts) injects memories into system prompt
- `/api/context` endpoint for plugin auto-injection
- `memory_session_start` + `memory_context` tools for session context
- Registry-based DB resolution (checks registry → projects dir → CWD → legacy)
- `127.0.0.1` instead of `localhost` (Windows IPv6 resolution delay)
- Dependencies: added `fastmcp>=3.4.0`, removed `aiohttp`, `pyzmq` (no longer needed)
- Version bumped to 8.5.0

### Fixed
- PyTorch 2.6 meta tensor crash ("Cannot copy out of meta tensor; no data!") — MCP v3 avoids by not loading embedder
- Multi-session CUDA crash — root cause was 2+ MCP servers each loading embedder on GPU
- Missing `import threading` in MCP server (crash on startup)
- Hardcoded `Desktop/SECRET_CODE/Mycerise_V2_Taur` path in `get_project_db_path`
- NULL embeddings (17 memories saved via direct sqlite had NULL vectors)
- Bun v1.3.13 segfault on Windows: added `"runtime": {"backend": "node"}` to opencode.json
- config_template.json: portable paths, no OpenCode hardcodes
- OpenRouter API key purged from git history (commit 2a45de0)

### Security
- Input length caps retained: content 100KB, query 5KB, label 200B, agent 100B
- Memory ID validation regex retained
- ABSOLUTE RULE: agent must never say "I don't have memory access"

## [8.4.0] — 2026-06-23 — 🧠 LIVING MEMORY

### Added
- 7 lifecycle tools: memory_promote, memory_auto_promote, memory_decay, memory_consolidate, memory_link, memory_get_links, memory_build_links
- 4-phase memory lifecycle: promote → decay → consolidate → link graph
- Spreading activation via link graph
- Ebbinghaus decay (5%/30d) for unused memories
- 26 new lifecycle tests in mathir_mcp/dev/test_lifecycle.py
- `--selftest`, `--list-tools`, `--version` CLI flags in `python -m mathir_mcp`
- ONBOARDING.md (683 lines) — standalone guide for new agents
- INSTALL_FOR_AGENTS.md (402 lines) — 4-layer integration for AI agent hosts
- HTML dashboard reports with Chart.js (dark theme MATHIR brand)
- 9/9 selftest, 173/173 pytest pass

### Fixed
- Stale `__editable__.mathir_lib-5.0.0.pth` + finder (shadowed new install)
- `__editable__.mathir_mcp-8.4.0.pth` pointed to package dir instead of parent
- Old empty `mathir_lib/` shim at repo root (namespace package shadow)
- `pyproject.toml` console scripts (mathir-client, mathir-watchdog broken)
- `__main__.py` import path (mathir_daemon → mathir_lib.mathir_daemon)
- mathir_mcp_server.py graceful mathir_dropin error with 3 fix options

### Changed
- 10 MCP tools → 17 MCP tools
- Install via `pip install -e ./mathir_mcp` (editable, portable)
- Console scripts: mathir-daemon, mathir-mcp, mathir-client, mathir-watchdog
- Tier taxonomy: working_memory | episodic | semantic | procedural | immunological (5 tiers; immunological is a real, queryable, writable tier for threat-signature / anomaly storage, not just an internal detection slot)

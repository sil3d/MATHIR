# MATHIR Changelog

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
- Tier taxonomy confirmed: working_memory | episodic | semantic | procedural (immunological is internal anomaly slot, NOT a tier)

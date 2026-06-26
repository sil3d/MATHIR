# MATHIR Architecture (v8.5.0 — Thin Proxy)

```
┌──────────────────────────────────────────────────────────────────┐
│                         AGENT HOST                               │
│  (OpenCode, Claude Code, Cursor, Cline, Roo Code, 50+ agents)   │
│                                                                  │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐ │
│  │  mathir-auto-inject │    │  GLOBAL_INSTRUCTIONS.md          │ │
│  │  (plugin)           │    │  "memory_session_start"          │ │
│  │                     │    │  "memory_context"                │ │
│  │  hook: system.      │    │  "memory_save"                   │ │
│  │  transform          │    │  mandatory triggers              │ │
│  └─────────┬───────────┘    └──────────────┬───────────────────┘ │
│            │                               │                     │
│            │  HTTP /api/context            │  MCP tools (stdio)  │
│            ▼                               ▼                     │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │                    MATHIR MCP SERVER (v3)                    ││
│  │                    Thin proxy — NO embedder                  ││
│  │                    Forwards to daemon via HTTP               ││
│  │                                                              ││
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  ││
│  │  │ 20 tools    │  │ /api/context │  │ /api/stats         │  ││
│  │  │ (19 memory  │  │ auto-inject  │  │ dashboard          │  ││
│  │  │  + health)  │  │              │  │                    │  ││
│  │  └──────┬──────┘  └──────┬───────┘  └────────────────────┘  ││
│  │         │                │                                   ││
│  │         ▼                ▼                                   ││
│  │    HTTP to daemon (127.0.0.1:7338)                          ││
│  └────────────────────┬─────────────────────────────────────────┘│
└───────────────────────┼──────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MATHIR DAEMON (Flask + Waitress)              │
│                    Port 7338 — 1 embedder (cached)              │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │ sentence-        │  │ mathir_vec.py    │  │ mathir.db      │ │
│  │ transformers     │  │ sqlite-vec       │  │ (sqlite)       │ │
│  │ 384d embeddings  │  │ cosine search    │  │                │ │
│  │ (cached global)  │  │                  │  │ memories       │ │
│  └──────────────────┘  └──────────────────┘  │ memory_links   │ │
│                                              │ memory_audit   │ │
│  Endpoints:                                  └────────────────┘ │
│    POST /api/memory/save, /recall, /stats, /delete, ...        │
│    GET  /api/context, /api/stats, /api/memories, /health       │
└──────────────────────────────────────────────────────────────────┘

CONFIGURATION (env vars — all paths are agent-agnostic):
  MATHIR_CONFIG      → config file (default: ~/.config/opencode/config/mathir.json)
  MATHIR_PROJECTS_DIR → projects directory (default: ~/.config/opencode/data/projects)
  MATHIR_DB          → legacy DB path (default: ~/.config/opencode/data/mathir.db)
  MATHIR_REGISTRY    → registry file (default: ~/.config/opencode/data/mathir_registry.json)

TIERS:
  working_memory → episodic → semantic → procedural
       ↑               ↑          ↑           ↑
    recall≥3        recall≥10   priority≥8
    age≥1d          age≥7d      label:how-to:

LIFECYCLE:
  Ebbinghaus decay:  -5% stability / 30 days no recall
  Consolidate:       cosine > 0.95 → merge duplicates
  Link graph:        cosine > 0.7 → weighted edges
```

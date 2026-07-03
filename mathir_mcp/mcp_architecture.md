# MATHIR Architecture (v8.6.0 — INT8 + Cross-Encoder)

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
│  │  │ 23 tools    │  │ /api/context │  │ /api/stats         │  ││
│  │  │ (22 memory  │  │ auto-inject  │  │ dashboard          │  ││
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
│  │ transformers     │  │ sqlite-vec INT8  │  │ (sqlite)       │ │
│  │ e5-small 384d    │  │ cosine search    │  │                │ │
│  │ + cross-encoder  │  │ 4x compressed    │  │ memories       │ │
│  │ (cached global)  │  │                  │  │ memory_links   │ │
│  └──────────────────┘  └──────────────────┘  │ memory_audit   │ │
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

STORAGE:
  Embeddings:        INT8 scalar quantization (384 bytes/vec, was 1536)
  Migration:         Automatic FLOAT→INT8 on first access (transparent)
  SQL:               vec_int8(X'...') function for sqlite-vec 0.1.9

RETRIEVAL:
  Vector:            cosine similarity via sqlite-vec INT8
  BM25:              Okapi BM25 (rank_bm25)
  Hybrid:            Vector + BM25 + RRF (k=60) fusion
  Reranking:         cross-encoder/ms-marco-MiniLM-L-6-v2 (optional, +20pp)

LIFECYCLE:
  Ebbinghaus decay:  -5% stability / 30 days no recall
  Consolidate:       cosine > 0.95 → merge duplicates
  Link graph:        cosine > 0.7 → weighted edges
  Anomaly:           Mahalanobis distance (threshold=25.0, immunological tier)
```

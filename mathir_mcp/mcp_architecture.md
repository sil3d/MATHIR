# MATHIR Architecture (v8.9.0 — 3-Layer Auto-Cache + INT8 + Cross-Encoder)

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
│  │  │ 26 tools    │  │ /api/context │  │ /api/stats         │  ││
│  │  │ (22 memory  │  │ auto-inject  │  │ /api/god/poll      │  ││
│  │  │ +2 god +1h) │  │              │  │ /api/god/agents    │  ││
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
│  ┌──────────────────────────────────────────────────────────────┐│
│  │              3-LAYER AUTO-CACHE (v8.9.0)                    ││
│  │  L1 Embedding LRU (1024)  — encode() ~60ms → <1ms          ││
│  │  L2 Recall TTL (256, 60s) — dedup queries across agents    ││
│  │  L3 Session (top-20, 5m)  — session_start/context instant  ││
│  │  Write-through invalidation on save/delete/promote/consol. ││
│  └──────────────────────────────────────────────────────────────┘│
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │ sentence-        │  │ mathir_vec.py    │  │ mathir.db      │ │
│  │ transformers     │  │ sqlite-vec INT8  │  │ (sqlite)       │ │
│  │ e5-small 384d    │  │ cosine search    │  │                │ │
│  │ + cross-encoder  │  │ 4x compressed    │  │ memories       │ │
│  │ (cached global)  │  │                  │  │ memory_links   │ │
│  └──────────────────┘  └──────────────────┘  │ memory_audit   │ │
│  Endpoints:                                  └────────────────┘ │
│    POST /api/memory/save, /recall, /stats, /delete, ...        │
│    GET  /api/context, /api/cache/stats, /api/memories, /health │
└──────────────────────────────────────────────────────────────────┘

CONFIGURATION (env vars — all paths are agent-agnostic):
  MATHIR_HOME        → base config dir (default: ~/.config/MATHIR)
  MATHIR_CONFIG      → config file (default: ~/.config/MATHIR/config/mathir.json)
  MATHIR_PROJECTS_DIR → projects directory (default: ~/.config/MATHIR/data/projects)
  MATHIR_DB          → legacy DB path (default: ~/.config/MATHIR/data/mathir.db)
  MATHIR_REGISTRY    → registry file (default: ~/.config/MATHIR/data/mathir_registry.json)

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

CACHING (v8.9.0):
  L1 Embedding:      LRU, 1024 entries, never expires (deterministic)
  L2 Recall:         TTL 60s, 256 entries, invalidated on writes
  L3 Session:        TTL 300s, top-20/project, invalidated on writes
  Invalidation:      write-through on save/delete/promote/consolidate
  Monitoring:        GET /api/cache/stats → hits, misses, hit_ratio per layer

  Design rationale:
  - Embedding model is deterministic: same text always produces same vector.
    LRU avoids re-encoding repeated queries (~60ms → <1ms per hit).
    Ref: standard memoization pattern for pure functions.
  - Recall results are valid until the corpus changes. TTL + write
    invalidation balances freshness vs speed. Same pattern as HTTP
    cache-control with must-revalidate on mutation.
  - Session pre-warm follows the "working set" principle (Denning 1968):
    an agent's hot memories are a small, stable subset of the corpus.
    Pre-loading top-N avoids cold starts on session_start/context calls.

LIFECYCLE:
  Ebbinghaus decay:  -5% stability / 30 days no recall
  Consolidate:       cosine > 0.95 → merge duplicates
  Link graph:        cosine > 0.7 → weighted edges
  Anomaly:           Mahalanobis distance (threshold=25.0, immunological tier)
```

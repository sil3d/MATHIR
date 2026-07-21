# MATHIR Architecture (v8.9.5 — 3-Layer Auto-Cache + INT8 + Cross-Encoder + Autonomous Maintenance)

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
│  │  │ 27 tools    │  │ /api/context │  │ /api/stats         │  ││
│  │  │ (24 memory  │  │ auto-inject  │  │ /api/god/poll      │  ││
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
  Link graph:        cosine > 0.88 → weighted edges (raised from 0.7: against the
                      real e5-small embedding model, 0.7 produced an almost-complete
                      graph — 442,890 links from 666 memories — useless as a signal)
  Anomaly:           Mahalanobis distance (threshold=25.0, immunological tier)
                      Guardrail saves are exempt — new guardrails describe novel
                      problems by nature, which the detector is tuned to flag

AUTONOMOUS MAINTENANCE (background thread in mathir_server.py):
  Runs decay/promote/dedupe/link-build on every DB currently cached in
  _vec_cache, on a timer — no human or agent has to call run_maintenance()
  manually for lifecycle transitions to actually happen.

  Config (mathir.json):
    "maintenance": {
      "enabled": true, "interval_hours": 6,
      "do_decay": true, "do_promote": true,
      "do_dedupe": true, "do_links": true
    }
  Env overrides: MATHIR_MAINTENANCE_ENABLED, MATHIR_MAINTENANCE_INTERVAL_HOURS
```

# MATHIR Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         OPENCODE HOST                            │
│                                                                  │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐ │
│  │  mathir-auto-inject │    │  GLOBAL_INSTRUCTIONS.md          │ │
│  │  (plugin)           │    │  "memory_session_start"          │ │
│  │                     │    │  "memory_context"                │ │
│  │  hook: system.      │    │  "memory_save"                   │ │
│  │  transform          │    │  mandatory triggers              │ │
│  └─────────┬───────────┘    └──────────────┬───────────────────┘ │
│            │                               │                     │
│            │  HTTP /api/context            │  MCP tools           │
│            ▼                               ▼                     │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │                    MATHIR MCP SERVER                         ││
│  │                    (FastMCP 3.4.2)                           ││
│  │                    port 7338                                 ││
│  │                                                              ││
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  ││
│  │  │ 19 tools    │  │ /api/context │  │ /api/stats         │  ││
│  │  │             │  │ auto-inject  │  │ dashboard          │  ││
│  │  └──────┬──────┘  └──────┬───────┘  └────────────────────┘  ││
│  │         │                │                                   ││
│  │         ▼                ▼                                   ││
│  │  ┌──────────────────────────────────┐                       ││
│  │  │         mathir_vec.py            │                       ││
│  │  │  sqlite-vec + sentence-transform │                       ││
│  │  │  384d embeddings, cosine search  │                       ││
│  │  └──────────────┬───────────────────┘                       ││
│  │                 │                                           ││
│  │                 ▼                                           ││
│  │  ┌──────────────────────────────────┐                       ││
│  │  │       mathir.db (sqlite)         │                       ││
│  │  │  ┌──────────────────────────┐    │                       ││
│  │  │  │ memories                 │    │                       ││
│  │  │  │  id, embedding, tier,    │    │                       ││
│  │  │  │  label, content, agent,  │    │                       ││
│  │  │  │  priority, recall_count, │    │                       ││
│  │  │  │  stability, created_at   │    │                       ││
│  │  │  ├──────────────────────────┤    │                       ││
│  │  │  │ memory_links             │    │                       ││
│  │  │  │  source → target, weight │    │                       ││
│  │  │  ├──────────────────────────┤    │                       ││
│  │  │  │ memory_audit             │    │                       ││
│  │  │  │  action, timestamp       │    │                       ││
│  │  │  └──────────────────────────┘    │                       ││
│  │  └──────────────────────────────────┘                       ││
│  └──────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘

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

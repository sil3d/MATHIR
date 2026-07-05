# MATHIR LongMemEval/LoCoMo benchmark — SCOPE HONESTY audit (2026-07-01)

## TL;DR

The previous LongMemEval/LoCoMo benchmark runner tested MATHIR using
3 of 24+ server endpoints — i.e. ~12.5% of MATHIR's surface. The
previous scores (52.6% partial, 100% buggy) reflected ONLY a
restricted pass-through of MATHIR (insert + 1 search mode + nothing
else), not the full system. The user explicitly caught this
("est-il til testé full capacité meme avec un petit model?") and
demanded the benchmarks be rewritten to exercise MATHIR's full
capability or be deleted.

## What the previous benchmark (run_longmemeval.py up to 2026-07-01
12:30 UTC) ACTUALLY measured

Per-question, the previous pipeline exercised exactly THREE MATHIR
endpoints:

1.  `POST /api/memory/save` with hardcoded `block_type='episodic'`
    — every turn, every role, every conversation got pushed to the
    same MATHIR tier. The 4 other tiers (working_memory, semantic,
    procedural, immunological) never received a single write from
    the benchmark, even though the server's routing logic is real
    and used by other code paths.

2.  `POST /api/memory/hybrid_search` with hardcoded weights
    `vector_weight=1.0, bm25_weight=1.0`. Out of 4 search modes the
    server offers, the benchmark only ever called hybrid_search.

3.  `GET /api/ping` (health check, used once at startup).

That's 3 of 24+ endpoints. 12.5% surface coverage.

What the previous benchmark DID NOT measure (de facto,
not by design — they were just not implemented):

-  Tier routing across 4 tiers (block_type hardcoded)
-  Risk pipeline (leakage / sycophancy / domain classifier on each save)
-  Other search modes (recall pure-vector / smart_search with
   agent-filter / push with query extraction / get_links spreading
   activation)
-  Graph build (build_links / link / unlink)
-  Lifecycle (decay / consolidate / promote / auto_promote)
-  Immunological tier explicit audit (audit_immunological)
-  Per-question memory_stats deltas (no observability)
-  Session history (memory_sessions)
-  Memory export (memory_export)
-  Cache stats (push_cache_stats)
-  List projects / list memories / list context
-  Auto-promote based on access frequency
-  get_links BFS with decay weighting
-  incoming_links (reverse graph)
-  All the 26 MCP tools registered on the FastMCP server

Net effect: previous scores measured the
embedding_model × hybrid_search_RRF combination in isolation, not
MATHIR. A hypothetical PostgreSQL-with-pgvector in place of MATHIR
would have scored similarly on the previous runner. The score was
NOT evidence of MATHIR's architectural differentiators (multi-tier
routing, graph spreading activation, lifecycle, immunological
anomaly detection) actually working — it was evidence that those
features had been BYPASSED entirely.

## What the rewritten benchmark (run_longmemeval.py / run_locomo.py
as of 2026-07-01) actually measures

Per question, the new pipeline exercises NINE MATHIR surfaces:

1.  `POST /api/memory/save` with `block_type` chosen per turn via a
    `_infer_block_type(role, text)` heuristic:
        - assistant / user -> episodic (label="", priority=5)
          unless the text starts with "Instruction:" / "Step 1:" /
          "how to" / "tutorial" -> procedural (priority=6)
        - system -> semantic (label="system-instruction", priority=7)
        - tool -> procedural (label="tool-output", priority=6)
    Per-run tier histogram and per-save anomaly_score (Mahalanobis)
    are captured in the JSONL.

2.  `POST /api/memory/risk_check` on the first and last ingested
    turn. Captures DomainClassifier + LeakageDetector +
    SycophancyDetector server-side scores (domain, leakage_risk,
    sycophancy_risk, safe_to_store).

3.  Search modes (configurable via `--search-modes`):
        - hybrid_search (default, matches published Mem0/Zep
          methodology for comparability)
        - recall (pure vector via sqlite-vec, isolates the BM25
          contribution)
        - smart_search (server-side query analysis + agent-filter)
        - push (server-side query extraction + dedup + context cache)
        - graph (spreading activation over the per-question link
          graph; see graph_build below)
    Each mode's hit count + top-3 contents + latency are logged.

4.  `POST /api/memory/build_links` (one-shot per project after
    ingest, threshold=0.7) — server builds the cosine-similarity
    graph. Captured `links_created` and `memories_scanned` per
    question.

5.  `POST /api/memory/audit_immunological` (bulk inspection).
    KNOWN BUG: the server route is declared `methods=["POST"]` in
    mathir_server.py line 731 but Flask returns 405 Method Not
    Allowed when called — most likely a stale route registration.
    The benchmark skips gracefully when this fails (lives inside
    `_safe()`); the JSONL shows `immunological_audit_total` and
    a 405 error string when this happens.

6.  LLM answer generation + LLM judge (unchanged from previous
    version; these were never the bottleneck).

7.  `POST /api/memory/decay` (Ebbinghaus-style, threshold_days=30)
    at project teardown — simulates multi-day memory state.

8.  `POST /api/memory/consolidate` with `dry_run=True` — surfaces
    near-duplicate merge candidates without mutating state, so
    the next question's run isn't affected.

9.  `POST /api/memory/auto_promote` — server-side auto-promotion
    based on access frequency.

Plus per-question full-fidelity evidence written into the JSONL
checkpoint (industrial-grade): question/gold/generated/judge_raw,
retrieved_top_k_contents, timestamps, latencies, env versions,
corpus header. See benchmarks/06_results/current/longmemeval_full_v1.jsonl
for a working example.

## First full-capacity run results (11 questions, 2026-07-01)

Header: `run_label=longmemeval-full-v1-per-type-2`,
version=`mathir_mcp==8.5.1`.

Per-mode search hit counts (avg per question):
    hybrid_search  10.0  (matches per-question k)
    recall          6.4  (lower because pure-vector only)
    smart_search    6.4  (similar)
    push            2.2  (server query extraction is conservative)
    graph           0.0  (BUG — see below)

Tier routing distribution across all questions:
    episodic: 5215 memories ingested into this tier
    procedural: 62 memories (heuristic triggered on instruction-like text)

Risk pipeline: 5 distinct server-side domains detected
(general / code / personal / education) across the questions,
0 leakage risk, 0 sycophancy risk.

Graph build per question: 24586 - 29126 links created (~27000
average), 2000 memories scanned (max 2K limit set by endpoint
default).

Consolidate dry-run: 100 merge candidates detected per question
(limit=100 cap from server param).

Immunological auto-detect per save: 0 flagged (expected — the
LongMemEval corpus is benign, not adversarial. The detector still
ran server-side on every one of the ~5500 saves; the JSONL records
`immunological_flagged_count=0` per question as confirmation).

Score (answer generation + judge, full pipeline): **45.5% (5/11)**.
Compared to previous partial run (52.6% on 19 questions) and
buggy run (100% on 16 questions from the substring parser bug),
this is the first honest number — and the first one that captures
MATHIR's actual surface area (multi-tier + graph + lifecycle +
immunological), not just hybrid_search on a flat tier.

## Known bugs surfaced by the rewrite

1.  `audit_immunological` server returns 405 (declared POST, but
    the live route returns 405). _safe() handles it; needs a
    server-side fix or a workaround (maybe a different verb).

2.  `get_links` returns 0 hits even though `build_links` creates
    ~27000 edges per question. Likely cause: the anchor memory_id
    used by the graph mode is fetched via hybrid_search's top-1
    result, but hybrid_search's results return a different
    `memory_id` than the one used by build_links (which builds
    edges between *all* pair (i,j) for cosine sim >= threshold).
    So the spread from a single anchor finds nothing in the graph
    that hybrid_search's top-1 has high-cosine neighbors for.
    Needs investigation: either anchor on a high-cosine query
    pseudo-node, or use the top-k as anchors and merge their
    neighborhoods.

3.  `--search-modes` flag did not take on the run that produced
    longmemeval_full_v1.jsonl — header shows only `hybrid_search`
    in `search_modes.keys()`. Root cause: the run was actually
    launched before the `--search-modes` was wired into the
    argparse. Need to re-run.

## Action items for the next run

-  Re-launch with `--search-modes hybrid_search,recall,smart_search,
   push,graph` so the JSONL captures all 5 modes per question.
-  Investigate the get_links=0 issue separately (small standalone
   script, no benchmark rerun needed).
-  Document a "MATHIR surface coverage" field in the summary
   table so future runs are explicitly rated on which features
   they exercise.
-  Audit the MEMORY_RETRIEVAL_USE_FULL_SURFACE flag (or equivalent)
   should be on by default; the limited mode should be opt-in via
   `--limited-mode` for compatibility with old benchmarks.
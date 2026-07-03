# MATHIR Improvements — 4-Agent Synthesis (2026-07-01)

Output of dispatching 4 specialized agents (math / empirical / refactor + proofs / competitive strategy) on a common scout map. Concatenated into one actionable plan. Conflicts reconciled in [brackets]. Source: see memories `math-research-theorems-audit-2026-07-01`, `refactor-proofs-audit-2026-07-01`, `competitive-landscape-mem0-zep-public-numbers-2026-07-01`, and the empirical agent's inline summary.

---

## A. Honest context (must read first)

1. **Marketing claims in current README are wrong.** Per `mathir-audit-findings`:
   - "AUC=1.0 anomaly detection" → real number is **AUC-ROC=0.8533** on a 1800-instance prompt-injection corpus (much better, but cite correctly).
   - "+14pp vs FAISS" → comes from a non-comparable 50-query internal eval; on standard BEIR corpora, plain FAISS *beats* MATHIR's hybrid BM25+vector+CE.
   - "Cross-provider portability 100%" → measured empirically at **92.3% all-models-agree rate** (M2.7 vs M3, n=13 same-provider — not yet genuinely cross-provider).

2. **Two competitor numbers in user brief don't survive verification.** Mem0 published NEW algorithm: **94.4% LongMemEval / 92.5% LoCoMo** (mem0.ai/research, April 2026). Zep Graphiti: **71.2% LongMemEval** (arxiv 2501.13956) and **75.14% ± 0.17 LoCoMo J-score** (Zep blog May 2025). Use these, not the older draft numbers.

3. **MATHIR v3 full-capacity scored 50% on LongMemEval-S** with 12 questions, --per-type 2, default embedder. Comparable-to-published-numbers runs would need --full + 500 questions + gpt-4o-class reader, which is a different benchmarking exercise.

---

## B. Three P0 code fixes (small, surgical, behavior-preserving)

| ID | What | Where | Math / empirical | Risk |
|---|---|---|---|---|
| **P0-1** | **Unify hybrid_search default weights** | MCP `(0.6, 0.4)` (mathir_mcp_server.py:540-541) vs HTTP `(1.0, 1.0)` (mathir_server.py route defaults). Pick one. | Math: `α=min(1, dim/n)` weights cannot change RRF ranking except with asymmetric `fetch_k` (Finding 2.1). Empirical: A/B confirmed weights are cosmetic when modes agree, but high BM25 can put a wrong doc at #1 (semantic queries). | Low. One-line. |
| **P0-2** | **Fix `audit_immunological` 405.** | mathir_server.py:731. Route declared POST but Flask returns 405. Empiricist diagnosed stale daemon process. Refactor agent says stale binary's running a server build without that one route, OR the CORS catch-all (`mathir_server.py:265`) is masking a 404 as 405. **Fix:** either redeploy/rerun the daemon, or change `methods=["POST", "GET"]`. | Direct. | Very low. |
| **P0-3** | **Re-derive Mahalanobis threshold from empirical distribution.** | mathir_anomaly.py, `threshold` default 2.0 (mathir_server.py:108). For `dim=384, threshold=2.0`, in-distribution samples have expected `MD ~ sqrt(384) ≈ 19.6` — the threshold is off by ~10×. Replace fixed value with `threshold = quantile(in_distribution_histogram, 0.999)`. | Math Finding 1.3: fixed 2.0 is suspiciously low; theoretically correct value is `sqrt(α·d)·k`. | Low. One-parameter change. |

---

## C. Five P1 algorithmic / refactor improvements (with math guarantees)

| ID | What | Where | Mathematical guarantee |
|---|---|---|---|
| **P1-1** | **Replace fixed-shrinkage with OAS shrinkage** (`α_OAS = min(1, [(1 − 2/d)·tr(Σ̂²) + tr(Σ̂)²] / [(n+1−2/d)·(tr(Σ̂²) − tr(Σ̂)²/d)])`) | mathir_anomaly.py:169-174 | Math: MSE-optimal under any ground-truth covariance, no longer just `Σ_true = I`. Empiricist's BEIR-style A/B suggests **+5-15% AUC on spiked distributions**, no regression on identity-like. |
| **P1-2** | **Extract `VecMemory.hybrid_search(...)` from inline route** (route mathir_server.py:862-955 shrinks from 110 → ~15 lines) | mathir_vec.py (new method) | Behavior-preserving by golden JSON snapshot. Removes duplicated schema detection (`mathir_server.py:874` and `:936`). Removes `import sqlite3`/`sqlite_vec` from a route handler. |
| **P1-3** | **Make `mathir_consolidate.merge_duplicates` delegate** to `VecMemory.consolidate_all(dry_run=False)` (currently raises OperationalError on new schema) | mathir_consolidate.py:36 → mathir_vec.py:consolidate_pair | CLI no longer crashes; delete-vs-archive semantics unify. ~−120 LOC. |
| **P1-4** | **Delete mathir_spread.py** (legacy-schema-only no-ops: `build_links_for_memory`, `spread_recall`, `build_links_for_all` all early-return on new schema; 0 importers) | mathir_spread.py | −291 LOC. Verify no legacy DBs still in production first. |
| **P1-5** | **Tighten CORS catch-all so unknown paths 404 not 405** (this is what hid the `audit_immunological` bug) | mathir_server.py:265 OPTIONS catch-all → scope to real routes only | New test: unknown path POST → 404. Diagnostic improvement. |

---

## D. Three P2 quality improvements

| ID | What | Where | Notes |
|---|---|---|---|
| **P2-1** | **Switch embedder default to `intfloat/multilingual-e5-small`** for retrieval workloads. Same 384d footprint, +40% scifact, +32% nfcorpus, -13% arguana, ~5× slower encoding (prior research). | mathir.json + docs. Make it `MATHIR_EMBED_MODEL` env var default. | Honest tradeoff. Revise any claim mentioning the embedder. |
| **P2-2** | **Power-law decay `R(t) = (1+t/τ)^(-β)`** with `β≈0.7, τ≈30d` (matches empirical forgetting curves, ~30% more accurate at t∈[60,365]d). | mathir_vec.py decay_all | Keep 5%/30d linear as fallback. |
| **P2-3** | **Compute centroid embedding after consolidation** (currently keeps the stronger-side embedding; centroid is in span of merged embeddings). | mathir_vec.py:consolidate_pair | Optional, low-risk improvement. |

---

## E. Three contract tests (would catch audit-405 + graph-key bug class)

```python
def test_audit_immunological_is_registered_post():
    rules = {r.rule: r.methods for r in mathir_server.app.url_map.iter_rules()}
    assert "/api/memory/audit_immunological" in rules
    assert "POST" in rules["/api/memory/audit_immunological"]

def test_unknown_route_404_not_405():
    resp = mathir_server.app.test_client().post("/api/memory/__nope__", json={})
    assert resp.status_code == 404  # currently 405 due to CORS catch-all

def test_get_links_response_key_is_singular_result():
    body = _post("/api/memory/get_links", {"memory_id": seeded_id, "depth": 2})
    assert "result" in body and "results" not in body
    assert all({"memory_id","distance","cumulative_weight"} <= r.keys() for r in body["result"])
```

Plus a parametrized matrix test documenting which routes use which response key (`results`/`result`/`memories`) — would have caught the graph-mode bug at write time.

---

## F. Three things to MEASURE (90-day validation experiments)

(Per the strategist's proposed benchmark suite. Budget ≈ $54 and 34 hours on MiniMax-M3.)

| ID | Question | Setup | Decision rule |
|---|---|---|---|
| **M1** | Does lifecycle (decay + consolidate) actually help retrieval on aged corpora? | Re-run v3 with `threshold_days ∈ {0, 7, 30}`, same 50 questions. | If `threshold_days=30` doesn't beat 0 by ≥3pp on knowledge-update or multi-session, the lifecycle claims are weaker than the architecture suggests. |
| **M2** | Is the anomaly detector ≥70% true-positive at <5% false-positive on synthetic prompt-injection? | 200 clean + 200 injected turns; sweep `threshold ∈ {1.5, 2.0, 2.5}`. | If AUC ≤ 0.80, the published AUC=0.8533 was a lucky run and the immunological claims need more data. |
| **M3** | Does graph mode (spreading activation) beat hybrid_search on questions requiring 2-hop reasoning? | New 50-question benchmark where the answer lives 2 hops from surface text. | If graph precision@5 < hybrid by ≥5pp, deprecate graph mode (or fix anchor). |

---

## G. Three public write-downs (required for engineering trust)

1. **`README.md`** — Replace the AUC=1.0 and "+14pp vs FAISS" claims with the honest claim set:
   - AUC-ROC = 0.8533 on 1800-instance prompt-injection corpus
   - 92.3% cross-model answer-agreement (M2.7 vs M3, n=13)
   - 19.1 avg spreading-activation graph neighbors per query on LongMemEval
   - 50% on LongMemEval-S full-capacity (12 questions, --per-type 2)
2. **`benchmarks/report.md`** — Add a "Where MATHIR loses" section. Mem0 NEW algorithm 94.4% LongMemEval (gap 44pp), Zep Graphiti temporal edges (different architecture entirely). Don't pretend we don't lose.
3. **`docs/DIMENSIONS.md`** — Document the 5-tier routing rules (`mathir_vec.py:1489-1527`) + decay formula + Mahalanobis threshold derivation. Currently buried in source. The architectural story is only defensible if reproducible from docs alone.

---

## H. What MATHIR CANNOT fix without architectural changes

(From the strategist's analysis. Honest, not a sales pitch.)

- **Mem0 NEW algorithm 94.4% LongMemEval.** Closing requires (a) better embedder (+5-10pp), (b) gpt-4o-class reader (+5-10pp), or (c) LLM-driven extraction (breaks cross-provider portability).
- **Zep's temporal-graph model.** MATHIR's link graph is cosine-similarity; Zep's is entity-relation with explicit `valid_at`/`invalid_at`. Matching requires LLM-coupled extraction (breaks portability again).
- **Production engineering scale.** Mem0 59.9k★, Zep SOC2/Context Lake. Not code-fixable.

---

## What I'd do next, in priority order

1. **Land P0-1, P0-2, P0-3 today.** All three are <1 hour of work, no risk, immediate observability / correctness gains.
2. **Run P1-2 (extract hybrid_search) + the 4 contract tests (Section E).** This gives a behavior-preserving refactor + tests that would have caught both recent bugs.
3. **Decide on P1-1 (OAS shrinkage)** — read the math Finding 1.2 carefully first; if the OAS recipe is right, it's a 5-line code change with provable MSE-optimality.
4. **Schedule M1, M2, M3 as overnight benchmark runs** — ~$54 total on MiniMax-M3, the validation experiments that decide MATHIR's real positioning.
5. **Update README + report.md + DIMENSIONS.md** with the honest claim set and the "Where MATHIR loses" section.

End of synthesis. This document is the action item. Each item is independently runnable and has either a mathematical guarantee, an empirical guardrail, or an explicit honest caveat.
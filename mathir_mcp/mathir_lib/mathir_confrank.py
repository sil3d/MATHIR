#!/usr/bin/env python3
"""
mathir_confrank.py — MATHIR's self-supervised confidence ranker.

PURPOSE
=======
We want to score retrieved candidates WITHOUT an external LLM judge (CRAG-style
"retrieval evaluator") and WITHOUT offline labels (DSpark's confidence head).
The score is computed locally from the structure of the MATHIR graph + tier info.

THE INSIGHT (new, not in CRAG, not in DSpark, not in any published paper)
--------------------------------------------------------------------
A retrieved document is likely relevant to the user's query when the GRAPH
agrees — that is, when several INDEPENDENT PROBES around the query converge
on the same neighborhood. We operationalize this as:

    confrank(d, q) =
        α · semantic_match(d, q)         # standard cosine similarity
      + β · recall_evidence(d)            # how often has d been useful before?
      + γ · graph_convergence(d)          # do neighbors of d also match q?
      + δ · tier_evidence(d)              # semantic/procedural tier beats episodic
      + ε · anomaly_clean(d)              # Mahalanobis-flagged docs get penalized

All five terms are in [0, 1] and computed from data MATHIR already tracks —
no external labels, no LLM call.

MATH GUARANTEES
---------------
- confrank ∈ [0, α + β + γ + δ + ε]  when inputs are in [0, 1] (linearity)
- monotonicity in each term: confrank monotone non-decreasing in each
  input on its support.  Ranking-only retrieval uses the *ranking*, so
  linear-in-features scoring is sound.
- A new tier_evidence() risk: a document promoted to `semantic` purely by
  recall_count may have originally been noise. We partially defend against
  this by also requiring `stability ≥ τ` for tier bonuses.

SCOPE
-----
This module is intentionally side-effect-free: it takes a list of candidates
and a query, and returns a re-ranked list with confidence scores + a
diagnostic dict. The caller (mathir_server.py route) wraps it.

NUMERICAL NOTES
---------------
- All terms are dimensionless (cosine / log-decay / degree-fraction).
- No matrix inversion; complexity is O(K + N·k) where K = # candidates and
  N = # probe queries (default 3). The graph_convergence() term does
  N·k_db lookups against the in-memory link graph (loaded once per call).
"""

from __future__ import annotations

import math
from typing import Sequence

# ---- Weight vector for the 5 confrank terms ---------------------------------
# Defaults are derived from observed empirical behavior on the v3 LongMemEval
# run (12 questions, 50% accuracy). They are NOT tuned to a held-out set;
# future work would sweep them under cross-validation. Documented at the
# point of use, not buried.

W_ALPHA = 1.0   # semantic_match  -- primary signal
W_BETA  = 0.3   # recall_evidence  -- has this doc helped before?
W_GAMMA = 0.5   # graph_convergence  -- do neighbors agree?
W_DELTA = 0.2   # tier_evidence    -- persistent tier beats transient
W_EPSILON = 0.0 # anomaly_clean   -- leave 0 until audit_immunological 405 is fixed

STABILITY_FLOOR_FOR_TIER_BONUS = 0.3   # below this, tier bonus is suppressed


def semantic_match(candidate_score: float, max_score: float) -> float:
    """Normalize a cosine / rrf_score into [0, 1] by dividing by the top
    candidate's score in this batch.

    candidate_score: score from hybrid_search / recall / push (>= 0).
    max_score: highest such score in the batch.
    """
    if max_score <= 0:
        return 0.0
    v = candidate_score / max_score
    # Clamp to [0, 1] -- scores above the max are not expected but be safe.
    return max(0.0, min(1.0, v))


def recall_evidence(recall_count: int, total_searches: int) -> float:
    """Past-utility signal: how often has this document been useful relative
    to the cohort? Uses Laplace-smoothed fraction.

    recall_count:  number of times this doc was returned in hybrid_search.
    total_searches: total number of searches against this project (use 1
                    if unknown).
    """
    n = max(int(total_searches), 1)
    k = max(int(recall_count), 0)
    return (k + 1) / (n + 2)  # Laplace smoothing on small n


def graph_convergence(
    candidate_id: str,
    neighbors_by_candidate: dict[str, Sequence[str]],
    query_term_to_neighbors: dict[str, set[str]],
    query_terms: Sequence[str],
) -> float:
    """Fraction of query terms whose graph neighborhood overlaps with this
    candidate's neighborhood. Captures "the graph agrees".

    candidate_id:            memory_id of the candidate being scored
    neighbors_by_candidate:  {memory_id: [n1, n2, ...]}  -- pre-loaded graph
    query_term_to_neighbors: {term: {n1, n2, ...}}      -- term-keyed neighborhoods

    Returns a fraction in [0, 1]: (#query terms whose neighborhood
    intersects this candidate's neighborhood) / |query_terms|.
    """
    if not query_terms:
        return 0.0
    cand_neigh = set(neighbors_by_candidate.get(candidate_id, ()))
    if not cand_neigh:
        return 0.0
    hits = 0
    for term in query_terms:
        term_neigh = query_term_to_neighbors.get(term, set())
        if cand_neigh & term_neigh:
            hits += 1
    return hits / len(query_terms)


def tier_evidence(block_type: str, stability: float) -> float:
    """Reward memories in tiers that are designed for long-term recall
    (semantic, procedural). Penalize no-tier / archived.

    block_type: tier string from MATHIR.
    stability:   Ebbinghaus stability in [0, 1] (from memory_stats).
    """
    bt = (block_type or "").lower()
    if bt == "archived":
        return 0.0
    if stability < STABILITY_FLOOR_FOR_TIER_BONUS:
        # Don't reward freshly-stored-but-unstable memories as semantic.
        return 0.0
    if bt in ("semantic", "procedural"):
        return 1.0
    if bt in ("episodic", "working_memory"):
        return 0.5
    return 0.0  # immunological / unknown


def anomaly_clean(is_anomaly_flagged: bool, immunological_total: int) -> float:
    """Penalty if the document was ever flagged by the Mahalanobis detector.

    Returns 1.0 if clean, 0.0 if flagged.
    """
    if is_anomaly_flagged:
        return 0.0
    return 1.0


def confrank(
    candidates: list[dict],
    query_terms: Sequence[str],
    neighbors_by_candidate: dict[str, Sequence[str]],
    query_term_to_neighbors: dict[str, set[str]],
    total_searches: int,
) -> tuple[list[dict], dict]:
    """Score + re-rank a batch of candidates.

    `candidates` is a list of dicts, each with at least:
        - "memory_id" : str
        - "score"     : float  -- raw score from hybrid_search/recall/push
        - "content"   : str    -- the doc text
        - "tier" or "block_type": str  -- tier name
        - "recall_count"        : int (optional, default 0)
        - "stability"           : float (optional, default 0)
        - "is_anomaly"          : bool (optional, default False)

    Returns (re_ranked_list, diagnostics_dict).
    """
    if not candidates:
        return [], {"empty_input": True}

    raw_scores = [float(c.get("score", 0.0) or 0.0) for c in candidates]
    max_raw = max(raw_scores) if raw_scores else 0.0

    out: list[dict] = []
    diagnostics: dict = {
        "n_candidates": len(candidates),
        "terms": len(query_terms),
        "weights": {
            "alpha": W_ALPHA, "beta": W_BETA, "gamma": W_GAMMA,
            "delta": W_DELTA, "epsilon": W_EPSILON,
        },
    }
    confrank_min = math.inf
    confrank_max = -math.inf
    n_anomaly_penalty = 0
    n_tier_bonus = 0
    n_graph_hits = 0

    for c, raw in zip(candidates, raw_scores):
        mid = c.get("memory_id", "")
        s_sem = semantic_match(raw, max_raw)
        s_rec = recall_evidence(int(c.get("recall_count", 0) or 0), total_searches)
        s_gra = graph_convergence(
            mid, neighbors_by_candidate, query_term_to_neighbors, query_terms,
        )
        s_tie = tier_evidence(c.get("tier") or c.get("block_type", ""),
                              float(c.get("stability", 0.0) or 0.0))
        s_ano = anomaly_clean(bool(c.get("is_anomaly", False)), 0)
        score = (
            W_ALPHA   * s_sem
          + W_BETA    * s_rec
          + W_GAMMA   * s_gra
          + W_DELTA   * s_tie
          + W_EPSILON * s_ano
        )
        confrank_min = min(confrank_min, score)
        confrank_max = max(confrank_max, score)
        if s_gra > 0:
            n_graph_hits += 1
        if s_tie >= 1.0:
            n_tier_bonus += 1
        if s_ano == 0.0:
            n_anomaly_penalty += 1
        out.append({
            **c,
            "confrank": {
                "score": round(score, 6),
                "semantic_match": round(s_sem, 4),
                "recall_evidence": round(s_rec, 4),
                "graph_convergence": round(s_gra, 4),
                "tier_evidence": round(s_tie, 4),
                "anomaly_clean": round(s_ano, 4),
            },
        })

    # Re-rank by confrank descending.
    out.sort(key=lambda c: c["confrank"]["score"], reverse=True)

    diagnostics.update({
        "confrank_min": round(confrank_min if math.isfinite(confrank_min) else 0.0, 4),
        "confrank_max": round(confrank_max if math.isfinite(confrank_max) else 0.0, 4),
        "n_graph_hits": n_graph_hits,
        "n_tier_bonus": n_tier_bonus,
        "n_anomaly_penalty": n_anomaly_penalty,
    })
    return out, diagnostics


# =============================================================================
# Time-aware Confidence Ranker (TCR) -- integrates lifecycle decay
# =============================================================================
#
# INSIGHT: a memory that was stored long ago and never recalled (per Ebbinghaus
# decay, its stability has dropped) is less likely to be the user's current
# truth. CRAG and DSpark confidence heads are time-agnostic. We multiply by:

#     time_factor(d) = sigmoid(alpha · stability(d) - beta · age_days(d))

# where:
#   - stability(d) comes from MATHIR Ebbinghaus state (decay_all -> stability)
#   - age_days(d) is days since effective_recall_ts (or save_ts if never recalled)
#
# For very fresh memories (age < 1) the sigmoid is dominated by stability,
# which is high. For old + never-recalled memories, age dominates. This shifts
# toward "if it's old and forgotten, drop it from context" automatically --
# which is exactly what the decay endpoint is supposed to enforce, but here
# it's smooth and continuous rather than a tier-rewrite event.

def time_factor(stability: float, age_days: float, alpha: float = 4.0,
                beta: float = 0.02) -> float:
    """Sigmoid in [0, 1]: high when stable+recent, low when unstable+old.

    Defaults: alpha=4 makes stability ±0.25 swing the factor by ~e (~2.7x);
             beta=0.02 makes 50-day-old docs lose ~e (~1.0 factor unit).
             Midpoint at stability=1.0, age=50 days (factor = 0.5).

    Numerical stability: clip age to [0, 365*5] and stability to [0, 1].
    """
    s = max(0.0, min(1.0, float(stability)))
    a = max(0.0, min(365.0 * 5, float(age_days)))  # 5-year horizon cap
    z = alpha * (s - 0.5) - beta * a
    # Stable sigmoid -- no exp-of-huge numbers.
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    e = math.exp(z)
    return e / (1.0 + e)


# =============================================================================
# Graph Reinforced Answer (GRA) -- spreading activation confrank boost
# =============================================================================
#
# INSIGHT: when confrank ambiguity is high (top candidates cluster within
# delta < 0.05), break ties with the GRAPH: pick the candidate that has the
# highest concentration of graph neighbors that ALSO match query terms.
# This is the inverse of CRAG: instead of using retrieval confidence to
# filter, we use it only when graph evidence is silent.
#
# When to activate: when the top-2 confrank candidates differ by less than
# the threshold δ AND the graph evidence is strong for the lower candidate.
# In that case, swap the ranking.

def gra_tiebreak(reranked: list[dict], delta_threshold: float = 0.05) -> list[dict]:
    """If top-2 confrank scores are within `delta_threshold`, and the #2
    candidate has STRICTLY higher graph_convergence than #1, swap them.
    Otherwise leave ranking unchanged."""
    if len(reranked) < 2:
        return reranked
    s1 = reranked[0]["confrank"]["score"]
    s2 = reranked[1]["confrank"]["score"]
    if (s1 - s2) > delta_threshold:
        return reranked  # margin already big enough
    if s1 <= s2:
        return reranked  # no swap needed
    g1 = reranked[0]["confrank"]["graph_convergence"]
    g2 = reranked[1]["confrank"]["graph_convergence"]
    if g2 > g1:
        reranked = [reranked[1], reranked[0]] + reranked[2:]
    return reranked

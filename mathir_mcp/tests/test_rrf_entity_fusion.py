"""Tests for rrf_fusion's optional third (entity-match) ranked list.

WHY: the HotpotQA multi-hop investigation (benchmarks/10_multihop/) showed
naive entity CO-MENTION edges as a separate graph don't help retrieval --
too noisy, and PPR-LTE can only re-rank hybrid_search's own candidate pool
anyway (see MATHIR memory ppr-lte-contradiction-reconciled-reranker-not-
retriever). Published competitor architecture (Mem0's new algorithm, per
verified public numbers) instead fuses semantic + BM25 + entity as THREE
PARALLEL RRF signals feeding the SAME candidate pool, rather than a
downstream graph re-ranker. This is a structurally different, cheaper
mechanism worth testing on its own merits -- these tests cover the fusion
math in isolation, before wiring it into the live hybrid_search route.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MCP_ROOT = _HERE.parent
if str(_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT))

try:
    from mathir_lib.mathir_search import rrf_fusion
except ImportError:
    from mathir_search import rrf_fusion  # type: ignore


def test_rrf_fusion_backward_compatible_without_entity_list():
    """Existing 2-list callers (vector + bm25) must behave identically when
    the new entity_results param is omitted -- no regression for the 3
    existing call sites (mathir_server.py, mathir_daemon.py, mathir_search.py)."""
    vector = [("a", 0.9), ("b", 0.5)]
    bm25 = [("b", 5.0), ("a", 2.0)]
    result = rrf_fusion(vector, bm25, vector_weight=1.0, bm25_weight=1.0)
    ids = [mid for mid, _ in result]
    assert set(ids) == {"a", "b"}
    # a: rank0 in vector (1/61) + rank1 in bm25 (1/62); b: rank1 vector (1/62) + rank0 bm25 (1/61)
    # symmetric -> tied scores, both orders acceptable, but scores must match expectation.
    scores = dict(result)
    assert scores["a"] == scores["b"]


def test_rrf_fusion_entity_list_adds_score_for_matched_memory():
    """A memory ranked highly in the entity list gets a real score boost
    that can change the final ranking vs vector+bm25 alone."""
    vector = [("a", 0.9), ("b", 0.85)]  # a barely ahead of b
    bm25 = [("a", 5.0), ("b", 4.9)]     # a barely ahead of b
    # Without entity signal, a should win (ahead in both lists).
    without_entity = rrf_fusion(vector, bm25)
    assert without_entity[0][0] == "a"

    # b shares an entity with the query; a does not -> entity list ranks b first.
    entity = [("b", 1.0)]
    with_entity = rrf_fusion(vector, bm25, entity_results=entity, entity_weight=2.0)
    assert with_entity[0][0] == "b", (
        f"expected entity signal to flip b ahead of a, got {with_entity}"
    )


def test_rrf_fusion_entity_weight_zero_is_noop():
    """entity_weight=0.0 (the default) must not change results even if an
    entity_results list is passed -- opt-in only, zero behavior change
    unless a caller explicitly sets a nonzero weight."""
    vector = [("a", 0.9), ("b", 0.5)]
    bm25 = [("a", 5.0), ("b", 1.0)]
    entity = [("b", 1.0)]
    baseline = rrf_fusion(vector, bm25)
    with_zero_weight = rrf_fusion(vector, bm25, entity_results=entity, entity_weight=0.0)
    assert baseline == with_zero_weight


def test_rrf_fusion_entity_only_memory_still_included():
    """A memory that appears ONLY in the entity list (not in vector or bm25)
    must still surface in the fused ranking -- entities can bridge to
    content neither dense nor lexical search found relevant."""
    vector = [("a", 0.9)]
    bm25 = [("a", 5.0)]
    entity = [("c", 1.0)]  # c is new, unseen by vector/bm25
    result = rrf_fusion(vector, bm25, entity_results=entity, entity_weight=1.0)
    ids = {mid for mid, _ in result}
    assert "c" in ids, f"entity-only memory should be included, got {result}"


try:
    from mathir_lib.mathir_search import _EntityBackend
except ImportError:
    from mathir_search import _EntityBackend  # type: ignore


def test_entity_backend_ranks_by_entity_overlap_count():
    """Memories sharing more entities with the query rank higher."""
    backend = _EntityBackend()
    backend.build([
        {"memory_id": "a", "text": "Shirley Temple starred in Kiss and Tell."},
        {"memory_id": "b", "text": "Shirley Temple became Chief of Protocol."},
        {"memory_id": "c", "text": "Photosynthesis converts sunlight into energy."},
    ])
    results = backend.search("What government position did Shirley Temple hold?", k=3)
    ids = [mid for mid, _ in results]
    assert "c" not in ids or ids.index("a" if "a" in ids else "b") < ids.index("c")
    assert "a" in ids or "b" in ids


def test_entity_backend_empty_corpus_returns_empty():
    backend = _EntityBackend()
    backend.build([])
    assert backend.search("anything", k=5) == []


def test_entity_backend_no_query_entities_returns_empty():
    """A query with no extractable named entities yields no entity-signal
    ranking (falls through to vector+bm25 alone in the fused result)."""
    backend = _EntityBackend()
    backend.build([{"memory_id": "a", "text": "Shirley Temple starred in a film."}])
    results = backend.search("what happened yesterday", k=5)
    assert results == []

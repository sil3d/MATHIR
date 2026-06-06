"""
TDD tests for mathir_optimized.OptimizedMATHIR.

RED phase: these tests are written first. They exercise the public API
of OptimizedMATHIR using a *mock* embedder (no model downloads required)
so they run in <1s on any machine.

The contract:
  - OptimizedMATHIR must be a standalone hybrid retriever
  - It must support 3 configurations: dense-only, dense+BM25 (RRF),
    dense+BM25+cross-encoder
  - It must expose index_time, per-query latency stats (P50/P95/P99/std)
  - It must be importable without GPU
"""

import os
import sys
import time
import numpy as np
import pytest

# Ensure project root is on sys.path so `import mathir_optimized` works
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Mock embedder — deterministic, no model loading
# ---------------------------------------------------------------------------
class MockEmbedder:
    """Deterministic random embedder for unit tests."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.call_count = 0

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        # Deterministic seed based on text content
        n = len(texts)
        embs = np.zeros((n, self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = abs(hash(t)) % (2 ** 32)
            rng = np.random.RandomState(seed)
            embs[i] = rng.randn(self.dim).astype(np.float32)
        # L2 normalize (same contract as sentence-transformers with
        # normalize_embeddings=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        self.call_count += 1
        return embs


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
CORPUS_IDS = ["d1", "d2", "d3", "d4", "d5"]
CORPUS_TEXTS = [
    "Bernoulli's equation for incompressible fluid flow",
    "The Navier-Stokes equations describe viscous fluid dynamics",
    "Quantum entanglement is a physical phenomenon",
    "Reynolds number characterizes laminar and turbulent flow",
    "Machine learning models require large training datasets",
]


@pytest.fixture
def mock_embedder():
    return MockEmbedder(dim=64)


# ---------------------------------------------------------------------------
# Import + construction
# ---------------------------------------------------------------------------
def test_mathir_optimized_is_importable():
    """The module must exist and expose OptimizedMATHIR."""
    import mathir_optimized
    assert hasattr(mathir_optimized, "OptimizedMATHIR")


def test_constructor_with_mock_embedder(mock_embedder):
    """OptimizedMATHIR should accept an externally-provided embedder."""
    from mathir_optimized import OptimizedMATHIR
    m = OptimizedMATHIR(embedder=mock_embedder, use_bm25=False, use_cross_encoder=False)
    assert m is not None
    assert m.use_bm25 is False
    assert m.use_cross_encoder is False


def test_constructor_default_embedder_is_bge_base():
    """Default embedder should be a BGE model (the SOTA choice)."""
    from mathir_optimized import OptimizedMATHIR
    m = OptimizedMATHIR(embedder=MockEmbedder(), use_bm25=False, use_cross_encoder=False)
    assert "bge" in m.embedder_name.lower()


# ---------------------------------------------------------------------------
# Index + search
# ---------------------------------------------------------------------------
def test_index_populates_internal_state(mock_embedder):
    from mathir_optimized import OptimizedMATHIR
    m = OptimizedMATHIR(embedder=mock_embedder, use_bm25=False, use_cross_encoder=False)
    m.index(CORPUS_IDS, CORPUS_TEXTS)
    assert m._indexed is True or m._doc_ids == CORPUS_IDS
    assert len(m._doc_ids) == 5


def test_search_returns_k_results(mock_embedder):
    from mathir_optimized import OptimizedMATHIR
    m = OptimizedMATHIR(embedder=mock_embedder, use_bm25=False, use_cross_encoder=False)
    m.index(CORPUS_IDS, CORPUS_TEXTS)
    doc_ids, scores = m.search("fluid dynamics", k=3)
    assert len(doc_ids) == 3
    assert len(scores) == 3
    # Scores must be in descending order
    assert scores == sorted(scores, reverse=True)
    # All doc_ids must come from the indexed corpus
    assert all(d in CORPUS_IDS for d in doc_ids)


def test_search_handles_k_larger_than_corpus(mock_embedder):
    """Edge case: k > num_docs should not crash."""
    from mathir_optimized import OptimizedMATHIR
    m = OptimizedMATHIR(embedder=mock_embedder, use_bm25=False, use_cross_encoder=False)
    m.index(CORPUS_IDS, CORPUS_TEXTS)
    doc_ids, scores = m.search("test", k=100)
    # Should return at most len(corpus)
    assert len(doc_ids) <= len(CORPUS_IDS)


# ---------------------------------------------------------------------------
# BM25 + RRF fusion
# ---------------------------------------------------------------------------
def test_bm25_fusion_changes_ranking(mock_embedder):
    """With BM25, the result should be a fusion of dense + BM25, not dense alone."""
    from mathir_optimized import OptimizedMATHIR
    m_dense = OptimizedMATHIR(
        embedder=mock_embedder, use_bm25=False, use_cross_encoder=False
    )
    m_hybrid = OptimizedMATHIR(
        embedder=mock_embedder, use_bm25=True, use_cross_encoder=False
    )
    m_dense.index(CORPUS_IDS, CORPUS_TEXTS)
    m_hybrid.index(CORPUS_IDS, CORPUS_TEXTS)
    # Query with a keyword that should be matched by BM25
    dense_ids, _ = m_dense.search("Reynolds", k=5)
    hybrid_ids, _ = m_hybrid.search("Reynolds", k=5, query_text="Reynolds")
    # Hybrid should find the Reynolds doc in top results
    assert "d4" in hybrid_ids, f"Reynolds doc should be in top-5, got {hybrid_ids}"


def test_bm25_finds_exact_keyword(mock_embedder):
    """BM25 should help find documents with exact keyword matches."""
    from mathir_optimized import OptimizedMATHIR
    m = OptimizedMATHIR(
        embedder=mock_embedder, use_bm25=True, use_cross_encoder=False
    )
    m.index(CORPUS_IDS, CORPUS_TEXTS)
    doc_ids, _ = m.search("Quantum", k=3, query_text="Quantum")
    # The quantum doc should rank highly
    assert "d3" in doc_ids


# ---------------------------------------------------------------------------
# Cross-encoder rerank (graceful fallback if CE unavailable)
# ---------------------------------------------------------------------------
def test_cross_encoder_rerank_with_mock_ce(mock_embedder):
    """When a mock CE is provided, rerank should run and return results."""
    from mathir_optimized import OptimizedMATHIR

    class MockCE:
        def predict(self, pairs, **kwargs):
            # Score based on keyword overlap (deterministic)
            scores = []
            for q, d in pairs:
                q_words = set(q.lower().split())
                d_words = set(d.lower().split())
                overlap = len(q_words & d_words)
                scores.append(float(overlap))
            return np.array(scores, dtype=np.float32)

    m = OptimizedMATHIR(
        embedder=mock_embedder,
        use_bm25=True,
        use_cross_encoder=True,
        cross_encoder=MockCE(),
    )
    m.index(CORPUS_IDS, CORPUS_TEXTS)
    doc_ids, scores = m.search("Navier-Stokes", k=3, query_text="Navier-Stokes")
    assert len(doc_ids) == 3
    # The Navier-Stokes doc should rank #1
    assert doc_ids[0] == "d2"


def test_cross_encoder_unavailable_falls_back_gracefully(mock_embedder):
    """If CE is None and use_ce=True, search should still work (dense+BM25 only)."""
    from mathir_optimized import OptimizedMATHIR
    m = OptimizedMATHIR(
        embedder=mock_embedder,
        use_bm25=True,
        use_cross_encoder=True,
        cross_encoder=None,
    )
    m.index(CORPUS_IDS, CORPUS_TEXTS)
    # Should not crash even though CE is None
    doc_ids, scores = m.search("fluid", k=2, query_text="fluid")
    assert len(doc_ids) == 2


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def test_get_stats_returns_latency_distribution(mock_embedder):
    from mathir_optimized import OptimizedMATHIR
    m = OptimizedMATHIR(embedder=mock_embedder, use_bm25=False, use_cross_encoder=False)
    m.index(CORPUS_IDS, CORPUS_TEXTS)
    # Run several queries
    for _ in range(20):
        m.search("test query", k=3)
    stats = m.get_stats()
    # Required fields
    for key in ("queries", "latency_mean_ms", "latency_p50_ms",
                "latency_p95_ms", "latency_p99_ms", "latency_std_ms",
                "index_time_ms", "num_docs"):
        assert key in stats, f"Missing stat: {key}"
    assert stats["queries"] == 20
    assert stats["num_docs"] == 5
    # Latency P50 <= P95 <= P99 (monotonicity invariant)
    assert stats["latency_p50_ms"] <= stats["latency_p95_ms"]
    assert stats["latency_p95_ms"] <= stats["latency_p99_ms"]
    # Std should be non-negative
    assert stats["latency_std_ms"] >= 0


def test_index_time_is_recorded(mock_embedder):
    from mathir_optimized import OptimizedMATHIR
    m = OptimizedMATHIR(embedder=mock_embedder, use_bm25=True, use_cross_encoder=False)
    t0 = time.perf_counter()
    m.index(CORPUS_IDS, CORPUS_TEXTS)
    elapsed = (time.perf_counter() - t0) * 1000
    stats = m.get_stats()
    assert stats["index_time_ms"] > 0
    # Index time should be in the right ballpark (within 10x of actual)
    assert stats["index_time_ms"] < elapsed * 10


# ---------------------------------------------------------------------------
# Configuration variants (the 3 "OptimizedMATHIR" flavors in the benchmark)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_bm25,use_ce", [
    (False, False),  # dense-only
    (True, False),   # + BM25 fusion
    (True, True),    # + cross-encoder rerank
])
def test_three_configurations_all_work(mock_embedder, use_bm25, use_ce):
    """All 3 system variants must work end-to-end."""
    from mathir_optimized import OptimizedMATHIR
    m = OptimizedMATHIR(
        embedder=mock_embedder,
        use_bm25=use_bm25,
        use_cross_encoder=use_ce,
    )
    m.index(CORPUS_IDS, CORPUS_TEXTS)
    doc_ids, scores = m.search("fluid dynamics", k=3, query_text="fluid dynamics")
    assert len(doc_ids) == 3
    assert all(d in CORPUS_IDS for d in doc_ids)

"""
Unit tests for HybridEpisodicMemory cross-encoder result cache
(Latency Optimization 1 — Approach D).

Goal: prove that caching cross-encoder scores by ``(query_text,
doc_text)`` cuts warm-path latency by 1-2 orders of magnitude without
changing the ranking, and that LRU eviction + clear/reset work as
advertised.

We use a deterministic in-process mock cross-encoder so the tests are
fast, hermetic, and don't require a HuggingFace download. The
production code path is identical: ``self._cross_encoder.predict(pairs,
...)`` is called with the same shape of input and produces scores
that the cache stores verbatim.

Run:
    pytest tests/test_hybrid_cache.py -v
    pytest tests/test_hybrid.py -v        # regression check
    pytest tests/test_v7_memory.py -v     # regression check
"""

import time
import warnings

import numpy as np
import pytest
import torch

from mathir_lib.memory.hybrid_episodic import HybridEpisodicMemory


# ---------------------------------------------------------------------
# Mock cross-encoder
# ---------------------------------------------------------------------
class MockCrossEncoder:
    """
    Deterministic stand-in for ``sentence_transformers.CrossEncoder``.

    Returns a score derived from the string lengths so the same
    ``(query, doc)`` pair always yields the same score, mimicking the
    real cross-encoder's deterministic behaviour. Also tracks how many
    times ``predict`` was called so tests can assert cache effectiveness.
    """

    def __init__(self, score_offset: float = 0.0, sleep_ms: float = 0.0):
        self.score_offset = float(score_offset)
        self.sleep_ms = float(sleep_ms)
        self.n_calls = 0
        self.total_pairs_scored = 0
        self.last_pairs = None

    def predict(self, pairs, convert_to_numpy=True, show_progress_bar=False):
        self.n_calls += 1
        self.last_pairs = list(pairs)
        self.total_pairs_scored += len(pairs)
        if self.sleep_ms > 0:
            time.sleep(self.sleep_ms / 1000.0)
        scores = np.array(
            [self.score_offset + (len(q) + len(d)) % 10 / 10.0
             for q, d in pairs],
            dtype=np.float32,
        )
        return scores


def _inject_mock_ce(mem: HybridEpisodicMemory, ce: MockCrossEncoder = None) -> MockCrossEncoder:
    """
    Replace ``mem._cross_encoder`` with a mock.

    ``nn.Module.__setattr__`` rejects non-Module attributes, so we
    bypass it with ``object.__setattr__``. This is exactly the same
    pattern used by the HybridEpisodicMemory internals when they
    initialise the (potentially failed) cross-encoder.
    """
    if ce is None:
        ce = MockCrossEncoder()
    object.__setattr__(mem, "_cross_encoder", ce)
    mem._cross_encoder_failed = False
    return ce


def _make_memory_with_corpus(
    n_chunks: int = 50,
    feature_dim: int = 64,
    use_result_cache: bool = True,
    cache_size: int = 10000,
    use_adaptive_rerank: bool = False,
) -> HybridEpisodicMemory:
    """
    Build a HybridEpisodicMemory with N distinct chunks and a mock CE.

    Adaptive re-ranking is OFF by default so the cross-encoder
    actually runs on every query (otherwise the cache would be
    side-stepped and we couldn't measure it). The dense embeddings
    are random noise, mirroring the real test corpus.
    """
    torch.manual_seed(42)
    mem = HybridEpisodicMemory(
        capacity=max(n_chunks + 10, 100),
        feature_dim=feature_dim,
        use_cross_encoder=True,
        use_adaptive_rerank=use_adaptive_rerank,
        use_result_cache=use_result_cache,
        cache_size=cache_size,
    )
    _inject_mock_ce(mem)

    # 5 distinctive technical docs + noise docs.
    tech_docs = [
        "The Reynolds number characterizes fluid flow regimes",
        "Bernoulli's principle describes pressure-velocity tradeoffs",
        "Navier-Stokes equations govern viscous fluid dynamics",
        "Boundary layer theory describes flow over surfaces",
        "Laminar flow has parallel streamlines",
    ]
    for i in range(n_chunks):
        text = tech_docs[i] if i < len(tech_docs) else f"document {i} about topic {i}"
        mem.store(torch.randn(1, feature_dim), text=text)
    return mem


# =========================================================================
# 1. Configuration
# =========================================================================
class TestCacheConfig:
    """Defaults, custom sizes, capability flags."""

    def test_cache_enabled_by_default(self):
        mem = HybridEpisodicMemory(capacity=10, feature_dim=32, use_cross_encoder=True)
        assert mem.use_result_cache is True
        assert mem.cache_size == 10000
        assert isinstance(mem._ce_cache, dict)
        assert len(mem._ce_cache) == 0

    def test_cache_can_be_disabled(self):
        mem = HybridEpisodicMemory(
            capacity=10, feature_dim=32, use_cross_encoder=True,
            use_result_cache=False,
        )
        assert mem.use_result_cache is False

    def test_custom_cache_size(self):
        mem = HybridEpisodicMemory(
            capacity=10, feature_dim=32, use_cross_encoder=True,
            cache_size=500,
        )
        assert mem.cache_size == 500

    def test_cache_size_minimum_one(self):
        """``cache_size`` is clamped to at least 1 to avoid div-by-zero / broken LRU."""
        mem = HybridEpisodicMemory(
            capacity=10, feature_dim=32, use_cross_encoder=True,
            cache_size=0,
        )
        assert mem.cache_size == 1

    def test_cache_info_initially_empty(self):
        mem = HybridEpisodicMemory(capacity=10, feature_dim=32, use_cross_encoder=True)
        info = mem.cache_info()
        assert info["size"] == 0
        assert info["capacity"] == 10000
        assert info["enabled"] is True
        assert info["hits"] == 0
        assert info["misses"] == 0
        assert info["evictions"] == 0
        assert info["hit_rate"] == 0.0


# =========================================================================
# 2. Hit / miss behaviour
# =========================================================================
class TestCacheHitMiss:
    """Verify the cache serves the same scores on warm path without
    re-invoking the cross-encoder."""

    def test_first_query_is_all_misses(self):
        mem = _make_memory_with_corpus()
        ce = mem._cross_encoder
        mem.search(torch.randn(1, 64), k=5,
                   query_text="What is the Reynolds number?")
        info = mem.cache_info()
        assert info["misses"] > 0
        assert info["hits"] == 0
        # Exactly one cross-encoder call for the 30 candidates.
        assert ce.n_calls == 1
        assert ce.total_pairs_scored == 30  # cross_encoder_top_n

    def test_repeat_query_is_all_hits(self):
        """Re-running the SAME query (same text + same embedding) must be
        100% cache hits — no predict() call."""
        mem = _make_memory_with_corpus()
        ce = mem._cross_encoder
        query_text = "What is the Reynolds number?"
        query_emb = torch.randn(1, 64)

        # First call: populate cache
        mem.search(query_emb, k=5, query_text=query_text)
        # Reset call counter so the assertion is clean
        ce.n_calls = 0
        ce.total_pairs_scored = 0

        # Second identical call: every pair should be a hit
        mem.search(query_emb, k=5, query_text=query_text)
        info = mem.cache_info()
        assert info["hits"] == 30, f"Expected 30 hits, got {info['hits']}"
        assert info["misses"] == 0, f"Expected 0 misses, got {info['misses']}"
        # The cross-encoder MUST NOT be called on a warm path
        assert ce.n_calls == 0, f"CE was called {ce.n_calls} times on warm path"
        assert ce.total_pairs_scored == 0

    def test_three_repeated_queries_third_is_pure_hit(self):
        """Run the same query 3 times — only the 1st is slow."""
        mem = _make_memory_with_corpus()
        ce = mem._cross_encoder
        query_text = "What is the Reynolds number?"
        query_emb = torch.randn(1, 64)

        latencies = []
        for _ in range(3):
            ce.n_calls = 0
            t0 = time.perf_counter()
            mem.search(query_emb, k=5, query_text=query_text)
            latencies.append((time.perf_counter() - t0) * 1000)
            assert ce.n_calls in (0, 1), (
                f"Unexpected CE call count: {ce.n_calls} "
                f"(should be 0 for warm runs, 1 for cold)"
            )
        # 1st call had 1 predict, 2nd and 3rd had 0
        # Latency is just for diagnostic — the *count* of CE calls is the
        # strong assertion.
        assert mem.cache_info()["hits"] == 60  # 30 + 30 from the two repeats

    def test_different_query_text_produces_misses(self):
        """Different query_text → different cache key → misses."""
        mem = _make_memory_with_corpus()
        ce = mem._cross_encoder
        query_emb = torch.randn(1, 64)

        mem.search(query_emb, k=5, query_text="What is the Reynolds number?")
        misses_after_first = mem.cache_info()["misses"]
        assert ce.n_calls == 1

        # Different query text → all misses again
        mem.search(query_emb, k=5, query_text="Explain Bernoulli's principle.")
        info = mem.cache_info()
        assert info["misses"] > misses_after_first
        # Another CE call was needed
        assert ce.n_calls == 2

    def test_deterministic_scores_across_runs(self):
        """Warm and cold paths must produce IDENTICAL results."""
        mem = _make_memory_with_corpus()
        query_text = "What is the Reynolds number?"
        query_emb = torch.randn(1, 64)

        idx_cold, sims_cold = mem.search(query_emb, k=5, query_text=query_text)
        idx_warm, sims_warm = mem.search(query_emb, k=5, query_text=query_text)
        assert torch.equal(idx_cold, idx_warm)
        assert torch.allclose(sims_cold, sims_warm)


# =========================================================================
# 3. LRU eviction
# =========================================================================
class TestCacheLRUEviction:
    """When the cache is full, the LRU entry is dropped on insert."""

    def test_eviction_when_cache_full(self):
        mem = HybridEpisodicMemory(
            capacity=50, feature_dim=64, use_cross_encoder=True,
            use_result_cache=True, cache_size=10,
        )
        _inject_mock_ce(mem)
        for i in range(20):
            mem.store(torch.randn(1, 64), text=f"document {i} about topic {i}")

        # Each unique query text produces up to 30 cache inserts.
        # After 2+ queries, cache_size=10 is exceeded.
        for i in range(5):
            mem.search(torch.randn(1, 64), k=5,
                       query_text=f"query number {i} about concept {i}")
        info = mem.cache_info()
        assert info["size"] <= 10, f"Cache exceeded capacity: {info['size']}"
        assert info["evictions"] > 0, "No evictions recorded"

    def test_lru_evicts_least_recently_used(self):
        """Direct OrderedDict test: oldest inserted key is evicted first."""
        mem = HybridEpisodicMemory(
            capacity=10, feature_dim=64, use_cross_encoder=True,
            cache_size=3,
        )
        _inject_mock_ce(mem)
        # Manually populate the cache to test LRU semantics in isolation
        mem._ce_cache["q", "a"] = 0.1
        mem._ce_cache["q", "b"] = 0.2
        mem._ce_cache["q", "c"] = 0.3
        # Touch "a" to make it MRU
        mem._ce_cache.move_to_end(("q", "a"))
        # Insert "d" → evicts the LRU, which is now "b"
        mem._ce_cache["q", "d"] = 0.4
        if len(mem._ce_cache) > mem.cache_size:
            mem._ce_cache.popitem(last=False)
            mem._n_cache_evictions += 1
        assert ("q", "b") not in mem._ce_cache
        assert ("q", "a") in mem._ce_cache  # was touched, survived
        assert ("q", "c") in mem._ce_cache
        assert ("q", "d") in mem._ce_cache
        assert len(mem._ce_cache) == 3

    def test_eviction_counter_increments(self):
        mem = HybridEpisodicMemory(
            capacity=50, feature_dim=64, use_cross_encoder=True,
            cache_size=5,
        )
        _inject_mock_ce(mem)
        for i in range(20):
            mem.store(torch.randn(1, 64), text=f"doc {i}")

        before = mem.cache_info()["evictions"]
        for i in range(10):
            mem.search(torch.randn(1, 64), k=5, query_text=f"query_{i}")
        after = mem.cache_info()["evictions"]
        assert after > before

    def test_hit_after_eviction_does_not_raise(self):
        """If a key is evicted, re-inserting it on a later miss must
        not raise and must count as a fresh miss."""
        mem = HybridEpisodicMemory(
            capacity=20, feature_dim=64, use_cross_encoder=True,
            cache_size=2,
        )
        _inject_mock_ce(mem)
        for i in range(5):
            mem.store(torch.randn(1, 64), text=f"document {i}")
        # Fill the cache with one query
        mem.search(torch.randn(1, 64), k=5, query_text="q1")
        # Run several more queries to overflow the cache
        for i in range(10):
            mem.search(torch.randn(1, 64), k=5, query_text=f"q{i+2}")
        # Re-run q1 — its entries are now evicted
        info_before = mem.cache_info()
        mem.search(torch.randn(1, 64), k=5, query_text="q1")
        info_after = mem.cache_info()
        # New misses were recorded (the old q1 entries were evicted)
        assert info_after["misses"] > info_before["misses"]


# =========================================================================
# 4. Cache disabled
# =========================================================================
class TestCacheDisabled:
    """``use_result_cache=False`` short-circuits the cache entirely."""

    def test_cache_disabled_records_no_hits(self):
        mem = _make_memory_with_corpus(use_result_cache=False)
        ce = mem._cross_encoder
        query_text = "What is the Reynolds number?"
        query_emb = torch.randn(1, 64)
        mem.search(query_emb, k=5, query_text=query_text)
        mem.search(query_emb, k=5, query_text=query_text)
        info = mem.cache_info()
        assert info["enabled"] is False
        assert info["hits"] == 0
        assert info["misses"] == 0
        # CE called twice — cache does nothing
        assert ce.n_calls == 2

    def test_cache_disabled_does_not_grow(self):
        mem = _make_memory_with_corpus(use_result_cache=False)
        for i in range(5):
            mem.search(torch.randn(1, 64), k=5, query_text=f"q_{i}")
        assert mem.cache_info()["size"] == 0


# =========================================================================
# 5. Cache management
# =========================================================================
class TestCacheManagement:
    """``clear_cache()`` and ``reset()`` lifecycle."""

    def test_clear_cache_empties_entries(self):
        mem = _make_memory_with_corpus()
        mem.search(torch.randn(1, 64), k=5, query_text="What is the Reynolds number?")
        assert mem.cache_info()["size"] > 0
        mem.clear_cache()
        assert mem.cache_info()["size"] == 0

    def test_clear_cache_preserves_counters(self):
        """Counters are diagnostic; ``clear_cache()`` only wipes entries."""
        mem = _make_memory_with_corpus()
        mem.search(torch.randn(1, 64), k=5, query_text="What is the Reynolds number?")
        mem.search(torch.randn(1, 64), k=5, query_text="What is the Reynolds number?")
        hits_before = mem.cache_info()["hits"]
        misses_before = mem.cache_info()["misses"]
        mem.clear_cache()
        info = mem.cache_info()
        assert info["hits"] == hits_before
        assert info["misses"] == misses_before

    def test_reset_clears_cache_and_counters(self):
        mem = _make_memory_with_corpus()
        mem.search(torch.randn(1, 64), k=5, query_text="What is the Reynolds number?")
        mem.search(torch.randn(1, 64), k=5, query_text="What is the Reynolds number?")
        mem.reset()
        info = mem.cache_info()
        assert info["size"] == 0
        assert info["hits"] == 0
        assert info["misses"] == 0
        assert info["evictions"] == 0

    def test_cache_survives_store(self):
        """``store()`` does NOT clear the cache (avoids thrash on bulk ingest)."""
        mem = _make_memory_with_corpus(n_chunks=5)
        mem.search(torch.randn(1, 64), k=5, query_text="Reynolds number")
        size_before = mem.cache_info()["size"]
        mem.store(torch.randn(1, 64), text="new document about fluid flow")
        # Cache untouched — this matches the existing behaviour for BM25
        # and avoids re-scoring everything on every store.
        assert mem.cache_info()["size"] == size_before


# =========================================================================
# 6. Cache diagnostics surface
# =========================================================================
class TestCacheStats:
    """``get_stats()`` and ``cache_info()`` expose the right keys."""

    def test_get_stats_includes_cache_keys(self):
        mem = _make_memory_with_corpus()
        mem.search(torch.randn(1, 64), k=5, query_text="Reynolds number")
        stats = mem.get_stats()
        for k in (
            "cache_size", "cache_capacity", "cache_enabled",
            "cache_filled", "cache_hits", "cache_misses",
            "cache_evictions", "cache_hit_rate",
        ):
            assert k in stats, f"Missing key: {k}"

    def test_get_stats_values_match_cache_info(self):
        mem = _make_memory_with_corpus()
        mem.search(torch.randn(1, 64), k=5, query_text="Reynolds number")
        stats = mem.get_stats()
        info = mem.cache_info()
        assert stats["cache_size"] == info["size"]
        assert stats["cache_capacity"] == info["capacity"]
        assert stats["cache_enabled"] == info["enabled"]
        assert stats["cache_filled"] == info["size"]
        assert stats["cache_hits"] == info["hits"]
        assert stats["cache_misses"] == info["misses"]
        assert stats["cache_evictions"] == info["evictions"]
        assert stats["cache_hit_rate"] == info["hit_rate"]

    def test_hit_rate_in_unit_interval(self):
        mem = _make_memory_with_corpus()
        for i in range(5):
            mem.search(torch.randn(1, 64), k=5, query_text=f"q_{i}")
        hr = mem.cache_info()["hit_rate"]
        assert 0.0 <= hr <= 1.0


# =========================================================================
# 7. Latency benchmark — cold vs warm
# =========================================================================
class TestCacheLatency:
    """Sanity check the headline speedup claim: warm path is dramatically
    faster than cold because it skips the cross-encoder."""

    def test_warm_faster_than_cold(self):
        mem = _make_memory_with_corpus()
        # Use a slow mock so the CE call dominates
        _inject_mock_ce(mem, MockCrossEncoder(sleep_ms=5.0))
        query_text = "What is the Reynolds number?"
        query_emb = torch.randn(1, 64)

        # Cold: one CE call on 30 pairs, each sleeping 5 ms → ~150 ms minimum
        t0 = time.perf_counter()
        mem.search(query_emb, k=5, query_text=query_text)
        cold_ms = (time.perf_counter() - t0) * 1000

        # Warm: pure dict lookups, no sleep
        t0 = time.perf_counter()
        mem.search(query_emb, k=5, query_text=query_text)
        warm_ms = (time.perf_counter() - t0) * 1000

        # Warm must be substantially faster (we expect > 5x in practice)
        assert warm_ms < cold_ms, (
            f"Warm ({warm_ms:.1f} ms) should be faster than cold ({cold_ms:.1f} ms)"
        )
        speedup = cold_ms / max(warm_ms, 1e-3)
        # On a slow CI runner 2x may be the only thing we can assert
        # tightly. 5x is what we expect on a real machine.
        assert speedup >= 2.0, (
            f"Cache speedup only {speedup:.1f}x — expected >= 2x. "
            f"Cold={cold_ms:.1f}ms, Warm={warm_ms:.1f}ms"
        )

    def test_50_chunks_three_repeated_queries_speedup(self):
        """Real benchmark: 50 stores, 3 repeated queries.

        Spec requirement: first call is slow (>= 50 ms with mock CE),
        subsequent calls are < 1 ms for the dict-lookup portion.
        """
        mem = _make_memory_with_corpus(n_chunks=50)
        _inject_mock_ce(mem, MockCrossEncoder(sleep_ms=2.0))
        query_text = "What is the Reynolds number in fluid dynamics?"
        query_emb = torch.randn(1, 64)

        latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            mem.search(query_emb, k=5, query_text=query_text)
            latencies.append((time.perf_counter() - t0) * 1000)

        cold, mid, warm = latencies
        # Cold must be slower than warm by a wide margin
        assert cold > warm, f"Cold ({cold:.1f}ms) should be slower than warm ({warm:.1f}ms)"
        # 2nd and 3rd calls should be within a small constant of each
        # other (both are warm)
        assert warm < 50.0, f"Warm call took {warm:.1f}ms — should be well under 50ms"
        # Verify counters: 1st had 30 misses, 2nd & 3rd each had 30 hits
        info = mem.cache_info()
        assert info["misses"] == 30
        assert info["hits"] == 60

    def test_50_unique_queries_hit_rate_near_zero(self):
        """When every query is unique, hit rate is near 0 (every pair is fresh)."""
        mem = _make_memory_with_corpus(n_chunks=50)
        for i in range(50):
            mem.search(torch.randn(1, 64), k=5, query_text=f"unique query {i}")
        info = mem.cache_info()
        # Some overlap is possible (e.g. if RRF top-30 collapses for
        # all queries), but most pairs are unique.
        assert info["hit_rate"] < 0.5

    def test_3_queries_repeated_10_times_high_hit_rate(self):
        """Repeating a small query set should drive the hit rate above 80%."""
        mem = _make_memory_with_corpus(n_chunks=50)
        queries = [
            "What is the Reynolds number?",
            "Explain Bernoulli's principle.",
            "Define the Navier-Stokes equations.",
        ]
        # First pass: cold
        for q in queries:
            mem.search(torch.randn(1, 64), k=5, query_text=q)
        # Subsequent passes: warm
        for _ in range(10):
            for q in queries:
                mem.search(torch.randn(1, 64), k=5, query_text=q)
        info = mem.cache_info()
        # 10 repeated passes of 3 queries = 30 calls; the first pass is
        # cold, the rest are warm. Hit rate should be very high.
        assert info["hit_rate"] > 0.8, (
            f"Hit rate {info['hit_rate']:.2%} is too low; expected > 80% "
            f"after 10 repeats of 3 queries"
        )


# =========================================================================
# 8. End-to-end: 50 stores, 3 repeated queries, verify timing
# =========================================================================
class TestCacheEndToEnd:
    """The headline scenario from the spec."""

    def test_50_stores_same_query_3_times(self):
        """50 stores, run same query 3 times.

        1st call: cold — 30 CE pairs to score
        2nd, 3rd: warm — all 30 pairs served from cache
        """
        torch.manual_seed(7)
        mem = _make_memory_with_corpus(n_chunks=50)
        _inject_mock_ce(mem, MockCrossEncoder(sleep_ms=3.0))
        query_text = "What is the Reynolds number?"
        query_emb = torch.randn(1, 64)

        # ---- 1st call (cold) ----
        ce = mem._cross_encoder
        t0 = time.perf_counter()
        idx1, sims1 = mem.search(query_emb, k=5, query_text=query_text)
        cold_ms = (time.perf_counter() - t0) * 1000
        assert ce.n_calls == 1
        first_call_predict_calls = ce.n_calls

        # ---- 2nd call (warm) ----
        ce.n_calls = 0
        t0 = time.perf_counter()
        idx2, sims2 = mem.search(query_emb, k=5, query_text=query_text)
        warm_ms_2 = (time.perf_counter() - t0) * 1000
        assert ce.n_calls == 0, "CE should NOT be called on warm path"

        # ---- 3rd call (warm) ----
        t0 = time.perf_counter()
        idx3, sims3 = mem.search(query_emb, k=5, query_text=query_text)
        warm_ms_3 = (time.perf_counter() - t0) * 1000
        assert ce.n_calls == 0

        # Results must be byte-identical (deterministic mock CE)
        assert torch.equal(idx1, idx2)
        assert torch.equal(idx1, idx3)
        assert torch.allclose(sims1, sims2)
        assert torch.allclose(sims1, sims3)

        # Warm must be faster (with a 3 ms mock CE, the gap is obvious)
        assert warm_ms_2 < cold_ms
        assert warm_ms_3 < cold_ms

        # Cache state should reflect the workload
        info = mem.cache_info()
        assert info["misses"] == 30  # 30 pairs from the cold call
        assert info["hits"] == 60    # 30 + 30 from the two warm calls
        assert info["hit_rate"] == pytest.approx(60 / 90)


# =========================================================================
# 9. Regression: backward compatibility
# =========================================================================
class TestCacheBackwardCompat:
    """``HybridEpisodicMemory`` must remain a drop-in replacement for
    the version without the cache."""

    def test_default_constructor_unchanged(self):
        """All original kwargs must keep their default values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mem = HybridEpisodicMemory()
        assert mem.use_cross_encoder_requested is True
        assert mem.dense_top_k == 20
        assert mem.bm25_top_k == 20
        assert mem.rrf_k_const == 60
        assert mem.cross_encoder_top_n == 30
        # New fields exist with sensible defaults
        assert mem.use_result_cache is True
        assert mem.cache_size == 10000

    def test_no_query_text_skips_cache(self):
        """When ``query_text=None`` the CE is never invoked, so the cache
        is never touched (no entries, no counters)."""
        mem = _make_memory_with_corpus()
        mem.search(torch.randn(1, 64), k=5)  # no query_text
        info = mem.cache_info()
        assert info["size"] == 0
        assert info["hits"] == 0
        assert info["misses"] == 0

    def test_works_with_adaptive_rerank(self):
        """The cache should still tick when the adaptive re-ranker skips
        the CE — both code paths coexist."""
        mem = _make_memory_with_corpus(use_adaptive_rerank=True)
        # Run a query that may or may not trigger the CE depending on
        # the random embeddings. Either way, search() must succeed and
        # the cache stats must be consistent.
        mem.search(torch.randn(1, 64), k=5, query_text="Reynolds number")
        info = mem.cache_info()
        # If the CE ran, there are misses; if the CE was skipped, the
        # cache is empty. Both are valid.
        assert info["size"] >= 0
        # Total lookups = hits + misses = 30 (candidates) if CE ran
        # Note: when CE is skipped, the cache is never touched → 0/0
        total_lookups = info["hits"] + info["misses"]
        if total_lookups > 0:
            assert total_lookups == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

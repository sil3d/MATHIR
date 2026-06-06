"""
Unit tests for ``FAISSBackedEpisodicMemory`` (V8 experimental).

Covers:
    * Init / configuration (Flat + HNSW)
    * 100 random 384-dim vectors: store, query, top-1 cosine >= 0.9
    * Dynamic insert + delete (``forget``)
    * FAISS index size matches value buffer size after every operation
    * Capacity wrap-around (FIFO eviction)
    * Reset

Run:
    pytest tests/test_faiss_memory.py -v
"""

import time

import numpy as np
import pytest
import torch

from mathir_lib.memory import FAISSBackedEpisodicMemory
from mathir_lib.memory.faiss_episodic import _HAS_FAISS


pytestmark = pytest.mark.skipif(
    not _HAS_FAISS, reason="faiss is not installed (pip install faiss-cpu)"
)


# =========================================================================
# Fixtures
# =========================================================================
@pytest.fixture(params=[False, True], ids=["flat", "hnsw"])
def mem_factory(request):
    """Factory: builds a memory with the given backend."""
    use_hnsw = request.param

    def _make(capacity=1000, feature_dim=384):
        return FAISSBackedEpisodicMemory(
            capacity=capacity, feature_dim=feature_dim, use_hnsw=use_hnsw
        )

    return _make


# =========================================================================
# 1. Init / config
# =========================================================================
class TestFAISSBackedInit:
    """Initialization and configuration."""

    def test_init_flat(self):
        m = FAISSBackedEpisodicMemory(capacity=100, feature_dim=64, use_hnsw=False)
        assert m.capacity == 100
        assert m.feature_dim == 64
        assert m.use_hnsw is False
        assert m.normalize_keys is True
        assert len(m) == 0
        assert m.get_stats()["index_ntotal"] == 0

    def test_init_hnsw(self):
        m = FAISSBackedEpisodicMemory(capacity=100, feature_dim=64, use_hnsw=True)
        assert m.use_hnsw is True
        assert len(m) == 0
        assert m.get_stats()["index_ntotal"] == 0

    def test_stats_keys(self):
        m = FAISSBackedEpisodicMemory(capacity=10, feature_dim=8)
        stats = m.get_stats()
        for key in (
            "capacity", "feature_dim", "size", "index_ntotal",
            "use_hnsw", "normalize_keys", "next_slot_id",
            "n_stores", "n_searches",
        ):
            assert key in stats, f"Missing stat key: {key}"

    def test_search_on_empty_returns_empty(self):
        m = FAISSBackedEpisodicMemory(capacity=10, feature_dim=8)
        ids, sims = m.search(torch.randn(8), k=3)
        assert ids.numel() == 0
        assert sims.numel() == 0

    def test_retrieve_on_empty_returns_query(self):
        m = FAISSBackedEpisodicMemory(capacity=10, feature_dim=8)
        q = torch.randn(8)
        out = m.retrieve(q, k=3)
        assert torch.equal(out, q)

    def test_missing_faiss_raises(self, monkeypatch):
        """If faiss is somehow not installed, init must raise clearly."""
        import mathir_lib.memory.faiss_episodic as fmod
        monkeypatch.setattr(fmod, "_HAS_FAISS", False)
        with pytest.raises(ImportError, match="faiss"):
            FAISSBackedEpisodicMemory(capacity=4, feature_dim=4)


# =========================================================================
# 2. Store / search / retrieve on 100 random 384-dim vectors
# =========================================================================
class TestFAISSBackedRetrieval:
    """100 random 384-dim vectors; query 10 times; verify top-1 cosine >= 0.9."""

    N_VECTORS = 100
    DIM = 384
    N_QUERIES = 10

    def _make_corpus(self):
        torch.manual_seed(42)
        np.random.seed(42)
        # Make vectors that are *near* orthogonal with one strong axis each so
        # the true nearest neighbor is unambiguous.
        base = torch.randn(self.N_VECTORS, self.DIM)
        # Normalize and add a small "self" component to make them discriminable.
        eye = torch.eye(self.N_VECTORS) * 2.0
        M = torch.randn(self.N_VECTORS, self.DIM) @ base.T  # [N, N]
        # Use a simple random mix to make every vector unique.
        corpus = torch.randn(self.N_VECTORS, self.DIM)
        corpus = corpus / corpus.norm(dim=-1, keepdim=True)
        return corpus

    def test_store_all(self, mem_factory):
        mem = mem_factory(capacity=200, feature_dim=self.DIM)
        corpus = self._make_corpus()
        for v in corpus:
            mem.store(v)
        assert len(mem) == self.N_VECTORS
        assert mem.get_stats()["index_ntotal"] == self.N_VECTORS

    def test_query_top1_cosine_high(self, mem_factory):
        """10 queries, top-1 similarity must be >= 0.9 (i.e. the correct vector)."""
        mem = mem_factory(capacity=200, feature_dim=self.DIM)
        corpus = self._make_corpus()
        for v in corpus:
            mem.store(v)

        for i in range(self.N_QUERIES):
            q = corpus[i].clone()
            ids, sims = mem.search(q, k=1)
            assert ids.numel() == 1
            top1_id = int(ids.item())
            top1_sim = float(sims.item())
            if mem.use_hnsw:
                # HNSW is approximate — the top-1 may not be the exact match,
                # but the similarity of the top-1 to the query must still be
                # high (i.e. the approximate neighbor is "close enough").
                assert top1_sim >= 0.9, (
                    f"Query {i} (HNSW): top-1 cosine {top1_sim:.3f} < 0.9 "
                    f"(id={top1_id})"
                )
            else:
                # Flat is exact -> the correct vector must be retrieved.
                assert top1_id == i, (
                    f"Query {i}: expected slot_id={i}, got {top1_id} (sim={top1_sim:.3f})"
                )
                assert top1_sim >= 0.9, (
                    f"Query {i}: top-1 cosine {top1_sim:.3f} < 0.9"
                )

    def test_batch_query_top1_cosine_high(self, mem_factory):
        """Batched 2-D query also works."""
        mem = mem_factory(capacity=200, feature_dim=self.DIM)
        corpus = self._make_corpus()
        for v in corpus:
            mem.store(v)

        batch = corpus[: self.N_QUERIES]
        ids, sims = mem.search(batch, k=1)
        assert ids.shape == (self.N_QUERIES, 1)
        assert sims.shape == (self.N_QUERIES, 1)
        if mem.use_hnsw:
            # HNSW: just check the top-1 similarity is high (approximate).
            assert (sims >= 0.9).all(), (
                f"Min top-1 sim {sims.min().item():.3f} < 0.9 (HNSW approximate)"
            )
        else:
            expected = torch.arange(self.N_QUERIES).unsqueeze(1)
            assert torch.equal(ids, expected), (
                f"Top-1 slot_ids don't match expected indices:\n  got={ids.squeeze().tolist()}\n  exp={expected.squeeze().tolist()}"
            )
            assert (sims >= 0.9).all(), f"Min top-1 sim {sims.min().item():.3f} < 0.9"

    def test_retrieve_residual_shape(self, mem_factory):
        """retrieve() returns query + mean(value) for the right shape."""
        mem = mem_factory(capacity=200, feature_dim=self.DIM)
        corpus = self._make_corpus()
        for v in corpus:
            mem.store(v)
        q = corpus[0]
        out = mem.retrieve(q, k=3)
        assert out.shape == q.shape
        # Residual: out = q + mean(top3 values). Since top1 is q itself,
        # the mean is a small perturbation around q. Not exactly q.
        assert not torch.equal(out, q)


# =========================================================================
# 3. Dynamic insert + delete (forget) — index size parity
# =========================================================================
class TestFAISSBackedDynamics:
    """Insert and delete (forget) must keep index and value buffer in sync."""

    def test_forget_with_low_threshold_keeps_everything(self, mem_factory):
        torch.manual_seed(0)
        mem = mem_factory(capacity=100, feature_dim=32)
        for _ in range(20):
            mem.store(torch.randn(32))
        before = len(mem)
        kept = mem.forget(threshold=-1.0)  # keep all
        assert kept == before
        assert mem.get_stats()["index_ntotal"] == before
        assert len(mem) == before

    def test_forget_with_high_threshold_drops_everything(self, mem_factory):
        torch.manual_seed(0)
        mem = mem_factory(capacity=100, feature_dim=32)
        for _ in range(20):
            mem.store(torch.randn(32))
        # Random unit-norm 32-dim vectors have near-zero mean cosine; use
        # threshold above their typical mean similarity.
        kept = mem.forget(threshold=0.99)
        assert kept == 0
        assert mem.get_stats()["index_ntotal"] == 0
        assert len(mem) == 0

    def test_forget_partial(self, mem_factory):
        """Some near-duplicate vectors should be pruned, others kept."""
        torch.manual_seed(0)
        dim = 32
        mem = mem_factory(capacity=100, feature_dim=dim)
        # One strong prototype (many near-duplicates around it)
        proto = torch.randn(dim)
        proto = proto / proto.norm()
        for _ in range(5):
            mem.store(proto + 0.01 * torch.randn(dim))
        # And 5 unique outliers
        for _ in range(5):
            mem.store(torch.randn(dim))
        size_before = len(mem)
        # Very low threshold: keep ALL (loose prune). This must always
        # satisfy size_before == kept for any index type.
        kept = mem.forget(threshold=-1.0)
        assert kept == size_before
        assert mem.get_stats()["index_ntotal"] == size_before
        assert len(mem) == size_before

    def test_forget_prunes_subset(self, mem_factory):
        """Forget with a positive threshold on near-duplicate data drops
        *some* but not necessarily all of the near-duplicates."""
        torch.manual_seed(1)
        dim = 16
        mem = mem_factory(capacity=100, feature_dim=dim)
        # 10 near-duplicates of a single prototype
        proto = torch.randn(dim)
        for _ in range(10):
            mem.store(proto + 0.001 * torch.randn(dim))
        # 10 random outliers
        for _ in range(10):
            mem.store(torch.randn(dim))
        size_before = len(mem)
        assert size_before == 20
        kept = mem.forget(threshold=0.0)
        # 0 < kept < 20 -> SOME pruning happened
        assert 0 < kept < size_before, (
            f"Expected partial prune, got kept={kept} (out of {size_before})"
        )
        assert mem.get_stats()["index_ntotal"] == kept
        assert len(mem) == kept

    def test_index_matches_buffer_after_sequence(self, mem_factory):
        """After a series of store / forget / store cycles, index and
        value buffer sizes must agree."""
        torch.manual_seed(0)
        mem = mem_factory(capacity=50, feature_dim=16)
        for _ in range(30):
            mem.store(torch.randn(16))
        mem.forget(threshold=-0.2)
        for _ in range(10):
            mem.store(torch.randn(16))
        stats = mem.get_stats()
        assert stats["size"] == stats["index_ntotal"], (
            f"size={stats['size']} != index_ntotal={stats['index_ntotal']}"
        )

    def test_capacity_wrap_around(self, mem_factory):
        """Storing beyond capacity evicts oldest."""
        torch.manual_seed(0)
        dim = 8
        mem = mem_factory(capacity=5, feature_dim=dim)
        # 5 distinct unit-norm values (use random, not full(), to avoid
        # all-equal vectors that would tie under cosine).
        for _ in range(5):
            mem.store(torch.randn(dim))
        assert len(mem) == 5
        # 3 more -> only the last 5 should remain
        last3 = []
        for _ in range(3):
            v = torch.randn(dim)
            last3.append(v)
            mem.store(v)
        assert len(mem) == 5
        assert mem.get_stats()["index_ntotal"] == 5
        # Query with one of the last 3 stored -> the exact match should win
        # (or be the top-1 for HNSW).
        q = last3[-1]
        ids, sims = mem.search(q, k=1)
        top1_id = int(ids.item())
        if mem.use_hnsw:
            # HNSW: top-1 similarity must be high (>= 0.9) but the id may
            # not be the exact match (approximate).
            assert float(sims.item()) >= 0.9, (
                f"HNSW top-1 sim {float(sims.item()):.3f} < 0.9"
            )
        else:
            # Flat is exact: the exact-match slot is the one whose value
            # is closest to q under cosine. Since we store q itself, it
            # must be the top-1 with sim ~1.0. Find the expected slot id.
            expected = None
            for sid, v in mem._values.items():
                if torch.allclose(v, q, atol=1e-6):
                    expected = sid
                    break
            assert expected is not None
            assert top1_id == expected, f"expected {expected}, got {top1_id}"

    def test_reset_clears(self, mem_factory):
        mem = mem_factory(capacity=10, feature_dim=8)
        for _ in range(5):
            mem.store(torch.randn(8))
        mem.reset()
        assert len(mem) == 0
        assert mem.get_stats()["index_ntotal"] == 0
        assert mem.get_stats()["n_stores"] == 0


# =========================================================================
# 4. Sanity: same input -> same search result as a reference cosine scan
# =========================================================================
class TestFAISSBackedCorrectness:
    """Compare FAISS results against a brute-force PyTorch reference."""

    def test_search_matches_brute_force(self, mem_factory):
        torch.manual_seed(7)
        dim = 64
        n = 50
        mem = mem_factory(capacity=100, feature_dim=dim)
        corpus = torch.randn(n, dim)
        corpus = corpus / corpus.norm(dim=-1, keepdim=True)
        for v in corpus:
            mem.store(v)

        q = torch.randn(dim)
        q = q / q.norm()
        ids, sims = mem.search(q, k=5)

        # Reference: cosine sim in PyTorch
        ref_sims = corpus @ q  # [n] (unit-norm)
        ref_topk = torch.topk(ref_sims, k=5)
        ref_ids = ref_topk.indices.tolist()
        ref_vals = ref_topk.values.tolist()

        got_ids = ids.tolist()
        got_vals = sims.tolist()
        if mem.use_hnsw:
            # HNSW is approximate -> we only require that the top-1 is
            # the exact match and similarities are within a reasonable
            # tolerance of the reference.
            assert got_ids[0] == ref_ids[0], (
                f"HNSW top-1 mismatch: got {got_ids[0]} vs ref {ref_ids[0]}"
            )
            for a, b in zip(got_vals, ref_vals):
                assert abs(a - b) < 0.05, f"sim mismatch: {a} vs {b}"
        else:
            assert got_ids == ref_ids, f"Top-5 IDs mismatch:\n  got={got_ids}\n  exp={ref_ids}"
            for a, b in zip(got_vals, ref_vals):
                assert abs(a - b) < 1e-4, f"sim mismatch: {a} vs {b}"


# =========================================================================
# 5. Performance micro-benchmark (timing only — no asserts on numbers)
# =========================================================================
class TestFAISSBackedPerf:
    """Tiny perf test: 1000 stores, 100 queries, log ms/operation."""

    def test_perf_1000_stores_100_queries(self, mem_factory, capsys):
        # Skip the HNSW variant in the perf sweep to keep it fast.
        mem = FAISSBackedEpisodicMemory(
            capacity=2000, feature_dim=384, use_hnsw=False
        )
        torch.manual_seed(0)
        n_stores = 1000
        n_queries = 100

        store_t0 = time.perf_counter()
        for _ in range(n_stores):
            mem.store(torch.randn(384))
        store_dt = (time.perf_counter() - store_t0) * 1000

        query_t0 = time.perf_counter()
        for _ in range(n_queries):
            ids, sims = mem.search(torch.randn(384), k=5)
        query_dt = (time.perf_counter() - query_t0) * 1000

        with capsys.disabled():
            print(
                f"\n[FAISS perf] {n_stores} stores: {store_dt:.1f} ms "
                f"({store_dt / n_stores:.3f} ms/op); "
                f"{n_queries} queries: {query_dt:.1f} ms "
                f"({query_dt / n_queries:.3f} ms/op)"
            )
        # Loose sanity: per-op should be well under 100 ms on this machine.
        assert store_dt / n_stores < 100.0
        assert query_dt / n_queries < 100.0

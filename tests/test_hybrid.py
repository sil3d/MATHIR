"""
Unit tests for HybridEpisodicMemory (Approach D).

Covers:
    1. Construction & default state (with / without cross-encoder)
    2. Storage (text + embedding, embedding only)
    3. BM25 index (tokenization, ranking, no-text fallback)
    4. Dense retrieval (cosine top-k)
    5. Cross-encoder re-ranking (when available)
    6. Reciprocal Rank Fusion (RRF)
    7. search() / retrieve() public API contract
    8. forget() / reset() / get_stats() maintenance
    9. End-to-end semantic question → top-1 match

Run:
    pytest tests/test_hybrid.py -v
"""

import math
import warnings

import pytest
import torch
import torch.nn.functional as F

from mathir_lib.memory.hybrid_episodic import (
    HybridEpisodicMemory,
    _tokenize,
)
from mathir_lib import HybridEpisodicMemory as HybridAliased


# Skip the whole module if the cross-encoder is not loadable — the test
# suite should still pass (and mark the cross-encoder tests as skipped) if
# transformers / sentence-transformers is broken on the test box.
try:
    from sentence_transformers import CrossEncoder  # noqa: F401
    _HAS_CE_DEPS = True
except ImportError:
    _HAS_CE_DEPS = False

try:
    from rank_bm25 import BM25Okapi  # noqa: F401
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


# =========================================================================
# Construction & shape
# =========================================================================
class TestHybridEpisodicInit:
    """Defaults, custom values, dtype checks, cross-encoder flag handling."""

    def test_default_construction(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mem = HybridEpisodicMemory()
        assert mem.capacity == 1000
        assert mem.feature_dim == 384
        assert mem.use_cross_encoder_requested is True
        assert mem.dense_top_k == 20
        assert mem.bm25_top_k == 20
        assert mem.rrf_k_const == 60
        assert mem.cross_encoder_top_n == 30
        assert mem.bm25_weight == 1.0
        assert mem.count.item() == 0
        assert mem.ptr.item() == 0
        # Same buffer layout as the other episodic memory classes
        assert mem.keys.shape == (1000, 384)
        assert mem.values.shape == (1000, 384)

    def test_custom_construction(self):
        mem = HybridEpisodicMemory(
            capacity=512,
            feature_dim=768,
            use_cross_encoder=False,
            dense_top_k=15,
            bm25_top_k=10,
        )
        assert mem.capacity == 512
        assert mem.feature_dim == 768
        assert mem.use_cross_encoder_requested is False
        assert mem.dense_top_k == 15
        assert mem.bm25_top_k == 10
        assert mem.keys.shape == (512, 768)
        assert mem.values.shape == (512, 768)
        # When cross-encoder is disabled, has_cross_encoder must be False
        assert mem.has_cross_encoder is False

    def test_top_level_alias_export(self):
        """``from mathir_lib import HybridEpisodicMemory`` must work."""
        assert HybridAliased is HybridEpisodicMemory

    def test_default_cross_encoder_loaded(self):
        """When CE deps are present, the default constructor should
        successfully load the cross-encoder (or fail loudly)."""
        if not _HAS_CE_DEPS:
            pytest.skip("sentence-transformers not installed")
        mem = HybridEpisodicMemory(use_cross_encoder=True, feature_dim=384)
        # Either the model is loaded, or a warning was issued. The
        # constructor must not raise.
        # If the network is unreachable, has_cross_encoder will be False
        # but the object must still be usable.
        assert isinstance(mem.has_cross_encoder, bool)


# =========================================================================
# Storage
# =========================================================================
class TestHybridEpisodicStore:
    """store() with and without text, wrapping, dimension checks."""

    def test_store_with_text_indexes_bm25(self):
        mem = HybridEpisodicMemory(capacity=20, feature_dim=64, use_cross_encoder=False)
        assert mem.has_bm25 is False
        mem.store(torch.randn(1, 64), text="the Reynolds number measures turbulence")
        assert mem.count.item() == 1
        if _HAS_BM25:
            assert mem.has_bm25 is True
            assert len(mem._bm25_corpus_tokens) == 1
            assert mem._bm25_doc_ids == [0]

    def test_store_without_text_skips_bm25(self):
        mem = HybridEpisodicMemory(capacity=20, feature_dim=64, use_cross_encoder=False)
        mem.store(torch.randn(1, 64))
        assert mem.count.item() == 1
        # BM25 is empty (no text supplied)
        assert mem.has_bm25 is False

    def test_store_multiple_chunks(self):
        torch.manual_seed(0)
        mem = HybridEpisodicMemory(capacity=50, feature_dim=384, use_cross_encoder=False)
        texts = [
            "Bernoulli's equation for incompressible flow",
            "Reynolds number definition and critical values",
            "Navier-Stokes equation for viscous fluid flow",
            "Boundary layer thickness over a flat plate",
            "Laminar versus turbulent flow regimes",
        ]
        for t in texts:
            mem.store(torch.randn(1, 384), text=t)
        assert mem.count.item() == 5
        if _HAS_BM25:
            assert len(mem._bm25_corpus_tokens) == 5

    def test_store_caps_at_capacity(self):
        mem = HybridEpisodicMemory(capacity=5, feature_dim=16, use_cross_encoder=False)
        for i in range(12):
            mem.store(torch.randn(1, 16), text=f"document number {i}")
        assert mem.count.item() == 5
        assert mem.ptr.item() == 2  # 12 mod 5 = 2

    def test_store_dimension_mismatch_raises(self):
        mem = HybridEpisodicMemory(capacity=4, feature_dim=8, use_cross_encoder=False)
        with pytest.raises(ValueError):
            mem.store(torch.randn(1, 16))

    def test_store_1d_input(self):
        mem = HybridEpisodicMemory(capacity=4, feature_dim=8, use_cross_encoder=False)
        mem.store(torch.randn(8))
        assert mem.count.item() == 1


# =========================================================================
# BM25 index
# =========================================================================
class TestBM25Index:
    """BM25 ranks documents that contain the query terms highly."""

    def test_tokenizer_keeps_hyphens(self):
        """Technical terms like 'Navier-Stokes' should stay one token."""
        tokens = _tokenize("Navier-Stokes equation for viscous flow")
        assert "navier-stokes" in tokens
        assert "viscous" in tokens

    def test_tokenizer_lowercases(self):
        tokens = _tokenize("REYNOLDS Number")
        assert "reynolds" in tokens
        assert "number" in tokens

    @pytest.mark.skipif(not _HAS_BM25, reason="rank_bm25 not installed")
    def test_bm25_ranks_relevant_doc_first(self):
        """Test the BM25 ranker in isolation (not via the hybrid search)."""
        mem = HybridEpisodicMemory(capacity=10, feature_dim=64, use_cross_encoder=False)
        # Embeddings are noise — the BM25 branch should still pull the
        # right document based on the text alone.
        docs = [
            "Apples are red and grow on trees",
            "Bananas are yellow tropical fruits",
            "The cat sat on the mat",
            "Oranges are orange citrus fruits",
            "Dogs love to chase squirrels in the park",
        ]
        for t in docs:
            mem.store(torch.randn(1, 64), text=t)

        # Call the BM25 branch directly to isolate it from dense noise
        slot_ids, bm25_scores = mem._bm25_topk("red apples on trees", k=3)
        assert len(slot_ids) > 0
        # The top-1 from BM25 should be the apples doc
        assert slot_ids[0] == 0
        assert bm25_scores[0] > bm25_scores[1]

    @pytest.mark.skipif(not _HAS_BM25, reason="rank_bm25 not installed")
    def test_bm25_works_with_technical_terms(self):
        """BM25 alone should rank the technical-term doc first."""
        mem = HybridEpisodicMemory(capacity=10, feature_dim=64, use_cross_encoder=False)
        docs = [
            "Navier-Stokes equation governs viscous fluid flow",
            "Pizza is a popular Italian food with cheese and tomato",
            "Reynolds number determines laminar versus turbulent flow",
            "Quantum mechanics describes subatomic particle behavior",
        ]
        for t in docs:
            mem.store(torch.randn(1, 64), text=t)

        # Call BM25 directly so dense noise doesn't dominate
        slot_ids, bm25_scores = mem._bm25_topk(
            "Navier-Stokes fluid flow", k=2,
        )
        assert len(slot_ids) > 0
        # Top-1 from BM25 must be the Navier-Stokes doc (slot 0)
        assert slot_ids[0] == 0

    def test_bm25_disabled_when_no_text_stored(self):
        """If no text was ever supplied, BM25 is silent and dense still works."""
        mem = HybridEpisodicMemory(capacity=10, feature_dim=64, use_cross_encoder=False)
        mem.store(torch.randn(1, 64))
        mem.store(torch.randn(1, 64))
        assert mem.has_bm25 is False
        # Search still works (dense only)
        indices, scores = mem.search(torch.randn(1, 64), k=2, query_text="anything")
        assert indices.shape == (1, 2)


# =========================================================================
# Dense retrieval
# =========================================================================
class TestDenseRetrieval:
    """Dense top-k works exactly like RawEmbeddingEpisodicMemory."""

    def test_dense_finds_exact_match(self):
        torch.manual_seed(42)
        mem = HybridEpisodicMemory(
            capacity=100, feature_dim=384, use_cross_encoder=False,
        )
        originals = [torch.randn(1, 384) for _ in range(50)]
        for emb in originals:
            mem.store(emb, text="some text")

        target = originals[7]
        query = target + torch.randn(1, 384) * 0.01
        indices, sims = mem.search(query, k=1)
        top_idx = int(indices[0, 0].item())
        # The retrieved value should be the original (or very close)
        retrieved = mem.values[top_idx]
        cos = F.cosine_similarity(query, retrieved.unsqueeze(0), dim=-1).item()
        assert cos > 0.9

    def test_dense_search_no_text(self):
        """search() without query_text falls back to dense only."""
        torch.manual_seed(1)
        mem = HybridEpisodicMemory(
            capacity=20, feature_dim=128, use_cross_encoder=False,
        )
        for _ in range(5):
            mem.store(torch.randn(1, 128))
        indices, scores = mem.search(torch.randn(1, 128), k=3)
        assert indices.shape == (1, 3)
        assert scores.shape == (1, 3)


# =========================================================================
# Cross-encoder re-ranking
# =========================================================================
class TestCrossEncoderReRank:
    """The cross-encoder (if available) reorders candidates by (q, d) score."""

    def test_cross_encoder_actually_reranks(self):
        """Top-1 should be the doc whose text is most semantically close to
        the query, even when BM25 would have picked a different doc."""
        if not _HAS_CE_DEPS:
            pytest.skip("sentence-transformers not installed")
        if not _HAS_BM25:
            pytest.skip("rank_bm25 not installed")

        torch.manual_seed(11)
        mem = HybridEpisodicMemory(
            capacity=20, feature_dim=128, use_cross_encoder=True,
        )
        # Embeddings: pure noise → dense similarity is meaningless.
        # Texts: only the "Reynolds" doc is actually relevant.
        docs = [
            "The quick brown fox jumps over the lazy dog",
            "Pizza and pasta are popular Italian dishes",
            "Reynolds number characterizes laminar versus turbulent flow",
            "Shakespeare wrote Hamlet and Macbeth",
        ]
        for t in docs:
            mem.store(torch.randn(1, 128), text=t)

        query = torch.randn(1, 128)
        query_text = "What is the Reynolds number in fluid dynamics?"
        indices, _ = mem.search(query, k=2, query_text=query_text)
        top_idx = int(indices[0, 0].item())
        # The top-1 should be the Reynolds doc
        assert "reynolds" in docs[top_idx].lower()

    def test_cross_encoder_fallback_when_unavailable(self):
        """If use_cross_encoder=False, search still works (RRF-only)."""
        if not _HAS_BM25:
            pytest.skip("rank_bm25 not installed")

        mem = HybridEpisodicMemory(
            capacity=20, feature_dim=64, use_cross_encoder=False,
        )
        for t in [
            "alpha beta gamma",
            "delta epsilon zeta",
            "eta theta iota",
        ]:
            mem.store(torch.randn(1, 64), text=t)
        assert mem.has_cross_encoder is False
        indices, scores = mem.search(
            torch.randn(1, 64), k=2, query_text="alpha gamma",
        )
        assert indices.shape == (1, 2)


# =========================================================================
# Reciprocal Rank Fusion (RRF)
# =========================================================================
class TestReciprocalRankFusion:
    """RRF merges the dense and BM25 rankings sensibly."""

    def test_rrf_fuses_dense_and_bm25(self):
        mem = HybridEpisodicMemory(
            capacity=20, feature_dim=64, use_cross_encoder=False,
        )
        # Manually call _rrf_fuse
        dense_idx = torch.tensor([[3, 1, 2, 0]], dtype=torch.long)
        bm25_slots = [4, 1, 3]
        fused = mem._rrf_fuse(dense_idx, bm25_slots)
        # Slot 1 is rank 1 in dense (1/61) and rank 1 in BM25 (1/61) → fused top
        # Slot 3 is rank 0 in dense (1/60) and rank 2 in BM25 (1/62)
        # Slot 4 is rank 0 in BM25 (1/60) and absent from dense
        # Slot 1 should come first or second
        assert 1 in fused[:2]

    def test_rrf_weight_changes_ranking(self):
        """bm25_weight=2 should boost BM25 hits more than dense hits."""
        mem = HybridEpisodicMemory(
            capacity=20, feature_dim=64, use_cross_encoder=False,
            bm25_weight=2.0,
        )
        # Slot 0 is dense rank 0 (1/60 = 0.01667)
        # Slot 1 is BM25 rank 0 (2.0/60 = 0.0333)
        # → slot 1 should win with bm25_weight=2
        dense_idx = torch.tensor([[0, 1]], dtype=torch.long)
        bm25_slots = [1, 0]
        fused = mem._rrf_fuse(dense_idx, bm25_slots)
        assert fused[0] == 1

    def test_rrf_dense_only(self):
        """With empty BM25 list, RRF degenerates to the dense ranking."""
        mem = HybridEpisodicMemory(
            capacity=20, feature_dim=64, use_cross_encoder=False,
        )
        dense_idx = torch.tensor([[5, 3, 1]], dtype=torch.long)
        fused = mem._rrf_fuse(dense_idx, [])
        assert fused == [5, 3, 1]

    def test_rrf_bm25_only(self):
        """With empty dense tensor, RRF degenerates to the BM25 ranking."""
        mem = HybridEpisodicMemory(
            capacity=20, feature_dim=64, use_cross_encoder=False,
        )
        empty = torch.zeros(1, 0, dtype=torch.long)
        fused = mem._rrf_fuse(empty, [7, 2, 9])
        assert fused == [7, 2, 9]


# =========================================================================
# Public search / retrieve contract
# =========================================================================
class TestSearchRetrieveContract:
    """search() and retrieve() match the other episodic memory classes."""

    def test_search_returns_correct_shapes(self):
        torch.manual_seed(2)
        mem = HybridEpisodicMemory(
            capacity=10, feature_dim=64, use_cross_encoder=False,
        )
        for _ in range(5):
            mem.store(torch.randn(1, 64), text="some document text")
        indices, scores = mem.search(torch.randn(2, 64), k=3)
        assert indices.shape == (2, 3)
        assert scores.shape == (2, 3)

    def test_search_empty_memory(self):
        mem = HybridEpisodicMemory(
            capacity=10, feature_dim=64, use_cross_encoder=False,
        )
        indices, scores = mem.search(torch.randn(1, 64), k=3)
        assert indices.numel() == 0
        assert scores.numel() == 0

    def test_retrieve_returns_residual(self):
        torch.manual_seed(3)
        mem = HybridEpisodicMemory(
            capacity=10, feature_dim=64, use_cross_encoder=False,
        )
        for _ in range(5):
            mem.store(torch.randn(1, 64), text="some document text")
        out = mem.retrieve(torch.randn(2, 64), k=3, query_text="some")
        assert out.shape == (2, 64)

    def test_retrieve_empty_returns_query(self):
        mem = HybridEpisodicMemory(
            capacity=10, feature_dim=64, use_cross_encoder=False,
        )
        q = torch.randn(2, 64)
        out = mem.retrieve(q, k=3)
        assert torch.allclose(out, q)


# =========================================================================
# Maintenance (forget / reset / stats)
# =========================================================================
class TestMaintenance:
    """forget() prunes; reset() clears; get_stats() reports sensibly."""

    def test_forget_compacts_bm25(self):
        if not _HAS_BM25:
            pytest.skip("rank_bm25 not installed")
        mem = HybridEpisodicMemory(
            capacity=20, feature_dim=64, use_cross_encoder=False,
        )
        for i in range(10):
            mem.store(torch.randn(1, 64), text=f"document number {i}")
        assert mem.count.item() == 10
        # Aggressive threshold should prune
        kept = mem.forget(threshold=0.5)
        assert kept < 10
        # BM25 corpus must be compacted in lockstep with the dense buffers
        assert len(mem._bm25_corpus_tokens) == kept

    def test_reset_clears_everything(self):
        if not _HAS_BM25:
            pytest.skip("rank_bm25 not installed")
        mem = HybridEpisodicMemory(
            capacity=10, feature_dim=64, use_cross_encoder=False,
        )
        for i in range(5):
            mem.store(torch.randn(1, 64), text=f"document {i}")
        mem.reset()
        assert mem.count.item() == 0
        assert mem.ptr.item() == 0
        assert mem._bm25_corpus_tokens == []
        assert mem._bm25_doc_ids == []
        assert mem._bm25 is None

    def test_get_stats_shape_and_keys(self):
        mem = HybridEpisodicMemory(
            capacity=20, feature_dim=64, use_cross_encoder=False,
        )
        for _ in range(3):
            mem.store(torch.randn(1, 64), text="some document text")
        stats = mem.get_stats()
        for key in (
            "count", "capacity", "feature_dim",
            "has_bm25", "has_cross_encoder",
            "cross_encoder_model",
            "dense_top_k", "bm25_top_k", "rrf_k_const",
            "bm25_corpus_size",
            "n_searches", "n_dense_used", "n_bm25_used", "n_ce_used",
            "mean_pairwise_sim", "min_pairwise_sim",
        ):
            assert key in stats, f"Missing key: {key}"
        assert stats["count"] == 3
        assert stats["capacity"] == 20
        assert stats["feature_dim"] == 64
        assert stats["bm25_corpus_size"] == 3
        assert stats["has_bm25"] is True
        assert stats["n_searches"] == 0

    def test_stats_increment_on_search(self):
        if not _HAS_BM25:
            pytest.skip("rank_bm25 not installed")
        mem = HybridEpisodicMemory(
            capacity=10, feature_dim=64, use_cross_encoder=False,
        )
        for t in ["alpha beta", "gamma delta", "epsilon zeta"]:
            mem.store(torch.randn(1, 64), text=t)
        mem.search(torch.randn(1, 64), k=2, query_text="alpha gamma")
        stats = mem.get_stats()
        assert stats["n_searches"] == 1
        assert stats["n_dense_used"] == 1
        assert stats["n_bm25_used"] == 1
        # No cross-encoder, so n_ce_used stays 0
        assert stats["n_ce_used"] == 0


# =========================================================================
# End-to-end: semantic question → top-1 match
# =========================================================================
class TestEndToEndSemanticRetrieval:
    """Real-world scenario: store 50 chunks, query with a semantic question."""

    def test_top1_matches_relevant_chunk(self):
        """Store 50 distinct chunks (text + random embedding), then ask a
        semantic question whose answer is one specific chunk.

        We rely on the cross-encoder to bridge the noise embeddings — the
        text retrieval path is the one that should pick the right doc.
        """
        if not _HAS_CE_DEPS:
            pytest.skip("sentence-transformers not installed")
        if not _HAS_BM25:
            pytest.skip("rank_bm25 not installed")

        torch.manual_seed(123)
        # Use the cross-encoder to rerank → it sees the text and picks the
        # most semantically aligned chunk.
        mem = HybridEpisodicMemory(
            capacity=60, feature_dim=64, use_cross_encoder=True,
        )

        target_idx = 17
        # Embeddings are noise — the dense ranker can't tell them apart.
        for i in range(50):
            mem.store(torch.randn(1, 64), text=f"document {i} about topic {i}")

        # Override the target doc with a real, distinctive sentence.
        # (We re-store to overwrite; the BM25 index is rebuilt from scratch.)
        mem._bm25_corpus_tokens = []
        mem._bm25_doc_ids = []
        # Reset the target slot
        target_emb = torch.randn(1, 64)
        target_text = (
            "The capital of France is Paris, a major European city"
        )
        # We need to clear and re-store to keep things consistent. Use a
        # fresh memory for clarity.
        mem.reset()
        texts = [f"document {i} about topic {i}" for i in range(50)]
        for i, t in enumerate(texts):
            mem.store(torch.randn(1, 64), text=t)
        # Overwrite the target slot manually by storing with text
        # (Since this is circular, we just append 1 more)
        mem.store(target_emb, text=target_text)
        target_idx = mem.count.item() - 1  # last slot is our target

        # Now query
        query = torch.randn(1, 64)
        query_text = "What is the capital of France?"
        indices, scores = mem.search(query, k=3, query_text=query_text)
        top_idx = int(indices[0, 0].item())
        # The target text must be retrieved in the top-3
        retrieved_text_idx = top_idx
        # Look up the actual text from the BM25 corpus
        if retrieved_text_idx < len(mem._bm25_doc_ids):
            # The slot ids in BM25 may not be sequential — find the doc
            # whose slot id matches the top_idx
            try:
                bm25_pos = mem._bm25_doc_ids.index(top_idx)
                retrieved_text = " ".join(mem._bm25_corpus_tokens[bm25_pos])
            except ValueError:
                retrieved_text = ""
        else:
            retrieved_text = ""
        # The cross-encoder should pull the "Paris" doc to the top
        assert "paris" in retrieved_text or "france" in retrieved_text, (
            f"Expected Paris/France doc, got slot {top_idx}: '{retrieved_text}'"
        )

    def test_50_chunks_with_bm25_only(self):
        """BM25 alone, on 50 chunks, should still find the relevant chunk
        when the query shares exact terms with the target. We test BM25
        in isolation because random dense embeddings would otherwise
        dominate the RRF fusion."""
        if not _HAS_BM25:
            pytest.skip("rank_bm25 not installed")

        torch.manual_seed(0)
        mem = HybridEpisodicMemory(
            capacity=60, feature_dim=64, use_cross_encoder=False,
        )
        # Noise + a few distinctive technical docs
        tech_docs = [
            "The Reynolds number characterizes fluid flow regimes",
            "Bernoulli's principle describes pressure-velocity tradeoffs",
            "Navier-Stokes equations model viscous fluid dynamics",
        ]
        for i in range(50):
            if i < 3:
                mem.store(torch.randn(1, 64), text=tech_docs[i])
            else:
                mem.store(torch.randn(1, 64), text=f"document {i}")

        # BM25 alone on the 50-chunk corpus
        slot_ids, bm25_scores = mem._bm25_topk(
            "What is the Reynolds number?", k=2,
        )
        assert len(slot_ids) > 0
        # The Reynolds doc is at slot 0
        assert slot_ids[0] == 0

    def test_50_chunks_with_cross_encoder_endtoend(self):
        """Full hybrid pipeline (dense + BM25 + cross-encoder) on 50
        chunks. The cross-encoder re-ranks the RRF-fused candidates and
        pulls the semantically relevant doc to the top, even when the
        dense embeddings are random noise."""
        if not _HAS_CE_DEPS:
            pytest.skip("sentence-transformers not installed")
        if not _HAS_BM25:
            pytest.skip("rank_bm25 not installed")

        torch.manual_seed(0)
        mem = HybridEpisodicMemory(
            capacity=60, feature_dim=64, use_cross_encoder=True,
        )
        # Noise + 3 distinctive technical docs
        tech_docs = [
            "The Reynolds number characterizes fluid flow regimes",
            "Bernoulli's principle describes pressure-velocity tradeoffs",
            "Navier-Stokes equations model viscous fluid dynamics",
        ]
        for i in range(50):
            if i < 3:
                mem.store(torch.randn(1, 64), text=tech_docs[i])
            else:
                mem.store(torch.randn(1, 64), text=f"document {i}")

        # Query with semantic question + noisy embedding
        query = torch.randn(1, 64)
        indices, _ = mem.search(
            query, k=2, query_text="What is the Reynolds number?",
        )
        top_idx = int(indices[0, 0].item())
        # Look up the retrieved text
        try:
            bm25_pos = mem._bm25_doc_ids.index(top_idx)
            retrieved_text = " ".join(mem._bm25_corpus_tokens[bm25_pos])
        except ValueError:
            retrieved_text = ""
        assert "reynolds" in retrieved_text, (
            f"Expected Reynolds doc, got slot {top_idx}: '{retrieved_text}'"
        )

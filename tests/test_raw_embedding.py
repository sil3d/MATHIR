"""
Unit tests for the RawEmbeddingEpisodicMemory class (Approach A).

Approach A bypasses the 272 → 64 projection that ``EpisodicMemory`` performs
on the way to the key-value store. The full raw embedding (e.g. 384-dim)
is kept as both key and value, so cosine similarity runs on every dimension
the embedding model produced.

Run:
    pytest tests/test_raw_embedding.py -v
"""

import math

import pytest
import torch
import torch.nn.functional as F

from mathir_lib.memory.raw_episodic import RawEmbeddingEpisodicMemory
from mathir_lib import RawEmbeddingEpisodicMemory as RawAliased


# =========================================================================
# Construction & shape
# =========================================================================
class TestRawEmbeddingEpisodicInit:
    """Defaults, custom values, dtype checks, projection flag handling."""

    def test_default_construction(self):
        mem = RawEmbeddingEpisodicMemory()
        assert mem.capacity == 1000
        assert mem.embedding_dim == 384
        assert mem.projection is False
        assert mem.proj_dim == 384  # falls back to embedding_dim when projection=False
        assert mem.count.item() == 0
        assert mem.ptr.item() == 0
        # When projection=False, keys buffer is at full embedding_dim
        assert mem.keys.shape == (1000, 384)
        assert mem.values.shape == (1000, 384)

    def test_custom_construction(self):
        mem = RawEmbeddingEpisodicMemory(capacity=512, embedding_dim=768)
        assert mem.capacity == 512
        assert mem.embedding_dim == 768
        assert mem.keys.shape == (512, 768)
        assert mem.values.shape == (512, 768)

    def test_projection_true_creates_mlp(self):
        mem = RawEmbeddingEpisodicMemory(
            capacity=100, embedding_dim=384, projection=True, proj_dim=64, hidden_dim=128
        )
        assert mem.projection is True
        assert mem.proj_dim == 64
        # Keys buffer is at proj_dim (smaller) when projection=True;
        # values stay at embedding_dim (lossless).
        assert mem.keys.shape == (100, 64)
        assert mem.values.shape == (100, 384)
        # MLP must have Linear layers with the right dimensions
        assert isinstance(mem.key_encoder, torch.nn.Sequential)
        linears = [m for m in mem.key_encoder.modules() if isinstance(m, torch.nn.Linear)]
        assert linears[0].in_features == 384
        assert linears[0].out_features == 128
        assert linears[-1].out_features == 64

    def test_projection_false_uses_identity(self):
        mem = RawEmbeddingEpisodicMemory(projection=False)
        assert isinstance(mem.key_encoder, torch.nn.Identity)

    def test_top_level_alias_export(self):
        """``from mathir_lib import RawEmbeddingEpisodicMemory`` must work."""
        assert RawAliased is RawEmbeddingEpisodicMemory


# =========================================================================
# Storage
# =========================================================================
class TestRawEmbeddingEpisodicStore:
    """store() writes raw embeddings, increments count, wraps around capacity."""

    def test_store_increments_count(self):
        mem = RawEmbeddingEpisodicMemory(capacity=10, embedding_dim=64)
        mem.store(torch.randn(1, 64))
        assert mem.count.item() == 1
        assert mem.ptr.item() == 1

    def test_store_50_random_increments_count(self):
        torch.manual_seed(0)
        mem = RawEmbeddingEpisodicMemory(capacity=100, embedding_dim=384)
        for _ in range(50):
            mem.store(torch.randn(1, 384))
        assert mem.count.item() == 50
        assert mem.ptr.item() == 50

    def test_store_caps_at_capacity(self):
        mem = RawEmbeddingEpisodicMemory(capacity=5, embedding_dim=16)
        for _ in range(12):
            mem.store(torch.randn(1, 16))
        # count should saturate at capacity
        assert mem.count.item() == 5
        # ptr should wrap around (12 mod 5 = 2)
        assert mem.ptr.item() == 2

    def test_store_writes_to_keys_and_values(self):
        torch.manual_seed(7)
        mem = RawEmbeddingEpisodicMemory(capacity=4, embedding_dim=8)
        x = torch.randn(1, 8)
        mem.store(x)
        # Same value written to keys[0] and values[0]
        assert torch.allclose(mem.keys[0], x.squeeze(0))
        assert torch.allclose(mem.values[0], x.squeeze(0))

    def test_store_dimension_mismatch_raises(self):
        mem = RawEmbeddingEpisodicMemory(capacity=4, embedding_dim=8)
        with pytest.raises(ValueError):
            mem.store(torch.randn(1, 16))

    def test_store_1d_input(self):
        mem = RawEmbeddingEpisodicMemory(capacity=4, embedding_dim=8)
        mem.store(torch.randn(8))
        assert mem.count.item() == 1


# =========================================================================
# Search & retrieval — the main contract
# =========================================================================
class TestRawEmbeddingEpisodicRetrieval:
    """Top-1 should be the original (cosine > 0.9)."""

    def test_query_returns_original_with_cosine_above_0_9(self):
        torch.manual_seed(42)
        mem = RawEmbeddingEpisodicMemory(capacity=200, embedding_dim=384)

        # Store 50 random embeddings
        originals = [torch.randn(1, 384) for _ in range(50)]
        for emb in originals:
            mem.store(emb)

        # Pick one and query with a noisy version (cosine should still be > 0.9)
        target = originals[7]
        noise = torch.randn(1, 384) * 0.05  # small noise
        query = target + noise

        indices, sims = mem.search(query, k=1)
        top_idx = indices[0, 0].item()
        top_sim = sims[0, 0].item()

        # The retrieved value must be the original
        retrieved = mem.values[top_idx]
        assert torch.allclose(retrieved, target.squeeze(0), atol=1e-5)
        # Cosine similarity between query and retrieved should be very high
        cos = F.cosine_similarity(query, retrieved.unsqueeze(0), dim=-1).item()
        assert cos > 0.9, f"Expected cos > 0.9, got {cos}"
        # The internal search similarity must also be high
        assert top_sim > 0.9, f"Expected search sim > 0.9, got {top_sim}"

    def test_retrieve_returns_residual_shape(self):
        torch.manual_seed(1)
        mem = RawEmbeddingEpisodicMemory(capacity=10, embedding_dim=384)
        for _ in range(5):
            mem.store(torch.randn(1, 384))
        q = torch.randn(2, 384)  # batch=2
        out = mem.retrieve(q, k=3)
        assert out.shape == (2, 384)

    def test_retrieve_empty_returns_query(self):
        """With nothing stored, retrieve() must return the query unchanged."""
        torch.manual_seed(2)
        mem = RawEmbeddingEpisodicMemory(capacity=10, embedding_dim=384)
        q = torch.randn(2, 384)
        out = mem.retrieve(q, k=3)
        assert torch.allclose(out, q)

    def test_search_empty_returns_empty_tensors(self):
        mem = RawEmbeddingEpisodicMemory(capacity=10, embedding_dim=64)
        indices, sims = mem.search(torch.randn(1, 64), k=5)
        assert indices.numel() == 0
        assert sims.numel() == 0

    def test_search_topk_ordering(self):
        """Similarity must be non-increasing in the returned top-k."""
        torch.manual_seed(3)
        mem = RawEmbeddingEpisodicMemory(capacity=20, embedding_dim=384)
        for _ in range(10):
            mem.store(torch.randn(1, 384))
        q = torch.randn(1, 384)
        _, sims = mem.search(q, k=5)
        for i in range(sims.size(1) - 1):
            assert sims[0, i].item() >= sims[0, i + 1].item()


# =========================================================================
# Forgetting
# =========================================================================
class TestRawEmbeddingEpisodicForget:
    """forget() prunes low-similarity memories; reset() clears everything."""

    def test_forget_returns_count(self):
        torch.manual_seed(4)
        mem = RawEmbeddingEpisodicMemory(capacity=20, embedding_dim=384)
        for _ in range(5):
            mem.store(torch.randn(1, 384))
        kept = mem.forget(threshold=0.0)
        # all memories should be kept (random embeddings have non-zero pairwise sim)
        assert isinstance(kept, int)
        assert 1 <= kept <= 5

    def test_forget_on_empty(self):
        mem = RawEmbeddingEpisodicMemory(capacity=10, embedding_dim=64)
        kept = mem.forget(threshold=0.5)
        assert kept == 0

    def test_reset_clears_memory(self):
        torch.manual_seed(5)
        mem = RawEmbeddingEpisodicMemory(capacity=10, embedding_dim=64)
        for _ in range(3):
            mem.store(torch.randn(1, 64))
        mem.reset()
        assert mem.count.item() == 0
        assert mem.ptr.item() == 0
        assert mem.keys.abs().sum().item() == 0
        assert mem.values.abs().sum().item() == 0


# =========================================================================
# Stats & diagnostic helpers
# =========================================================================
class TestRawEmbeddingEpisodicStats:
    def test_get_usage(self):
        torch.manual_seed(6)
        mem = RawEmbeddingEpisodicMemory(capacity=20, embedding_dim=64)
        for _ in range(7):
            mem.store(torch.randn(1, 64))
        assert mem.get_usage() == 7

    def test_stats_shape_and_keys(self):
        torch.manual_seed(8)
        mem = RawEmbeddingEpisodicMemory(capacity=20, embedding_dim=64)
        for _ in range(4):
            mem.store(torch.randn(1, 64))
        stats = mem.get_stats()
        for key in ("count", "capacity", "embedding_dim", "projection", "proj_dim",
                    "mean_pairwise_sim", "min_pairwise_sim"):
            assert key in stats
        assert stats["count"] == 4
        assert stats["capacity"] == 20
        assert stats["embedding_dim"] == 64
        assert stats["projection"] is False
        # Pairwise sim between random Gaussian embeddings is ~0
        assert abs(stats["mean_pairwise_sim"]) < 0.5

    def test_stats_on_empty_memory(self):
        mem = RawEmbeddingEpisodicMemory(capacity=20, embedding_dim=64)
        stats = mem.get_stats()
        assert stats["count"] == 0
        assert stats["mean_pairwise_sim"] == 0.0


# =========================================================================
# Projection mode (projection=True)
# =========================================================================
class TestRawEmbeddingProjectionMode:
    """When projection=True, key is reduced to proj_dim but value stays raw."""

    def test_projection_recovers_original_in_value(self):
        torch.manual_seed(9)
        mem = RawEmbeddingEpisodicMemory(
            capacity=20, embedding_dim=384, projection=True, proj_dim=64
        )
        originals = [torch.randn(1, 384) for _ in range(10)]
        for emb in originals:
            mem.store(emb)

        # The projection MLP is randomly initialized so we cannot rely on
        # accurate retrieval quality here. Instead, verify that:
        #   (1) search() returns a valid index/similarity
        #   (2) the VALUES buffer still contains the original raw embedding
        #       (i.e. retrieval can be lossless on the value side)
        target = originals[3]
        query = target + torch.randn(1, 384) * 0.05
        indices, sims = mem.search(query, k=1)
        top_idx = indices[0, 0].item()
        assert 0 <= top_idx < 10
        assert -1.0 <= sims[0, 0].item() <= 1.0
        # Value buffer holds the original (lossless)
        assert torch.allclose(mem.values[top_idx], originals[top_idx].squeeze(0), atol=1e-5)

    def test_projection_keys_buffer_still_full_dim(self):
        """Even with projection, keys buffer holds the projected (low-dim) vector —
        verify the value buffer stays at the full dimension."""
        mem = RawEmbeddingEpisodicMemory(
            capacity=5, embedding_dim=384, projection=True, proj_dim=64
        )
        # Values MUST be at embedding_dim (lossless)
        assert mem.values.shape == (5, 384)


# =========================================================================
# Backward compatibility
# =========================================================================
class TestBackwardCompatibility:
    """The default config must NOT enable raw embedding (Approach A is opt-in)."""

    def test_default_config_disables_raw_embedding(self):
        from mathir_lib.config import get_default_config
        cfg = get_default_config()
        assert cfg["memory"]["use_raw_embedding"] is False
        assert cfg["memory"]["raw_embedding_dim"] == 384

    def test_plugin_v7_default_uses_standard_episodic(self):
        from mathir_lib.plugin_v7 import MATHIRPluginV7
        from mathir_lib.memory import EpisodicMemory
        plugin = MATHIRPluginV7(embedding_dim=4096)
        assert isinstance(plugin.episodic, EpisodicMemory)
        assert plugin.use_raw_embedding is False

    def test_plugin_v7_with_raw_flag(self):
        from mathir_lib.plugin_v7 import MATHIRPluginV7
        plugin = MATHIRPluginV7(
            embedding_dim=384,
            config={"memory": {"use_raw_embedding": True, "raw_embedding_dim": 384}},
        )
        assert isinstance(plugin.episodic, RawEmbeddingEpisodicMemory)
        assert plugin.use_raw_embedding is True

    def test_plugin_v7_perceive_with_raw_flag(self):
        """Full end-to-end perceive() should work with the raw memory active."""
        from mathir_lib.plugin_v7 import MATHIRPluginV7
        torch.manual_seed(11)
        plugin = MATHIRPluginV7(
            embedding_dim=384,
            config={"memory": {"use_raw_embedding": True, "raw_embedding_dim": 384}},
        )
        # Perceive one embedding to make sure the route through input_proj works
        emb = torch.randn(2, 384)
        out = plugin.perceive(emb)
        assert "enhanced_embedding" in out
        assert out["enhanced_embedding"].shape == (2, 384)
        # Store 2 distinct memories. The store() contract averages a batched
        # input into a single slot, so we call store() twice with single
        # memories to populate 2 slots.
        plugin.store({"embedding": emb[0].unsqueeze(0)})  # [1, 384] -> 1 slot
        plugin.store({"embedding": emb[1].unsqueeze(0)})  # [1, 384] -> 1 slot
        assert plugin.episodic.count.item() == 2
        memories = plugin.recall(emb, k=2)
        assert len(memories) == 2
        assert memories[0]["value"].shape == (384,)

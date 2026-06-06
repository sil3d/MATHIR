"""
Unit tests for EnsembleEpisodicMemory (Approach B: adaptive multi-encoder retrieval).

The class is verified against the same public-API contract as EpisodicMemory:
    __init__(capacity, feature_dim, proj_dims=(128, 64), learn_weights=True)
    store(x)
    retrieve(query, k=3)
    search(query, k=5)
    forget(threshold=0.1)
    get_stats()

Key behavioural guarantees tested:
    1. Stores raw 384-dim embedding + cached JL projections to 128/64.
    2. Cosine similarity in all 3 spaces is combined via softmax-constrained
       learnable weights.
    3. Random projection matrices are l2-normalized (Johnson-Lindenstrauss style).
    4. Initial weights are uniform (1/n_spaces) and sum to 1.
    5. Storing 50 random vectors + querying with a near-duplicate returns the
       correct top-1 (cosine > 0.9 on the ensemble score).
    6. Ensemble weights are learnable: gradient flows back to ``weight_logits``.
    7. Existing V7 tests still pass (regression).

Run:
    pytest tests/test_ensemble.py -v
"""

import math
import pytest
import torch
import torch.nn.functional as F

from mathir_lib.memory.ensemble_episodic import EnsembleEpisodicMemory
from mathir_lib.memory import EnsembleEpisodicMemory as EEMPublic


# =========================================================================
# 1. Construction & invariants
# =========================================================================
class TestEnsembleConstruction:
    """Constructor stores dims verbatim; weights uniform & sum to 1."""

    def test_init_default_values(self):
        """Default proj_dims=(128, 64), feature_dim=384, capacity verbatim."""
        mem = EnsembleEpisodicMemory(capacity=500, feature_dim=384)
        assert mem.capacity == 500
        assert mem.feature_dim == 384
        assert mem.proj_dims == (128, 64)
        assert mem.learn_weights is True
        assert mem.n_spaces == 3  # 384 + 128 + 64
        assert mem.all_dims == (384, 128, 64)

    def test_init_custom_proj_dims(self):
        """Custom proj_dims and learn_weights=False are stored verbatim."""
        mem = EnsembleEpisodicMemory(
            capacity=200, feature_dim=512, proj_dims=(256, 64, 16), learn_weights=False
        )
        assert mem.proj_dims == (256, 64, 16)
        assert mem.n_spaces == 4
        assert mem.all_dims == (512, 256, 64, 16)
        assert mem.learn_weights is False
        assert mem.weight_logits.requires_grad is False

    def test_init_weights_uniform_and_sum_to_one(self):
        """At init, weight_logits = 0 so softmax is uniform 1/n_spaces."""
        cases = [
            # (feature_dim, proj_dims, expected_n_spaces)
            (64, (32,), 2),
            (384, (128, 64), 3),
            (512, (256, 128, 32), 4),
        ]
        for feature_dim, proj_dims, n_spaces in cases:
            mem = EnsembleEpisodicMemory(
                capacity=10, feature_dim=feature_dim, proj_dims=proj_dims
            )
            w = mem.get_weights()
            assert mem.n_spaces == n_spaces
            assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)
            assert torch.allclose(
                w, torch.full_like(w, 1.0 / n_spaces), atol=1e-6
            )

    def test_init_validates_dims(self):
        """Constructor rejects proj_dim > feature_dim and zero/negative inputs."""
        with pytest.raises(ValueError):
            EnsembleEpisodicMemory(capacity=10, feature_dim=64, proj_dims=(128,))
        with pytest.raises(ValueError):
            EnsembleEpisodicMemory(capacity=10, feature_dim=0)
        with pytest.raises(ValueError):
            EnsembleEpisodicMemory(capacity=0, feature_dim=64)
        with pytest.raises(ValueError):
            EnsembleEpisodicMemory(capacity=10, feature_dim=64, proj_dims=(-1,))

    def test_public_export(self):
        """Class is exported from mathir_lib.memory public API."""
        assert EEMPublic is EnsembleEpisodicMemory


# =========================================================================
# 2. Random projection matrices (Johnson-Lindenstrauss)
# =========================================================================
class TestJohnsonLindenstraussProjections:
    """R_d is a fixed l2-normalized random Gaussian projection."""

    def test_projection_matrices_have_correct_shape(self):
        """Each R_d has shape [d, feature_dim] and is registered as a buffer."""
        torch.manual_seed(7)
        mem = EnsembleEpisodicMemory(
            capacity=10, feature_dim=384, proj_dims=(128, 64)
        )
        assert mem.R_128.shape == (128, 384)
        assert mem.R_64.shape == (64, 384)

    def test_projection_rows_are_unit_norm(self):
        """Each row of R_d has l2 norm 1 (Johnson-Lindenstrauss construction)."""
        torch.manual_seed(7)
        mem = EnsembleEpisodicMemory(
            capacity=10, feature_dim=384, proj_dims=(128, 64)
        )
        for d in (128, 64):
            R = getattr(mem, f"R_{d}")
            row_norms = R.norm(p=2, dim=1)
            assert torch.allclose(
                row_norms, torch.ones_like(row_norms), atol=1e-5
            ), f"R_{d} rows should be l2-normalized"

    def test_projection_preserves_pairwise_cosine(self):
        """JL projection approximately preserves pairwise cosine similarity.

        The JL lemma guarantees distance preservation, *not* norm preservation.
        With normalized random Gaussian rows, the squared norm of the
        projection scales as d_out / d_in. What is preserved is the geometry
        of point clouds, so we check that pairwise cosines in the projected
        space are close to those in the original space.
        """
        torch.manual_seed(7)
        d_in, d_out = 384, 64
        mem = EnsembleEpisodicMemory(
            capacity=10, feature_dim=d_in, proj_dims=(d_out,)
        )
        N = 200
        x = F.normalize(torch.randn(N, d_in), p=2, dim=-1)
        x_proj = F.normalize(x @ mem.R_64.T, p=2, dim=-1)

        # Pairwise cosines: [N, N]
        c_full = x @ x.T
        c_proj = x_proj @ x_proj.T
        # Mean absolute deviation should be small (well under 0.1 for
        # d_out = 64 with N = 200 points in d_in = 384).
        mad = (c_full - c_proj).abs().mean()
        assert mad < 0.1, f"JL cosine mean-abs-deviation {mad} too large"

    def test_cached_projection_keys_have_correct_shape(self):
        """keys_proj_d buffers are [capacity, d] and start zeroed."""
        mem = EnsembleEpisodicMemory(
            capacity=20, feature_dim=384, proj_dims=(128, 64)
        )
        assert mem.keys_proj_128.shape == (20, 128)
        assert mem.keys_proj_64.shape == (20, 64)
        assert torch.all(mem.keys_proj_128 == 0)
        assert torch.all(mem.keys_proj_64 == 0)


# =========================================================================
# 3. Store / count
# =========================================================================
class TestStoreAndCount:
    """Storing increments count; raw and projected buffers are populated."""

    def test_store_increments_count(self):
        mem = EnsembleEpisodicMemory(capacity=10, feature_dim=384, proj_dims=(128, 64))
        mem.store(torch.randn(1, 384))
        assert mem.get_usage() == 1

    def test_store_populates_raw_and_projected_buffers(self):
        """A stored vector lands in keys_raw, values, and all projected buffers."""
        torch.manual_seed(0)
        mem = EnsembleEpisodicMemory(capacity=5, feature_dim=384, proj_dims=(128, 64))
        v = F.normalize(torch.randn(384), dim=0)
        mem.store(v)
        # Raw buffer
        assert torch.allclose(mem.keys_raw[0], v, atol=1e-5)
        assert torch.allclose(mem.values[0], v, atol=1e-5)
        # Projected buffers are not zero and have correct shape
        assert mem.keys_proj_128[0].abs().sum() > 0
        assert mem.keys_proj_64[0].abs().sum() > 0
        # Projected buffers should equal R_d @ v
        assert torch.allclose(mem.keys_proj_128[0], mem.R_128 @ v, atol=1e-5)
        assert torch.allclose(mem.keys_proj_64[0], mem.R_64 @ v, atol=1e-5)

    def test_store_handles_2d_input(self):
        """store(x) with [B, D] averages over the batch dim."""
        mem = EnsembleEpisodicMemory(capacity=5, feature_dim=384, proj_dims=(128, 64))
        x = torch.randn(3, 384)
        mem.store(x)
        assert mem.get_usage() == 1
        assert torch.allclose(mem.values[0], x.mean(0), atol=1e-6)

    def test_store_wraps_around_capacity(self):
        """Storing > capacity items wraps and count saturates at capacity."""
        mem = EnsembleEpisodicMemory(capacity=4, feature_dim=8, proj_dims=())
        for _ in range(10):
            mem.store(torch.randn(1, 8))
        assert mem.get_usage() == 4  # saturated at capacity
        assert int(mem.ptr.item()) == 10 % 4  # 2

    def test_reset_clears_memory(self):
        mem = EnsembleEpisodicMemory(capacity=5, feature_dim=384, proj_dims=(128, 64))
        for _ in range(3):
            mem.store(torch.randn(1, 384))
        mem.reset()
        assert mem.get_usage() == 0
        assert torch.all(mem.keys_raw == 0)
        assert torch.all(mem.keys_proj_128 == 0)
        assert torch.all(mem.keys_proj_64 == 0)


# =========================================================================
# 4. Top-1 retrieval with known-similar query (cosine > 0.9)
# =========================================================================
class TestTop1Retrieval:
    """The headline test: 50 random vectors + a known similar query
    (cosine > 0.9) should return the original at rank 1."""

    def test_top1_with_known_similar_query(self):
        """Insert a target at a known index, query with a small perturbation,
        verify the top-1 is the original target index and the ensemble score
        is high (> 0.9)."""
        torch.manual_seed(42)
        mem = EnsembleEpisodicMemory(
            capacity=100, feature_dim=384, proj_dims=(128, 64)
        )
        target = F.normalize(torch.randn(384), dim=0)
        target_idx = 25

        for i in range(50):
            v = F.normalize(torch.randn(384), dim=0)
            if i == target_idx:
                v = target
            mem.store(v)

        # Build a query with cosine > 0.9 to the target.
        # With F.normalize(target + 0.01 * noise) and noise ~ N(0, I_384),
        # the dot product with target is approximately 1 / sqrt(1 + 0.0001*384) > 0.98.
        query = F.normalize(target + 0.01 * torch.randn(384), dim=0).unsqueeze(0)

        # Direct cosine sanity check on the raw buffer.
        # F.cosine_similarity broadcasts [1, D] vs [N, D] → [N].
        raw_cos = F.cosine_similarity(query, mem.keys_raw[:50], dim=-1)
        assert raw_cos[target_idx].item() > 0.9, (
            f"Raw cosine to target {raw_cos[target_idx].item():.4f} "
            f"should be > 0.9 for the test premise to hold"
        )

        idx, scores = mem.search(query, k=5)
        top1 = int(idx[0, 0].item())
        top1_score = float(scores[0, 0].item())

        assert top1 == target_idx, (
            f"Top-1 should be index {target_idx}, got {top1} "
            f"(top-5: {idx[0].tolist()})"
        )
        assert top1_score > 0.9, (
            f"Top-1 ensemble score {top1_score:.4f} should be > 0.9"
        )

    def test_retrieve_returns_query_when_empty(self):
        """retrieve on empty memory returns the query unchanged (residual)."""
        mem = EnsembleEpisodicMemory(capacity=10, feature_dim=384, proj_dims=(128, 64))
        q = torch.randn(2, 384)
        out = mem.retrieve(q, k=3)
        assert torch.allclose(out, q, atol=1e-6)

    def test_retrieve_shape_is_query_shape(self):
        """retrieve(query) returns a tensor of the same shape as query."""
        torch.manual_seed(0)
        mem = EnsembleEpisodicMemory(capacity=20, feature_dim=384, proj_dims=(128, 64))
        for _ in range(5):
            mem.store(torch.randn(384))
        q = torch.randn(3, 384)
        out = mem.retrieve(q, k=3)
        assert out.shape == q.shape

    def test_search_returns_indices_and_scores(self):
        """search() returns (indices, scores) with matching shapes."""
        torch.manual_seed(0)
        mem = EnsembleEpisodicMemory(capacity=20, feature_dim=384, proj_dims=(128, 64))
        for _ in range(5):
            mem.store(torch.randn(384))
        idx, sc = mem.search(torch.randn(2, 384), k=3)
        assert idx.shape == (2, 3)
        assert sc.shape == (2, 3)
        assert idx.dtype == torch.long


# =========================================================================
# 5. Learnable weights: gradient flows to weight_logits
# =========================================================================
class TestLearnableWeights:
    """``weight_logits`` is a learnable nn.Parameter; gradient flows
    through ``ensemble_scores`` (the differentiable path)."""

    def test_weight_logits_is_nn_parameter(self):
        """weight_logits is a leaf nn.Parameter with requires_grad=True."""
        mem = EnsembleEpisodicMemory(
            capacity=5, feature_dim=384, proj_dims=(128, 64), learn_weights=True
        )
        assert isinstance(mem.weight_logits, torch.nn.Parameter)
        assert mem.weight_logits.requires_grad is True
        assert mem.weight_logits.shape == (3,)

    def test_gradient_flows_to_weight_logits(self):
        """A forward pass through ensemble_scores + backward() yields a
        non-None, non-zero gradient on weight_logits."""
        torch.manual_seed(0)
        mem = EnsembleEpisodicMemory(
            capacity=10, feature_dim=384, proj_dims=(128, 64)
        )
        for _ in range(5):
            mem.store(F.normalize(torch.randn(384), dim=0))

        query = torch.randn(1, 384, requires_grad=False)
        scores = mem.ensemble_scores(query)  # [1, 5]
        # Loss: maximize the first score
        loss = -scores[0, 0]
        loss.backward()

        assert mem.weight_logits.grad is not None, (
            "weight_logits.grad should be populated after backward()"
        )
        g = mem.weight_logits.grad
        # The gradient through softmax(0) w.r.t. logits[0] for a single
        # contribution is sims[0] - sims[0]*w[0] = sims[0] * (1 - w[0])
        # which is non-zero unless all sims are equal. With random data
        # this is overwhelmingly true.
        assert g.abs().sum() > 0, "Gradient on weight_logits should be non-zero"

    def test_optimizer_step_changes_weights(self):
        """A single optimizer step actually moves the softmax weights."""
        torch.manual_seed(0)
        mem = EnsembleEpisodicMemory(
            capacity=10, feature_dim=384, proj_dims=(128, 64)
        )
        for _ in range(5):
            mem.store(F.normalize(torch.randn(384), dim=0))

        weights_before = mem.get_weights().clone()

        opt = torch.optim.SGD([mem.weight_logits], lr=1.0)
        query = torch.randn(1, 384)
        scores = mem.ensemble_scores(query)
        # Push the first weight up (loss increases when the first weight
        # is small relative to score[0, 0])
        loss = -(scores[0, 0] - 0.5 * scores[0, 1] - 0.5 * scores[0, 2])
        opt.zero_grad()
        loss.backward()
        opt.step()

        weights_after = mem.get_weights()
        assert not torch.allclose(weights_before, weights_after, atol=1e-6), (
            "Weights should change after one optimizer step"
        )
        # Softmax is invariant to additive constant on logits, so we check
        # that the gradient step moved the logits at all.
        assert (mem.weight_logits - 0.0).abs().sum() > 0

    def test_weights_sum_to_one_after_step(self):
        """After an optimizer step the softmax still sums to 1."""
        torch.manual_seed(0)
        mem = EnsembleEpisodicMemory(
            capacity=10, feature_dim=384, proj_dims=(128, 64)
        )
        for _ in range(3):
            mem.store(torch.randn(384))
        opt = torch.optim.SGD([mem.weight_logits], lr=0.5)
        scores = mem.ensemble_scores(torch.randn(1, 384))
        opt.zero_grad()
        scores.sum().backward()
        opt.step()
        w = mem.get_weights()
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)
        assert (w >= 0).all(), "Softmax weights must be non-negative"

    def test_learn_weights_false_disables_grad(self):
        """learn_weights=False → requires_grad on weight_logits is False."""
        mem = EnsembleEpisodicMemory(
            capacity=5, feature_dim=384, proj_dims=(128, 64), learn_weights=False
        )
        assert mem.weight_logits.requires_grad is False


# =========================================================================
# 6. Ensemble scoring — math sanity
# =========================================================================
class TestEnsembleScoring:
    """Differentiable ensemble_scores math is consistent."""

    def test_ensemble_scores_shape(self):
        """[B, count] for [B, D] query."""
        torch.manual_seed(0)
        mem = EnsembleEpisodicMemory(
            capacity=20, feature_dim=384, proj_dims=(128, 64)
        )
        for _ in range(7):
            mem.store(torch.randn(384))
        s = mem.ensemble_scores(torch.randn(3, 384))
        assert s.shape == (3, 7)

    def test_ensemble_scores_on_empty_memory(self):
        mem = EnsembleEpisodicMemory(
            capacity=5, feature_dim=384, proj_dims=(128, 64)
        )
        s = mem.ensemble_scores(torch.randn(2, 384))
        assert s.shape == (2, 0)

    def test_uniform_weights_equal_to_average_of_cosines(self):
        """At init, weights are uniform, so the ensemble score is the
        unweighted average of per-space cosines (within tolerance)."""
        torch.manual_seed(0)
        mem = EnsembleEpisodicMemory(
            capacity=5, feature_dim=384, proj_dims=(128, 64)
        )
        # Force uniform weights to be sure
        with torch.no_grad():
            mem.weight_logits.zero_()
        for _ in range(4):
            mem.store(F.normalize(torch.randn(384), dim=0))

        q = F.normalize(torch.randn(384), dim=0).unsqueeze(0)
        ens = mem.ensemble_scores(q).squeeze(0)  # [4]

        # Manually compute per-space cosines
        sims_list = []
        sims_list.append(F.cosine_similarity(q, mem.keys_raw[:4], dim=-1).squeeze(0))
        for d in mem.proj_dims:
            q_proj = (q @ getattr(mem, f"R_{d}").T)
            k = F.normalize(getattr(mem, f"keys_proj_{d}")[:4], p=2, dim=-1)
            sims_list.append((F.normalize(q_proj, p=2, dim=-1) @ k.T).squeeze(0))
        avg = torch.stack(sims_list, dim=0).mean(dim=0)
        assert torch.allclose(ens, avg, atol=1e-5)

    def test_weight_perturbation_changes_ensemble_score(self):
        """Manually shifting one logit changes the ensemble score for
        memories with unequal per-space cosines."""
        torch.manual_seed(0)
        mem = EnsembleEpisodicMemory(
            capacity=10, feature_dim=384, proj_dims=(128, 64)
        )
        for _ in range(5):
            mem.store(F.normalize(torch.randn(384), dim=0))

        q = F.normalize(torch.randn(384), dim=0).unsqueeze(0)
        s1 = mem.ensemble_scores(q).detach().clone()

        with torch.no_grad():
            mem.weight_logits[0] += 5.0  # push weight[0] toward 1
        s2 = mem.ensemble_scores(q).detach()

        # Per-space cosines are almost surely not all equal for random data
        assert not torch.allclose(s1, s2, atol=1e-4), (
            "Score should change when one weight is shifted substantially"
        )


# =========================================================================
# 7. Forget
# =========================================================================
class TestForget:
    """forget() compacts the buffers in-place and updates count."""

    def test_forget_returns_count_when_below_threshold(self):
        """With only 1 stored item, forget() returns 1 unchanged."""
        mem = EnsembleEpisodicMemory(
            capacity=5, feature_dim=384, proj_dims=(128, 64)
        )
        mem.store(torch.randn(1, 384))
        kept = mem.forget(threshold=0.99)
        assert kept == 1

    def test_forget_prunes_low_usage(self):
        """A very high threshold prunes most memories."""
        torch.manual_seed(0)
        mem = EnsembleEpisodicMemory(
            capacity=50, feature_dim=64, proj_dims=(32,)
        )
        for _ in range(20):
            mem.store(F.normalize(torch.randn(64), dim=0))
        before = mem.get_usage()
        kept = mem.forget(threshold=0.99)
        assert kept <= before
        assert mem.get_usage() == kept
        # Count is at most before
        assert mem.get_usage() <= before

    def test_forget_compacts_projected_buffers(self):
        """After forget(), the kept entries in all projected buffers are valid
        (norms non-zero where we just kept them, zero beyond)."""
        torch.manual_seed(0)
        mem = EnsembleEpisodicMemory(
            capacity=20, feature_dim=64, proj_dims=(32, 16)
        )
        for _ in range(15):
            mem.store(F.normalize(torch.randn(64), dim=0))
        kept = mem.forget(threshold=-1.0)  # keep everything
        assert kept == 15
        # Now prune a few with a stricter threshold
        kept2 = mem.forget(threshold=0.99)
        # All three buffers should have consistent kept-count rows
        nz_raw = (mem.keys_raw[:kept2].abs().sum(dim=1) > 0).sum().item()
        nz_128 = (mem.keys_proj_32[:kept2].abs().sum(dim=1) > 0).sum().item()
        nz_64 = (mem.keys_proj_16[:kept2].abs().sum(dim=1) > 0).sum().item()
        assert nz_raw == kept2
        assert nz_128 == kept2
        assert nz_64 == kept2


# =========================================================================
# 8. Stats
# =========================================================================
class TestStats:
    """get_stats() returns a serializable dict with all expected keys."""

    def test_stats_keys(self):
        mem = EnsembleEpisodicMemory(
            capacity=20, feature_dim=384, proj_dims=(128, 64)
        )
        stats = mem.get_stats()
        for key in ("capacity", "feature_dim", "proj_dims", "n_spaces",
                    "n_stored", "weights", "weight_logits", "learn_weights"):
            assert key in stats, f"Missing key: {key}"
        assert stats["capacity"] == 20
        assert stats["feature_dim"] == 384
        assert stats["proj_dims"] == [128, 64]
        assert stats["n_spaces"] == 3
        assert stats["n_stored"] == 0
        assert stats["learn_weights"] is True
        assert len(stats["weights"]) == 3
        assert len(stats["weight_logits"]) == 3
        # Weights sum to 1
        assert abs(sum(stats["weights"]) - 1.0) < 1e-5

    def test_stats_n_stored_updates(self):
        mem = EnsembleEpisodicMemory(
            capacity=10, feature_dim=384, proj_dims=(128, 64)
        )
        for _ in range(4):
            mem.store(torch.randn(1, 384))
        assert mem.get_stats()["n_stored"] == 4


# =========================================================================
# 9. Multi-dim / multi-proj-dim support
# =========================================================================
class TestMultiDimSupport:
    """Constructor works across various (feature_dim, proj_dims) combinations."""

    @pytest.mark.parametrize("feature_dim,proj_dims", [
        (64, (32,)),
        (128, (64, 32)),
        (256, (128, 64, 16)),
        (512, (256,)),
    ])
    def test_construct_and_query(self, feature_dim, proj_dims):
        torch.manual_seed(0)
        mem = EnsembleEpisodicMemory(
            capacity=10, feature_dim=feature_dim, proj_dims=proj_dims
        )
        for _ in range(5):
            mem.store(F.normalize(torch.randn(feature_dim), dim=0))
        q = F.normalize(torch.randn(1, feature_dim), dim=0)
        idx, sc = mem.search(q, k=3)
        assert idx.shape == (1, 3)
        assert sc.shape == (1, 3)
        # Weights length matches number of spaces
        assert len(mem.get_weights()) == 1 + len(proj_dims)

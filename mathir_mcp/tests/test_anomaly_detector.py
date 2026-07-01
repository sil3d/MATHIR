"""Tests for MahalanobisDetector — the real anomaly-detection math.

These are fast, synthetic-data unit tests for the detector's numerical
behavior (shrinkage, warmup gate, score monotonicity). They do not claim
to measure real-world detection quality — see
tests/data/anomaly_eval/run_eval.py for the honest, real-embedding
evaluation that is run separately (not part of the fast pytest suite).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_MCP_ROOT = _HERE.parent
if str(_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT))

try:
    from mathir_lib.mathir_anomaly import MahalanobisDetector
except ImportError:
    from mathir_anomaly import MahalanobisDetector  # type: ignore[no-redef]


def test_not_warmed_up_below_threshold():
    """Fewer than warmup_count updates -> is_warmed_up() is False."""
    d = MahalanobisDetector(dim=8, threshold=2.0, warmup_count=30)
    rng = np.random.RandomState(0)
    for _ in range(29):
        d.update(rng.randn(8).astype(np.float32))
    assert d.is_warmed_up() is False


def test_warmed_up_at_threshold():
    """Exactly warmup_count updates -> is_warmed_up() becomes True."""
    d = MahalanobisDetector(dim=8, threshold=2.0, warmup_count=30)
    rng = np.random.RandomState(0)
    for _ in range(30):
        d.update(rng.randn(8).astype(np.float32))
    assert d.is_warmed_up() is True


def test_score_low_for_in_distribution_point():
    """A point drawn from the same distribution as training scores low.

    Uses dim=4 (not 16): the expected Mahalanobis distance of a point drawn
    from a d-dimensional standard normal is approximately sqrt(d) (it
    follows a chi distribution with d degrees of freedom). At d=16 that's
    ~4.0, which would make a `< 3.0` assertion fail on typical in-distribution
    points, not just true outliers. At d=4 the expectation is ~2.0, giving
    real headroom below the 3.0 threshold used here and in the outlier test.
    """
    rng = np.random.RandomState(1)
    d = MahalanobisDetector(dim=4, threshold=3.0, warmup_count=30, regularization=1e-3)
    for _ in range(200):
        d.update(rng.randn(4).astype(np.float32))
    in_dist = rng.randn(4).astype(np.float32)
    score = d.score(in_dist)
    assert score < 3.0, f"expected in-distribution score < 3.0, got {score}"


def test_score_high_for_far_outlier():
    """A point far outside the training distribution scores high."""
    rng = np.random.RandomState(2)
    d = MahalanobisDetector(dim=16, threshold=3.0, warmup_count=30, regularization=1e-3)
    for _ in range(200):
        d.update(rng.randn(16).astype(np.float32))
    outlier = rng.randn(16).astype(np.float32) * 0.1 + 50.0  # far from origin
    score = d.score(outlier)
    assert score > 3.0, f"expected outlier score > 3.0, got {score}"


def test_shrinkage_keeps_matrix_invertible_at_low_sample_count():
    """Even with very few samples (rank-deficient covariance), score() must
    not raise — shrinkage regularization keeps the matrix invertible."""
    d = MahalanobisDetector(dim=32, threshold=2.0, warmup_count=3, regularization=1e-3)
    rng = np.random.RandomState(3)
    # Only 3 samples for a 32-dim space: covariance is rank-deficient
    # without shrinkage.
    for _ in range(3):
        d.update(rng.randn(32).astype(np.float32))
    assert d.is_warmed_up() is True
    # Must not raise LinAlgError.
    score = d.score(rng.randn(32).astype(np.float32))
    assert isinstance(score, float)
    assert score >= 0.0


def test_to_dict_from_dict_roundtrip():
    """Serializing and reloading a detector preserves its scoring behavior."""
    rng = np.random.RandomState(4)
    d = MahalanobisDetector(dim=8, threshold=2.5, warmup_count=10, regularization=1e-3)
    for _ in range(50):
        d.update(rng.randn(8).astype(np.float32))

    state = d.to_dict()
    d2 = MahalanobisDetector.from_dict(state)

    probe = rng.randn(8).astype(np.float32)
    assert d.score(probe) == pytest.approx(d2.score(probe), rel=1e-5)
    assert d2.is_warmed_up() == d.is_warmed_up()


try:
    from mathir_lib.mathir_vec import VecMemory
except ImportError:
    from mathir_vec import VecMemory  # type: ignore[no-redef]


def test_check_and_update_anomaly_not_warmed_up(tmp_path):
    """Before warmup_count saves, every embedding is treated as non-anomalous
    and silently folded into the baseline."""
    db = tmp_path / "anomaly_warmup.db"
    vm = VecMemory(db, embedding_dim=384)
    rng = np.random.RandomState(10)
    emb = rng.randn(384).astype(np.float32)
    result = vm.check_and_update_anomaly(emb, threshold=2.0, warmup_count=30)
    assert result["is_anomaly"] is False
    assert result["warmed_up"] is False


def test_check_and_update_anomaly_flags_outlier_after_warmup(tmp_path):
    """After warmup, a far-outlier embedding is flagged; an in-distribution
    one is not, and the baseline does not absorb the outlier.

    dim=64 is the smallest value accepted by VecMemory.VALID_DIMS. For a
    d-dimensional standard normal, the expected Mahalanobis distance to a
    well-estimated baseline is ~sqrt(d) (chi distribution with d degrees of
    freedom) -- for d=64 that's ~8, with in-distribution samples commonly
    landing in the 8-11 range. threshold=20 and warmup_count=500 (>> dim,
    so the covariance estimate is well-conditioned by the time warmup ends)
    give a wide, seed-stable margin between in-distribution scores (~9-11)
    and the deliberately extreme outlier (score in the hundreds).
    """
    db = tmp_path / "anomaly_flag.db"
    vm = VecMemory(db, embedding_dim=64)
    rng = np.random.RandomState(11)

    # Build the baseline with the real 64-dim VecMemory path by calling
    # check_and_update_anomaly 500 times with in-distribution points
    # (matches what memory_save will do for non-anomalous saves).
    for _ in range(500):
        normal = rng.randn(64).astype(np.float32)
        vm.check_and_update_anomaly(normal, threshold=20.0, warmup_count=500)

    outlier = (rng.randn(64).astype(np.float32) * 0.1) + 50.0
    result = vm.check_and_update_anomaly(outlier, threshold=20.0, warmup_count=500)
    assert result["warmed_up"] is True
    assert result["is_anomaly"] is True
    assert result["score"] > 20.0

    in_dist = rng.randn(64).astype(np.float32)
    result2 = vm.check_and_update_anomaly(in_dist, threshold=20.0, warmup_count=500)
    assert result2["is_anomaly"] is False


def test_anomaly_state_persists_across_vecmemory_instances(tmp_path):
    """Detector state survives a daemon restart (new VecMemory on same db_path)."""
    db = tmp_path / "anomaly_persist.db"
    rng = np.random.RandomState(12)

    vm1 = VecMemory(db, embedding_dim=64)
    for _ in range(40):
        vm1.check_and_update_anomaly(rng.randn(64).astype(np.float32), threshold=2.0, warmup_count=30)
    vm1.close()

    vm2 = VecMemory(db, embedding_dim=64)
    outlier = (rng.randn(64).astype(np.float32) * 0.1) + 50.0
    result = vm2.check_and_update_anomaly(outlier, threshold=2.0, warmup_count=30)
    assert result["warmed_up"] is True, "detector state should have persisted across instances"
    assert result["is_anomaly"] is True


def test_search_include_embeddings_returns_raw_vectors(tmp_path):
    """VecMemory.search(include_embeddings=True) attaches the raw stored
    vector to each result; default (False) omits it. This is the P0 fix
    that lets downstream retrieval algorithms work on real per-memory
    vectors instead of a collapsed 1-dim score proxy."""
    db = tmp_path / "emb_expose.db"
    vm = VecMemory(db, embedding_dim=384)
    rng = np.random.RandomState(7)
    stored = rng.randn(384).astype(np.float32)
    vm.store("mem_x", stored, {"content": "hello vectors", "agent": "t",
                               "block_type": "episodic", "label": "", "priority": 5})

    # Default: no embedding key.
    res_default = vm.search(query_embedding=stored, k=1)
    assert res_default and "embedding" not in res_default[0]

    # include_embeddings=True: embedding present, right length, matches stored.
    res_emb = vm.search(query_embedding=stored, k=1, include_embeddings=True)
    assert res_emb and "embedding" in res_emb[0]
    emb = res_emb[0]["embedding"]
    assert isinstance(emb, list) and len(emb) == 384
    assert np.allclose(np.array(emb, dtype=np.float32), stored, atol=1e-5)

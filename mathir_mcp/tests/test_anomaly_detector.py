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

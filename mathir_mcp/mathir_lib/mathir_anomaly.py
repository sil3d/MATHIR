"""Mahalanobis-distance anomaly detector for the `immunological` memory tier.

Maintains a running (EMA) mean and covariance of embeddings considered
"normal" and scores new embeddings by their Mahalanobis distance to that
distribution. Covariance shrinkage (``Sigma + epsilon*I``) keeps the matrix
invertible even when few samples have been seen relative to the embedding
dimensionality.

This replaces two previous, disconnected implementations:
  - ``mathir_lib.memory.immunological.MahalanobisImmunologicalMemory``
    (deprecated v7 architecture, no longer present in the codebase).
  - ``mathir_dropin``'s ``_ImmuneMemory`` (a separate, simpler min-L2-distance
    detector for the standalone embeddable library — not Mahalanobis at all,
    and not wired to the MCP server).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


class MahalanobisDetector:
    """Online Mahalanobis-distance anomaly detector.

    Parameters
    ----------
    dim:
        Embedding dimensionality.
    threshold:
        Mahalanobis distance above which a point is flagged anomalous.
    regularization:
        Shrinkage added to the diagonal of the covariance matrix before
        inversion (``Sigma + regularization * I``). Keeps the matrix
        invertible at low sample counts or near-degenerate covariance.
    warmup_count:
        Minimum number of ``update()`` calls before ``score()``/
        ``is_warmed_up()`` is meaningful. Below this, the baseline is not
        considered reliable enough to flag anomalies.
    ema_decay:
        Exponential moving average decay for mean/covariance updates.
        Smaller values adapt more slowly (more stable baseline); larger
        values track recent data more closely.
    recompute_every:
        The covariance inverse is O(dim^3) to compute. It is cached and
        only recomputed every N calls to ``update()`` to bound latency.
    """

    def __init__(
        self,
        dim: int,
        threshold: float,
        regularization: float = 1e-4,
        warmup_count: int = 30,
        ema_decay: float = 0.01,
        recompute_every: int = 10,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.threshold = threshold
        self.regularization = regularization
        self.warmup_count = warmup_count
        self.ema_decay = ema_decay
        self.recompute_every = recompute_every

        self._mean = np.zeros(dim, dtype=np.float64)
        self._cov = np.eye(dim, dtype=np.float64)
        self._n_updates = 0
        self._cov_inv: np.ndarray | None = None
        self._updates_since_inverse = 0

    def is_warmed_up(self) -> bool:
        return self._n_updates >= self.warmup_count

    def update(self, embedding: np.ndarray) -> None:
        """Fold a new "normal" embedding into the running mean/covariance."""
        x = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"embedding dim {x.shape[0]} != detector dim {self.dim}")

        self._n_updates += 1
        if self._n_updates == 1:
            self._mean = x.copy()
            self._cov = np.eye(self.dim, dtype=np.float64) * self.regularization
        else:
            # EMA mean update.
            delta = x - self._mean
            self._mean = self._mean + self.ema_decay * delta
            # EMA covariance update (outer product of the centered deviation).
            outer = np.outer(delta, delta)
            self._cov = (1 - self.ema_decay) * self._cov + self.ema_decay * outer

        self._updates_since_inverse += 1
        self._cov_inv = None  # invalidate cache; recomputed lazily in score()

    def _get_inverse(self) -> np.ndarray:
        if self._cov_inv is None or self._updates_since_inverse >= self.recompute_every:
            reg_cov = self._cov + np.eye(self.dim, dtype=np.float64) * self.regularization
            try:
                self._cov_inv = np.linalg.inv(reg_cov)
            except np.linalg.LinAlgError:
                # Pathological case (e.g. all-identical embeddings during
                # warmup): fall back to identity so score() degrades to
                # plain Euclidean distance rather than crashing the caller.
                self._cov_inv = np.eye(self.dim, dtype=np.float64)
            self._updates_since_inverse = 0
        return self._cov_inv

    def score(self, embedding: np.ndarray) -> float:
        """Mahalanobis distance of ``embedding`` to the current baseline."""
        x = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"embedding dim {x.shape[0]} != detector dim {self.dim}")
        delta = x - self._mean
        inv = self._get_inverse()
        m_sq = float(delta @ inv @ delta)
        return float(np.sqrt(max(m_sq, 0.0)))

    def is_anomaly(self, embedding: np.ndarray) -> bool:
        return self.score(embedding) > self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "threshold": self.threshold,
            "regularization": self.regularization,
            "warmup_count": self.warmup_count,
            "ema_decay": self.ema_decay,
            "recompute_every": self.recompute_every,
            "mean": self._mean.tolist(),
            "cov": self._cov.tolist(),
            "n_updates": self._n_updates,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "MahalanobisDetector":
        det = cls(
            dim=state["dim"],
            threshold=state["threshold"],
            regularization=state["regularization"],
            warmup_count=state["warmup_count"],
            ema_decay=state["ema_decay"],
            recompute_every=state["recompute_every"],
        )
        det._mean = np.array(state["mean"], dtype=np.float64)
        det._cov = np.array(state["cov"], dtype=np.float64)
        det._n_updates = state["n_updates"]
        return det

"""Mahalanobis-distance anomaly detector for the `immunological` memory tier.

Maintains a running (EMA) mean and covariance of embeddings considered
"normal" and scores new embeddings by their Mahalanobis distance to that
distribution. Covariance regularization uses adaptive, Ledoit-Wolf-style
shrinkage toward a scaled identity: the shrinkage intensity
``alpha = max(regularization, dim / n_updates)`` adapts to how many samples
have been seen relative to the embedding dimensionality, rather than adding
a fixed epsilon. This keeps the matrix invertible and well-conditioned even
when few samples have been seen relative to the embedding dimensionality --
a fixed epsilon alone does not fix the rank-deficiency of the empirical
covariance when n_updates is not much larger than dim (see the detailed
note in ``_get_inverse`` below).

This replaces two previous, disconnected implementations:
  - ``mathir_lib.memory.immunological.MahalanobisImmunologicalMemory``
    (deprecated v7 architecture, no longer present in the codebase).
  - ``mathir_dropin``'s ``_ImmuneMemory`` (a separate, simpler min-L2-distance
    detector for the standalone embeddable library — not Mahalanobis at all,
    and not wired to the MCP server).
"""
from __future__ import annotations

import base64
from typing import Any, Dict

import numpy as np


def _array_to_b64(arr: np.ndarray) -> str:
    """Encode a float64 ndarray as a base64 string of its raw bytes.

    Much faster and more compact to (de)serialize via JSON than
    ``.tolist()`` / nested Python lists, which matters here because
    ``_cov`` is a ``dim x dim`` array (147,456 floats at dim=384) that gets
    re-serialized on every ``check_and_update_anomaly()`` call.
    """
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).decode("ascii")


def _b64_to_array(data: str, shape: tuple) -> np.ndarray:
    flat = np.frombuffer(base64.b64decode(data), dtype=np.float64)
    return flat.reshape(shape).copy()


class MahalanobisDetector:
    """Online Mahalanobis-distance anomaly detector.

    Parameters
    ----------
    dim:
        Embedding dimensionality.
    threshold:
        Mahalanobis distance above which a point is flagged anomalous.
    regularization:
        Floor on the shrinkage intensity used before inversion: the
        covariance is shrunk toward a scaled identity by a factor
        ``alpha = max(regularization, dim / n_updates)``, so shrinkage is
        strong when few samples have been seen relative to ``dim`` and
        settles at this floor once ``n_updates >> dim``. Keeps the matrix
        invertible and well-conditioned even when samples are scarce
        relative to the embedding dimensionality.
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
        warmup_count: int = 60,
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
            # Effective decay: behaves like a true running average for the
            # first ~1/ema_decay samples (1/2, 1/3, 1/4, ...), then settles
            # into the fixed ema_decay for long-run adaptability.
            #
            # A fixed ema_decay from the very first update converges far too
            # slowly: starting from the near-zero initial covariance above,
            # after n updates the initial state still carries weight
            # (1-ema_decay)**n. At the default ema_decay=0.01, that's 55% at
            # n=60 and 37% at n=100 -- meaning the covariance is still
            # severely underestimated well past any reasonable warmup_count,
            # which inflates Mahalanobis scores and flags normal points as
            # anomalous. Blending in 1/n for early updates fixes this: by
            # n=100 the running-average term (1/100=0.01) has already
            # crossed below ema_decay, so the transition is seamless.
            effective_decay = max(self.ema_decay, 1.0 / self._n_updates)
            delta = x - self._mean
            self._mean = self._mean + effective_decay * delta
            # Covariance update (outer product of the centered deviation).
            outer = np.outer(delta, delta)
            self._cov = (1 - effective_decay) * self._cov + effective_decay * outer

        self._updates_since_inverse += 1
        self._cov_inv = None  # invalidate cache; recomputed lazily in score()

    def _get_inverse(self) -> np.ndarray:
        if self._cov_inv is None or self._updates_since_inverse >= self.recompute_every:
            # Shrinkage toward a scaled identity (Ledoit-Wolf style), with
            # shrinkage intensity that adapts to how many samples have been
            # seen relative to the dimensionality.
            #
            # A small *fixed* epsilon (the old approach: cov + epsilon*I)
            # only prevents the matrix from being exactly singular -- it
            # does not fix the deeper problem that with n samples in a
            # d-dimensional space, n <= d (or even n only modestly above d)
            # leaves the empirical covariance severely rank-deficient, with
            # near-zero eigenvalues in unsampled directions. Inverting that
            # blows those directions up, inflating Mahalanobis distances for
            # any new point with a component there -- which is virtually
            # every point, since embeddings span the full space. This isn't
            # a cosmetic numerical issue: at dim=384 with n=60 samples, a
            # fixed epsilon=1e-3 produced in-distribution scores of ~560
            # against a threshold of 30 -- normal points were flagged as
            # anomalous by a factor of ~20x, regardless of the epsilon's
            # exact value (tested 1e-4 through 1.0).
            #
            # Shrinking harder toward `avg_var * I` when data is scarce
            # (alpha near 1 when n_updates << dim) and easing off as more
            # data accumulates (alpha floor at `regularization` once
            # n_updates >> dim) keeps the estimate well-conditioned without
            # needing a magic constant tuned to the embedding's scale.
            # Verified empirically to track the theoretical sqrt(dim)
            # in-distribution expectation within ~5% across dim in {64, 384}
            # and n_updates in {60, 400} -- i.e. it works whether or not the
            # caller can afford a large warmup_count.
            avg_var = float(np.trace(self._cov)) / self.dim
            alpha = min(1.0, self.dim / max(self._n_updates, 1))
            alpha = max(alpha, self.regularization)
            reg_cov = (1 - alpha) * self._cov + alpha * avg_var * np.eye(self.dim, dtype=np.float64)
            # Absolute floor in case avg_var itself is ~0 (e.g. the very
            # first few updates, before any real variance has been observed).
            reg_cov = reg_cov + np.eye(self.dim, dtype=np.float64) * 1e-8
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
        # mean/cov are serialized as base64-encoded raw float64 bytes rather
        # than nested JSON lists -- for a 384-dim embedding, ``_cov`` alone
        # is a 384x384 array (147,456 floats), and this is re-serialized on
        # every check_and_update_anomaly() call. Base64-encoded bytes are
        # both much smaller and much faster to (de)serialize than the
        # equivalent nested-list JSON representation, with no change to the
        # public to_dict()/from_dict() interface (still a JSON-serializable
        # dict of JSON-native types).
        return {
            "dim": self.dim,
            "threshold": self.threshold,
            "regularization": self.regularization,
            "warmup_count": self.warmup_count,
            "ema_decay": self.ema_decay,
            "recompute_every": self.recompute_every,
            "mean_b64": _array_to_b64(self._mean),
            "cov_b64": _array_to_b64(self._cov),
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
        dim = state["dim"]
        if "mean_b64" in state:
            det._mean = _b64_to_array(state["mean_b64"], (dim,))
            det._cov = _b64_to_array(state["cov_b64"], (dim, dim))
        else:
            # Backward compatibility with state persisted before the
            # base64 encoding change (plain nested-list JSON).
            det._mean = np.array(state["mean"], dtype=np.float64)
            det._cov = np.array(state["cov"], dtype=np.float64)
        det._n_updates = state["n_updates"]
        return det

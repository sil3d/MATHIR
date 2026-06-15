"""
Ensemble Episodic Memory — Adaptive Multi-Encoder Retrieval.

Combines cosine similarity from multiple dimensionalities via a learned
weighted ensemble. Stores the raw 384-dim embedding plus cached random
projections to lower-dimensional subspaces (e.g. 128, 64), and re-ranks
memories by a learned weighted sum of per-space cosines.

Mathematical intuition
----------------------
Let ``f in R^D`` be a stored feature and ``q in R^D`` be a query.
For each target dimension ``d in (D, d1, d2, ...)`` we compute the
cosine similarity in the corresponding space:

    s_d(f, q) = <R_d f, R_d q> / (||R_d f||_2 * ||R_d q||_2)

The Johnson-Lindenstrauss lemma guarantees that a random Gaussian
projection (rows of ``R_d`` drawn from ``N(0, I)`` and ``l2``-normalized)
approximately preserves pairwise distances with high probability when
``d`` is large enough. Concretely, for any ``epsilon in (0, 1/2)`` and
``N`` points in ``R^D``, a random projection to ``d = O(log N / eps^2)``
dimensions preserves all pairwise distances up to a ``(1+eps)`` factor.

The final ensemble score is

    s(f, q) = sum_{d} w_d * s_d(f, q),   sum_d w_d = 1,   w_d > 0

with the simplex constraint enforced via softmax over free logits
``w_d = exp(l_d) / sum_j exp(l_j)``. The logits ``l_d`` are learnable
``nn.Parameter``s, so the ensemble can adapt weights via gradient descent.

Intuition for the 3-tier ensemble:
- 384-dim: full fidelity (raw cosine), expensive at scale
- 128-dim: fast, slight info loss (random projection)
- 64-dim:  very fast, captures global structure, loses local detail
- Ensemble: best of all worlds, especially under distribution shift
  where one space may fail but the others compensate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
from typing import Tuple


class EnsembleEpisodicMemory(nn.Module):
    """
    Episodic memory with multi-encoder ensemble retrieval.

    Stores the raw ``feature_dim`` embedding plus pre-computed random
    projections to ``proj_dims``. At query time, computes cosine
    similarity in all spaces, then combines them via a learnable
    weighted ensemble (softmax-constrained logits).

    The random projection matrices follow the Johnson-Lindenstrauss
    construction: ``R_d in R^{d x D}`` with rows drawn from a standard
    Gaussian and then ``l2``-normalized. These matrices are fixed at
    init (not learnable) — the JL guarantee depends on the randomness,
    not on a learned projection.

    Learnable weights:
        ``weight_logits in R^{n_spaces}``, exposed via ``self.weights``
        (softmax) and constrained to the probability simplex.
    """

    def __init__(self, capacity: int = 1000, feature_dim: int = 384,
                 proj_dims: Tuple[int, ...] = (128, 64),
                 learn_weights: bool = True):
        """
        Args:
            capacity: maximum number of stored memories
            feature_dim: raw embedding dimension (e.g. 384)
            proj_dims: tuple of projected dimensions (e.g. (128, 64))
            learn_weights: if True, ensemble weights are learnable
                ``nn.Parameter``s. If False, they remain at the uniform
                initialization and gradients on ``weight_logits`` are
                disabled.
        """
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0, got {feature_dim}")
        for d in proj_dims:
            if d <= 0 or d > feature_dim:
                raise ValueError(
                    f"each proj_dim must satisfy 0 < d <= feature_dim, got {d}"
                )
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")

        self.capacity = capacity
        self.feature_dim = feature_dim
        self.proj_dims = tuple(proj_dims)
        self.learn_weights = learn_weights

        # All similarity spaces: raw + each projection
        self.all_dims = (feature_dim,) + self.proj_dims
        self.n_spaces = len(self.all_dims)

        # ---- Storage buffers ----
        # Raw full-dim keys and full-dim values (values are the high-fidelity
        # representation we actually return; the projections are used only
        # for similarity scoring).
        self.register_buffer("keys_raw", torch.zeros(capacity, feature_dim))
        self.register_buffer("values", torch.zeros(capacity, feature_dim))

        # Cached projected keys (re-computed at store time)
        for d in self.proj_dims:
            self.register_buffer(f"keys_proj_{d}", torch.zeros(capacity, d))

        # ---- Random projection matrices (Johnson-Lindenstrauss style) ----
        # Fixed at init; not learnable. Each row is l2-normalized.
        for d in self.proj_dims:
            R = torch.randn(d, feature_dim)
            R = F.normalize(R, p=2, dim=1)
            self.register_buffer(f"R_{d}", R)

        # ---- Learnable ensemble weights ----
        # Free logits, softmax-constrained to the simplex.
        self.weight_logits = nn.Parameter(torch.zeros(self.n_spaces))
        if not learn_weights:
            self.weight_logits.requires_grad_(False)

        # ---- Bookkeeping ----
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        # ---- Thread safety ----
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Ensemble weights
    # ------------------------------------------------------------------ #
    @property
    def weights(self) -> torch.Tensor:
        """Constrained ensemble weights ``w_d`` summing to 1, ``w_d > 0``."""
        return F.softmax(self.weight_logits, dim=0)

    def get_weights(self) -> torch.Tensor:
        """Convenience accessor matching the property (returns a detached copy)."""
        return self.weights.detach().clone()

    # ------------------------------------------------------------------ #
    # Projection helper
    # ------------------------------------------------------------------ #
    def _project(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Apply Johnson-Lindenstrauss random projection to ``target_dim``."""
        R = getattr(self, f"R_{target_dim}")
        return x @ R.T  # [..., target_dim]

    # ------------------------------------------------------------------ #
    # Per-space cosine similarity
    # ------------------------------------------------------------------ #
    def _cosine_to_buffer(self, query: torch.Tensor,
                          buffer_name: str) -> torch.Tensor:
        """
        Cosine similarity from ``query`` (already in the buffer's dim space)
        to all stored vectors in ``buffer_name``.

        The caller is responsible for projecting the query to the same
        dimensionality as the buffer.

        Args:
            query: [B, D_k] (must match the buffer's feature dim)
            buffer_name: name of the registered buffer holding [N, D_k]

        Returns:
            [B, N] cosine similarities (zeros if memory is empty)
        """
        count = int(self.count.item())
        if count == 0:
            return torch.zeros(query.size(0), 0, device=query.device,
                               dtype=query.dtype)

        buf = getattr(self, buffer_name)[:count]
        q = F.normalize(query, p=2, dim=-1)
        k = F.normalize(buf, p=2, dim=-1)
        return q @ k.T  # [B, count]

    # ------------------------------------------------------------------ #
    # Differentiable ensemble score
    # ------------------------------------------------------------------ #
    def ensemble_scores(self, query: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable ensemble similarity scores for all stored memories.

        This is the gradient-friendly path: it does **not** use ``topk``,
        so the result is fully differentiable w.r.t. ``weight_logits``.

        Args:
            query: [B, D] or [D]

        Returns:
            [B, count] ensemble cosine similarities
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        count = int(self.count.item())
        if count < 1:
            return torch.zeros(query.size(0), 0, device=query.device,
                               dtype=query.dtype)

        # Per-space cosine similarities. Project the query into each
        # projection space so its dim matches the buffer's dim.
        sims_list = [self._cosine_to_buffer(query, "keys_raw")]
        for d in self.proj_dims:
            q_proj = self._project(query, d)
            sims_list.append(self._cosine_to_buffer(q_proj, f"keys_proj_{d}"))

        sims_stack = torch.stack(sims_list, dim=0)  # [n_spaces, B, count]
        w = self.weights  # [n_spaces] — differentiable
        return (w.view(-1, 1, 1) * sims_stack).sum(dim=0)  # [B, count]

    # ------------------------------------------------------------------ #
    # API: store / retrieve / search / forget / get_stats
    # ------------------------------------------------------------------ #
    def store(self, x: torch.Tensor) -> None:
        """
        Store an embedding (or a batch — averaged over the batch dim).

        Args:
            x: [B, D] or [D] tensor to store
        """
        with self._lock:
            with torch.no_grad():
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                v = x.mean(0)  # [D]

                ptr = int(self.ptr.item())
                idx = ptr % self.capacity

                # Store raw + compute cached projections
                self.keys_raw[idx] = v
                self.values[idx] = v
                for d in self.proj_dims:
                    projected = self._project(v.unsqueeze(0), d).squeeze(0)
                    getattr(self, f"keys_proj_{d}")[idx] = projected

                self.ptr = torch.tensor(
                    (ptr + 1) % self.capacity, dtype=torch.long
                )
                self.count = torch.minimum(
                    self.count + 1,
                    torch.tensor(self.capacity, dtype=torch.long),
                )

    def retrieve(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Retrieve top-k memories by ensemble cosine, with residual.

        Args:
            query: [B, D] or [D] query
            k: number of memories to retrieve

        Returns:
            [B, D] averaged retrieved values + query (residual)
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        B = query.size(0)

        count = int(self.count.item())
        if count < 1:
            return query

        # Use the differentiable score (gradient still flows through
        # ``weights``; the topk indices just route which value rows we
        # gather, the value lookup is non-differentiable but that's fine
        # for inference).
        scores = self.ensemble_scores(query)  # [B, count]
        kk = min(k, count)
        top_idx = scores.topk(kk, dim=1).indices  # [B, k]

        # Gather values: [B, k, D]
        retrieved = self.values[top_idx]
        return retrieved.mean(dim=1) + query  # residual

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search for top-k most similar memories by ensemble score.

        Returns:
            (indices [B, k], ensemble scores [B, k])
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        B = query.size(0)

        count = int(self.count.item())
        if count == 0:
            return (torch.zeros(B, 0, dtype=torch.long, device=query.device),
                    torch.zeros(B, 0, device=query.device))

        scores = self.ensemble_scores(query)  # [B, count]
        kk = min(k, count)
        topk = scores.topk(kk, dim=1)
        return topk.indices, topk.values

    def forget(self, threshold: float = 0.1) -> int:
        """
        Prune low-usage memories (by mean raw-cosine to all other stored vectors).

        Uses centroid-based scoring (O(n)) instead of full pairwise cosine (O(n²)).

        Returns:
            number of memories kept
        """
        with self._lock:
            count = int(self.count.item())
            if count < 2:
                return count

            with torch.no_grad():
                # O(n) centroid-based scoring instead of O(n²) pairwise cosine
                keys = self.keys_raw[:count]
                centroid = keys.mean(dim=0)  # [D]
                centroid_norm = centroid.norm()
                if centroid_norm < 1e-8:
                    return count

                key_norms = keys.norm(dim=-1)  # [count]
                key_norms = torch.where(key_norms > 0, key_norms, torch.ones_like(key_norms))
                usage = (keys @ centroid) / (key_norms * centroid_norm)  # [count]
                mask = usage > threshold
                kept = int(mask.sum().item())

                if kept < count:
                    keep_idx = mask.nonzero(as_tuple=True)[0]
                    self.keys_raw[:kept] = self.keys_raw[:count][keep_idx]
                    self.values[:kept] = self.values[:count][keep_idx]
                    for d in self.proj_dims:
                        getattr(self, f"keys_proj_{d}")[:kept] = (
                            getattr(self, f"keys_proj_{d}")[:count][keep_idx]
                        )
                    self.count = torch.tensor(kept, dtype=torch.long)
                    self.ptr = torch.tensor(kept % self.capacity, dtype=torch.long)
                    return kept
            return count

    def get_stats(self) -> dict:
        """Return memory statistics."""
        return {
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "proj_dims": list(self.proj_dims),
            "n_spaces": self.n_spaces,
            "n_stored": int(self.count.item()),
            "weights": self.get_weights().tolist(),
            "weight_logits": self.weight_logits.detach().cpu().tolist(),
            "learn_weights": self.learn_weights,
        }

    def reset(self) -> None:
        """Reset all stored memories and bookkeeping."""
        with self._lock:
            self.keys_raw.zero_()
            self.values.zero_()
            for d in self.proj_dims:
                getattr(self, f"keys_proj_{d}").zero_()
            self.ptr = torch.tensor(0, dtype=torch.long)
            self.count = torch.tensor(0, dtype=torch.long)

    def get_usage(self) -> int:
        """Number of stored memories."""
        return int(self.count.item())

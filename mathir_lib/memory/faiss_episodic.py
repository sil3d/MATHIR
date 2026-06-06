"""
FAISS-Backed Episodic Memory
=============================

Drop-in replacement for ``EpisodicMemory`` that uses FAISS as a **backing index**
for fast exact (``IndexFlatIP``) or approximate (``IndexHNSWFlat``) nearest-
neighbor search, while keeping the **online-learning loop** intact (i.e. the
ability to insert and remove slots dynamically, one at a time, without a full
rebuild for every update).

Design
------
* **Keys** live in FAISS. We wrap the base index with ``IndexIDMap2`` so we can
  attach stable, monotonically-increasing 64-bit ``slot_id``s to each vector.
  That makes ``add_with_ids`` and ``remove_ids`` cheap and unambiguous.
* **Values** live in a parallel Python ``dict[int, Tensor]`` (FAISS only stores
  keys, not the values they refer to).
* **Cosine similarity** is implemented by L2-normalizing keys before adding /
  searching, so the inner-product returned by FAISS equals cosine.
* **Online learning**: every ``store`` call appends a new slot with a fresh
  ``slot_id``. If capacity is exceeded, the oldest slot is evicted in O(1).
* **Forget** prunes slots whose mean cosine similarity to all other slots is
  below ``threshold``. ``IndexFlatIP`` supports ``remove_ids`` in place;
  ``IndexHNSWFlat`` does not — in that case we rebuild the index from the
  surviving slots (still O(N log N) and only triggered on explicit forgets).

Interface
---------
Matches :class:`mathir_lib.memory.episodic.EpisodicMemory`::

    def __init__(self, capacity, feature_dim, use_hnsw=False, **kwargs)
    def store(self, x)                                      # [D] or [B, D]
    def retrieve(self, query, k=3) -> Tensor                # residual
    def search(self, query, k=5)   -> (indices, similarities)
    def forget(self, threshold=0.1) -> int                  # kept
    def get_stats(self) -> dict
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:  # pragma: no cover - import guard
    _HAS_FAISS = False
    faiss = None  # type: ignore


class FAISSBackedEpisodicMemory(nn.Module):
    """
    Episodic memory backed by a FAISS index.

    Args:
        capacity:       Soft cap on number of slots. When exceeded during
                        ``store``, the oldest slot (by insertion order) is
                        evicted in O(1).
        feature_dim:    Dimension of input feature / value / key. The same
                        dimension is used for the FAISS vectors and for the
                        value buffer.
        use_hnsw:       If ``True`` use ``IndexHNSWFlat`` (fast approximate,
                        sub-linear query). If ``False`` (default) use
                        ``IndexFlatIP`` (exact, linear scan, in-place
                        ``remove_ids``).
        hnsw_m:         HNSW graph degree (only used when ``use_hnsw=True``).
        hnsw_ef_construction: HNSW construction-time candidate list size.
        hnsw_ef_search: HNSW search-time candidate list size.
        normalize_keys: If ``True`` (default) L2-normalize keys before adding
                        / searching so that inner product equals cosine
                        similarity. Set to ``False`` to use raw inner
                        product.
        device:         ``"cpu"`` or ``"cuda"``. Keys are computed on this
                        device; FAISS always stores vectors on CPU (FAISS-GPU
                        is a separate package).
    """

    # ---------- construction ----------
    def __init__(
        self,
        capacity: int = 1000,
        feature_dim: int = 272,
        use_hnsw: bool = False,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        normalize_keys: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        if not _HAS_FAISS:
            raise ImportError(
                "faiss is not installed. Install with `pip install faiss-cpu`."
            )

        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.use_hnsw = bool(use_hnsw)
        self.normalize_keys = bool(normalize_keys)
        self.device = torch.device(device)

        # Build the FAISS index.
        if self.use_hnsw:
            # Use InnerProduct for cosine-like scoring (we L2-normalize keys
            # before add/search, so IP = cosine). The default metric for
            # IndexHNSWFlat is L2, which would silently break all our tests.
            base = faiss.IndexHNSWFlat(
                self.feature_dim,
                int(hnsw_m),
                faiss.METRIC_INNER_PRODUCT,
            )
            base.hnsw.efConstruction = int(hnsw_ef_construction)
            base.hnsw.efSearch = int(hnsw_ef_search)
        else:
            base = faiss.IndexFlatIP(self.feature_dim)
        self.index: faiss.Index = faiss.IndexIDMap2(base)

        # Parallel value storage. slot_id is a unique, monotonically increasing
        # integer that FAISS's IDMap can use without conflicts.
        self._values: dict[int, torch.Tensor] = {}
        self._ids: List[int] = []   # insertion order, for stable iteration
        self._next_id: int = 0

        # Lightweight stat buffer (purely informational, not used in math).
        self.register_buffer(
            "_n_stores", torch.tensor(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_n_searches", torch.tensor(0, dtype=torch.long),
            persistent=False,
        )

    # ---------- helpers ----------
    @staticmethod
    def _as_feature(x: torch.Tensor) -> torch.Tensor:
        """Flatten a [B, D] (or [..., D]) input to a 1-D ``[D]`` tensor."""
        if x.dim() > 1:
            return x.reshape(-1, x.shape[-1]).mean(0)
        return x

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalize along last dim. Zero vectors stay zero."""
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return x / norm

    def _new_index(self) -> faiss.Index:
        """Recreate the base index (used by ``reset`` and HNSW rebuild)."""
        if self.use_hnsw:
            base = faiss.IndexHNSWFlat(
                self.feature_dim, 32, faiss.METRIC_INNER_PRODUCT
            )
            base.hnsw.efConstruction = 200
            base.hnsw.efSearch = 64
        else:
            base = faiss.IndexFlatIP(self.feature_dim)
        return faiss.IndexIDMap2(base)

    # ---------- public API ----------
    def store(self, x: torch.Tensor) -> None:
        """
        Store a feature tensor as an episodic memory.

        Args:
            x: Tensor of shape ``[D]`` or ``[B, D]``. If ``[B, D]`` we store
               the per-feature mean across the batch (matching
               :class:`EpisodicMemory`).
        """
        with torch.no_grad():
            feat = self._as_feature(x.detach()).to(torch.float32).to(self.device)

            if self.capacity <= 0:
                # Degenerate capacity: drop everything.
                return

            # Evict oldest if at capacity.
            if len(self._ids) >= self.capacity:
                oldest_id = self._ids.pop(0)
                self._values.pop(oldest_id, None)
                if self.use_hnsw:
                    # HNSW doesn't support remove_ids -> rebuild from survivors.
                    self._rebuild_inplace()
                else:
                    try:
                        self.index.remove_ids(
                            np.array([oldest_id], dtype=np.int64)
                        )
                    except Exception:
                        # Defensive: fall back to a full rebuild.
                        self._rebuild_inplace()

            slot_id = self._next_id
            self._next_id += 1

            # Key: optionally normalized (cosine via IP).
            key = self._normalize(feat) if self.normalize_keys else feat
            self.index.add_with_ids(
                key.reshape(1, -1).cpu().numpy().astype(np.float32),
                np.array([slot_id], dtype=np.int64),
            )

            self._values[slot_id] = feat.detach().clone().cpu()
            self._ids.append(slot_id)
            self._n_stores += 1

    def search(
        self, query: torch.Tensor, k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search for top-k most similar memories.

        Returns:
            (indices, similarities) where ``indices`` is a ``[B, k]`` long
            tensor of FAISS ``slot_id``s and ``similarities`` is the matching
            ``[B, k]`` float tensor of cosine scores (or raw inner products if
            ``normalize_keys=False``).
        """
        if not self._ids:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0)

        with torch.no_grad():
            q = query.detach().to(torch.float32)
            single = q.dim() == 1
            if single:
                q = q.unsqueeze(0)
            q = q.to(self.device)
            if self.normalize_keys:
                q = self._normalize(q)

            k_eff = min(int(k), len(self._ids))
            D, I = self.index.search(
                q.cpu().numpy().astype(np.float32), k_eff
            )
            sims = torch.from_numpy(D).to(torch.float32)  # [B, k]
            ids = torch.from_numpy(I).to(torch.long)       # [B, k]

            if single:
                sims = sims.squeeze(0)
                ids = ids.squeeze(0)

            self._n_searches += 1
            return ids, sims

    def retrieve(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Retrieve top-k most similar memories and return ``query + mean(value)``
        (residual connection, matching :class:`EpisodicMemory`).

        Returns ``query`` unchanged if the memory is empty.
        """
        if not self._ids:
            return query

        single = query.dim() == 1
        ids, _ = self.search(query, k=k)
        if ids.numel() == 0:
            return query

        # ``ids`` is [k] for a 1-D query and [B, k] for a 2-D batched query.
        if single:
            ids_2d = ids.unsqueeze(0)                # [1, k]
        else:
            ids_2d = ids                              # [B, k]
        B, K = ids_2d.shape

        # Gather values for the returned ids.
        flat_ids = ids_2d.reshape(-1).tolist()
        rows: List[torch.Tensor] = []
        for sid in flat_ids:
            v = self._values.get(int(sid))
            rows.append(v if v is not None else torch.zeros(self.feature_dim))
        stacked = torch.stack(rows, dim=0).reshape(B, K, self.feature_dim)  # [B, k, D]
        avg = stacked.mean(dim=1)                                        # [B, D]
        if single:
            return avg.squeeze(0) + query
        return avg + query

    def forget(self, threshold: float = 0.1) -> int:
        """
        Prune low-usage memories. A memory is "used" if its mean cosine
        similarity to all other stored keys exceeds ``threshold`` (same
        heuristic as :class:`EpisodicMemory.forget`).

        Returns:
            Number of memories kept.
        """
        if len(self._ids) < 2:
            return len(self._ids)

        with torch.no_grad():
            valid_ids = list(self._ids)
            keys = torch.stack(
                [
                    self._normalize(self._values[sid].to(torch.float32))
                    if self.normalize_keys
                    else self._values[sid].to(torch.float32)
                    for sid in valid_ids
                ],
                dim=0,
            )  # [N, D]
            sims = F.cosine_similarity(
                keys.unsqueeze(1), keys.unsqueeze(0), dim=-1
            )  # [N, N]
            usage = sims.mean(dim=1)  # [N]
            keep_mask = usage > float(threshold)
            keep_ids = [
                sid for sid, keep in zip(valid_ids, keep_mask.tolist()) if keep
            ]
            remove_ids = [
                sid for sid, keep in zip(valid_ids, keep_mask.tolist()) if not keep
            ]

            if remove_ids:
                if self.use_hnsw:
                    # HNSW doesn't support remove_ids -> rebuild from keepers.
                    # Update local state first so the rebuild sees the keepers.
                    for sid in remove_ids:
                        self._values.pop(sid, None)
                    self._ids = list(keep_ids)
                    self._rebuild_inplace()
                else:
                    try:
                        self.index.remove_ids(
                            np.array(remove_ids, dtype=np.int64)
                        )
                        for sid in remove_ids:
                            self._values.pop(sid, None)
                        self._ids = list(keep_ids)
                    except Exception:
                        for sid in remove_ids:
                            self._values.pop(sid, None)
                        self._ids = list(keep_ids)
                        self._rebuild_inplace()

            return len(keep_ids)

    def _rebuild_inplace(self) -> None:
        """Rebuild the FAISS index in-place, preserving the current
        ``_ids`` ordering and ``slot_id`` -> value mapping.

        Used when the underlying FAISS index type (e.g. ``IndexHNSWFlat``)
        cannot remove a single id — we just rebuild from the survivors.
        """
        # Snapshot the current state.
        survivors = list(self._ids)
        old_values = [self._values[sid].clone() for sid in survivors]

        # Tear down and rebuild the FAISS index.
        self.index = self._new_index()

        # Re-add each survivor with the SAME slot_id (re-add into a fresh
        # IDMap must not collide since the new index starts empty).
        for sid, v in zip(survivors, old_values):
            key = self._normalize(v) if self.normalize_keys else v
            self.index.add_with_ids(
                key.reshape(1, -1).cpu().numpy().astype(np.float32),
                np.array([sid], dtype=np.int64),
            )

    def get_stats(self) -> dict:
        """Return memory statistics."""
        return {
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "size": len(self._ids),
            "index_ntotal": int(self.index.ntotal),
            "use_hnsw": self.use_hnsw,
            "normalize_keys": self.normalize_keys,
            "next_slot_id": self._next_id,
            "n_stores": int(self._n_stores.item()),
            "n_searches": int(self._n_searches.item()),
        }

    def reset(self) -> None:
        """Reset the memory to empty."""
        self.index = self._new_index()
        self._values.clear()
        self._ids = []
        self._next_id = 0
        self._n_stores.zero_()
        self._n_searches.zero_()

    def __len__(self) -> int:
        return len(self._ids)

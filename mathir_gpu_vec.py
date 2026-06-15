"""
GPU-Accelerated Vector Memory for MATHIR
=========================================

Drop-in replacement for ``VecMemory`` (sqlite-vec) that stores ALL embeddings
in a single GPU tensor and uses matrix multiplication for brute-force cosine
search.  Typically 10-100x faster than sqlite-vec on GPU, and competitive
with sqlite-vec on CPU via numpy fallback.

Key optimizations
-----------------
1. All embeddings stored as one contiguous ``[N, dim]`` float32 tensor.
2. Search = normalized matrix multiply: ``query @ embeddings.T → [1, N]``.
3. ``torch.topk`` for O(N log k) partial sort (no full sort needed).
4. Batch insert via ``torch.cat`` — single allocation.
5. Deletion via boolean mask + compact (amortised O(N) on GPU).

Interface
---------
Matches :class:`mathir_vec.VecMemory` plus extras::

    GPUVecMemory(dim=1024, device=None, capacity=0)
    .store(memory_id, embedding, metadata)
    .store_batch(ids, embeddings, metadatas)
    .search(query, k=5, agent_filter=None, block_type_filter=None)
    .delete(memory_id)
    .stats() → dict
    .close()
    # also works as context manager
"""

from __future__ import annotations

import json
import time
import threading
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch

try:
    import sqlite_vec as _sqlite_vec  # noqa: F401 — detect availability
    _HAS_SQLITE_VEC = True
except ImportError:
    _HAS_SQLITE_VEC = False


# ---------------------------------------------------------------------------
# Device detection  (reuses MATHIR's pattern from device_utils.py)
# ---------------------------------------------------------------------------

def _detect_device() -> torch.device:
    """Return ``cuda:0`` when a GPU with ≥1 GB free VRAM exists, else ``cpu``."""
    if torch.cuda.is_available():
        try:
            free_gb = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
            if free_gb >= 1.0:
                return torch.device("cuda:0")
        except Exception:
            return torch.device("cuda:0")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# GPUVecMemory
# ---------------------------------------------------------------------------

class GPUVecMemory:
    """
    GPU-accelerated vector memory using torch brute-force cosine search.

    Parameters
    ----------
    dim : int
        Embedding dimensionality (must match the model's output dim).
    device : str | torch.device | None
        Force a specific device.  ``None`` → auto-detect best available.
    capacity : int
        Soft cap on stored memories.  ``0`` means unlimited.
    """

    def __init__(
        self,
        dim: int = 1024,
        device: Optional[str | torch.device] = None,
        capacity: int = 0,
    ):
        self.dim = dim
        self.capacity = capacity

        # --- Device ---
        if device is None:
            self.device = _detect_device()
        else:
            self.device = torch.device(device)

        self._is_gpu = self.device.type == "cuda"

        # --- Storage ---
        # embeddings: [N, dim] contiguous float32 tensor on device
        self._embeddings: Optional[torch.Tensor] = None
        # metadata index: memory_id → (slot_index, metadata_dict)
        self._meta_index: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        # ordered list of memory_ids (for stable iteration / deletion)
        self._ids: List[str] = []
        # monotonically increasing slot counter
        self._next_slot: int = 0

        # --- Thread safety ---
        self._lock = threading.RLock()

        # --- Stats ---
        self._n_stores: int = 0
        self._n_searches: int = 0
        self._total_search_us: float = 0.0

    # ------------------------------------------------------------------
    # Public API — store
    # ------------------------------------------------------------------

    def store(
        self,
        memory_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a single embedding with metadata (drop-in for VecMemory)."""
        self.store_batch([memory_id], [embedding], [metadata or {}])

    def store_batch(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Bulk-insert *ids* embeddings in a single GPU allocation.

        Parameters
        ----------
        ids : list[str]
        embeddings : list[np.ndarray]  — each shape ``(dim,)`` or ``(1, dim)``
        metadatas : list[dict] | None

        Returns
        -------
        int — number of newly inserted memories.
        """
        if not ids:
            return 0

        n = len(ids)
        if metadatas is None:
            metadatas = [{}] * n
        if len(metadatas) != n:
            raise ValueError("metadatas length must match ids length")

        # Validate & convert to numpy float32
        arrs: List[np.ndarray] = []
        for i, emb in enumerate(embeddings):
            arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            if arr.shape[0] != self.dim:
                raise ValueError(
                    f"Embedding dim mismatch: expected {self.dim}, got {arr.shape[0]} "
                    f"(id={ids[i]})"
                )
            arrs.append(arr)

        new_embs = np.stack(arrs, axis=0)  # [n, dim]

        with self._lock:
            # --- Eviction (if capacity > 0) ---
            insert_ids = list(ids)
            insert_embs = new_embs

            if self.capacity > 0 and len(self._ids) + len(insert_ids) > self.capacity:
                # FIFO eviction: drop oldest entries to make room
                overflow = (len(self._ids) + len(insert_ids)) - self.capacity
                evict_ids = self._ids[:overflow]
                self._ids = self._ids[overflow:]
                # Remove evicted from meta_index
                for eid in evict_ids:
                    self._meta_index.pop(eid, None)
                # Compact embeddings tensor
                if self._embeddings is not None and overflow > 0:
                    if overflow < self._embeddings.shape[0]:
                        self._embeddings = self._embeddings[overflow:].contiguous()
                    else:
                        self._embeddings = None

            # --- Append to GPU tensor ---
            new_tensor = torch.from_numpy(insert_embs).to(self.device)
            if self._embeddings is None:
                self._embeddings = new_tensor
            else:
                self._embeddings = torch.cat([self._embeddings, new_tensor], dim=0)

            # --- Update indices ---
            for i, mid in enumerate(insert_ids):
                slot = self._next_slot + i
                self._meta_index[mid] = (slot, metadatas[i])
                self._ids.append(mid)

            self._next_slot += n
            self._n_stores += n
            return n

    # ------------------------------------------------------------------
    # Public API — search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        agent_filter: Optional[str] = None,
        block_type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Top-k cosine similarity search.

        Parameters
        ----------
        query : np.ndarray — shape ``(dim,)``
        k : int
        agent_filter : str | None — post-filter by ``metadata["agent"]``
        block_type_filter : str | None — post-filter by ``metadata["tier"]``

        Returns
        -------
        list[dict] with keys: memory_id, similarity, metadata
        """
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"Query dim mismatch: expected {self.dim}, got {q.shape[0]}")

        t0 = time.perf_counter()

        with self._lock:
            n = len(self._ids)
            if n == 0:
                self._n_searches += 1
                return []

            # --- GPU brute-force cosine similarity ---
            q_tensor = torch.from_numpy(q).to(self.device)

            # L2-normalize for cosine similarity
            q_norm = q_tensor / (q_tensor.norm() + 1e-12)
            emb_norms = self._embeddings / (
                self._embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            )

            # Similarity = dot product of unit vectors = cosine similarity
            # [1, dim] @ [dim, N] → [1, N]
            sims = torch.mm(q_norm.unsqueeze(0), emb_norms.t()).squeeze(0)  # [N]

            # If filters are needed, fetch more candidates then post-filter
            fetch_k = min(k * 10, n) if (agent_filter or block_type_filter) else min(k, n)
            topk_sims, topk_idx = torch.topk(sims, fetch_k)

            # --- Post-filter (CPU dict lookup — negligible cost) ---
            results: List[Dict[str, Any]] = []
            for i in range(fetch_k):
                slot = topk_idx[i].item()
                sim = topk_sims[i].item()

                # Map slot → memory_id
                mid = self._ids[slot] if slot < len(self._ids) else None
                if mid is None or mid not in self._meta_index:
                    continue

                _, meta = self._meta_index[mid]

                if agent_filter and meta.get("agent", "") != agent_filter:
                    continue
                if block_type_filter and meta.get("tier", "") != block_type_filter:
                    continue

                results.append({
                    "memory_id": mid,
                    "similarity": sim,
                    "metadata": meta,
                })

                if len(results) >= k:
                    break

            elapsed_us = (time.perf_counter() - t0) * 1_000_000
            self._n_searches += 1
            self._total_search_us += elapsed_us

            return results

    # ------------------------------------------------------------------
    # Public API — delete
    # ------------------------------------------------------------------

    def delete(self, memory_id: str) -> bool:
        """
        Remove a single memory by id.

        Returns ``True`` if the memory was found and removed, ``False`` otherwise.

        Deletion compacts the embeddings tensor (GPU memcpy) — O(N) but
        happens infrequently compared to search.
        """
        with self._lock:
            if memory_id not in self._meta_index:
                return False

            # Find the actual position in _ids (the source of truth for row mapping).
            # After prior deletions the slot stored in _meta_index may be stale.
            try:
                pos = self._ids.index(memory_id)
            except ValueError:
                # Should not happen if _meta_index is consistent, but be safe.
                self._meta_index.pop(memory_id, None)
                return False

            # Remove from ordered id list and meta_index.
            self._ids.pop(pos)
            self._meta_index.pop(memory_id, None)

            # Compact the embeddings tensor by removing row `pos`.
            if self._embeddings is not None and pos < self._embeddings.shape[0]:
                mask = torch.ones(
                    self._embeddings.shape[0], dtype=torch.bool, device=self.device
                )
                mask[pos] = False
                self._embeddings = self._embeddings[mask].contiguous()

            # Rebuild _meta_index slot mapping (all slots shifted after deletion).
            self._next_slot = len(self._ids)
            new_index: Dict[str, Tuple[int, Dict[str, Any]]] = {}
            for i, mid in enumerate(self._ids):
                _, meta = self._meta_index.get(mid, (i, {}))
                new_index[mid] = (i, meta)
            self._meta_index = new_index

            return True

    # ------------------------------------------------------------------
    # Public API — stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        with self._lock:
            n = len(self._ids)
            mem_bytes = 0
            if self._embeddings is not None:
                mem_bytes = self._embeddings.nelement() * self._embeddings.element_size()

            avg_search_us = (
                self._total_search_us / self._n_searches
                if self._n_searches > 0
                else 0.0
            )

            return {
                "count": n,
                "dim": self.dim,
                "device": str(self.device),
                "is_gpu": self._is_gpu,
                "capacity": self.capacity,
                "memory_bytes": mem_bytes,
                "memory_mb": round(mem_bytes / (1024 ** 2), 2),
                "n_stores": self._n_stores,
                "n_searches": self._n_searches,
                "avg_search_us": round(avg_search_us, 1),
            }

    # ------------------------------------------------------------------
    # Public API — reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored memories."""
        with self._lock:
            self._embeddings = None
            self._meta_index.clear()
            self._ids.clear()
            self._next_slot = 0
            self._n_stores = 0
            self._n_searches = 0
            self._total_search_us = 0.0

    # ------------------------------------------------------------------
    # Context manager / close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release GPU memory."""
        self.reset()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)

    def __repr__(self) -> str:
        return (
            f"GPUVecMemory(dim={self.dim}, device={self.device}, "
            f"count={len(self)}, capacity={self.capacity})"
        )


# ---------------------------------------------------------------------------
# Drop-in compatibility shim: same constructor signature as VecMemory
# ---------------------------------------------------------------------------

class GPUVecMemoryCompat(GPUVecMemory):
    """
    Exact drop-in for ``VecMemory(db_path, dim)``.

    The ``db_path`` argument is accepted but ignored — GPU memory is
    ephemeral (not persisted to disk).  If you need persistence, call
    ``save()`` / ``load()`` to serialize to disk.
    """

    def __init__(self, db_path: str = "mathir_gpu.db", dim: int = 1024):
        super().__init__(dim=dim)
        self.db_path = db_path

    def save(self, path: Optional[str] = None) -> None:
        """Persist embeddings + metadata to disk as a ``.pt`` file."""
        path = path or self.db_path.replace(".db", ".pt")
        with self._lock:
            data = {
                "dim": self.dim,
                "embeddings": self._embeddings.cpu() if self._embeddings is not None else None,
                "ids": self._ids,
                "meta": {k: (s, m) for k, (s, m) in self._meta_index.items()},
                "next_slot": self._next_slot,
            }
            torch.save(data, path)

    def load(self, path: Optional[str] = None) -> None:
        """Load previously saved state from a ``.pt`` file."""
        path = path or self.db_path.replace(".db", ".pt")
        data = torch.load(path, map_location=self.device, weights_only=False)
        with self._lock:
            self.dim = data["dim"]
            self._embeddings = (
                data["embeddings"].to(self.device)
                if data["embeddings"] is not None
                else None
            )
            self._ids = data["ids"]
            self._meta_index = {k: v for k, v in data["meta"].items()}
            self._next_slot = data["next_slot"]


__all__ = ["GPUVecMemory", "GPUVecMemoryCompat"]

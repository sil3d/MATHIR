"""GPU-Accelerated Vector Memory for MATHIR — torch brute-force cosine search."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_device() -> torch.device:
    """Return cuda:0 if ≥1 GB free VRAM, else cpu."""
    if torch.cuda.is_available():
        try:
            if torch.cuda.mem_get_info(0)[0] / (1024 ** 3) >= 1.0:
                return torch.device("cuda:0")
        except Exception:
            return torch.device("cuda:0")
    return torch.device("cpu")


def _torch_normalize(emb: torch.Tensor) -> torch.Tensor:
    """L2-normalize last dimension (in-place safe, returns new tensor)."""
    return emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# ---------------------------------------------------------------------------
# GPUVecMemory
# ---------------------------------------------------------------------------

class GPUVecMemory:
    """GPU-accelerated vector memory using torch brute-force cosine similarity.

    All embeddings stored as one contiguous [N, dim] float32 tensor.
    Search = normalized matrix multiply + torch.topk.

    Parameters
    ----------
    dim : int — embedding dimensionality.
    device : str | torch.device | None — force device, None → auto-detect.
    capacity : int — soft cap (0 = unlimited).
    """

    def __init__(self, dim: int = 1024,
                 device: Optional[str | torch.device] = None,
                 capacity: int = 0):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        if not isinstance(capacity, int) or capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity!r}")
        self.dim = dim
        self.capacity = capacity
        self.device = torch.device(device) if device else _detect_device()
        self._is_gpu = self.device.type == "cuda"
        self._embeddings: Optional[torch.Tensor] = None
        self._meta_index: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        self._ids: List[str] = []
        self._next_slot = 0
        self._lock = threading.RLock()
        self._n_stores = 0
        self._n_searches = 0
        self._total_search_us = 0.0

    # -- Store ---------------------------------------------------------------

    def store(self, memory_id: str, embedding: np.ndarray,
              metadata: Optional[Dict[str, Any]] = None) -> None:
        self.store_batch([memory_id], [embedding], [metadata or {}])

    def store_batch(self, ids: List[str], embeddings: List[np.ndarray],
                    metadatas: Optional[List[Dict[str, Any]]] = None) -> int:
        if not ids:
            return 0
        n = len(ids)
        for i, mid in enumerate(ids):
            if not isinstance(mid, str) or not mid.strip():
                raise ValueError(f"ids[{i}] must be non-empty string, got {mid!r}")
        if metadatas is None:
            metadatas = [{}] * n
        if len(metadatas) != n:
            raise ValueError("metadatas length must match ids length")

        arrs = []
        for i, emb in enumerate(embeddings):
            arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            if arr.shape[0] != self.dim:
                raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {arr.shape[0]} (id={ids[i]})")
            arrs.append(arr)
        new_embs = np.stack(arrs, axis=0)

        with self._lock:
            # FIFO eviction
            insert_ids = list(ids)
            if self.capacity > 0 and len(self._ids) + len(insert_ids) > self.capacity:
                overflow = (len(self._ids) + len(insert_ids)) - self.capacity
                for eid in self._ids[:overflow]:
                    self._meta_index.pop(eid, None)
                self._ids = self._ids[overflow:]
                if self._embeddings is not None:
                    self._embeddings = (
                        self._embeddings[overflow:].contiguous() if overflow < self._embeddings.shape[0]
                        else None
                    )

            # Append to tensor
            new_tensor = torch.from_numpy(new_embs).to(self.device)
            self._embeddings = (
                new_tensor if self._embeddings is None
                else torch.cat([self._embeddings, new_tensor], dim=0)
            )
            for i, mid in enumerate(insert_ids):
                self._meta_index[mid] = (self._next_slot + i, metadatas[i])
                self._ids.append(mid)
            self._next_slot += n
            self._n_stores += n
            return n

    # -- Search --------------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 5,
               agent_filter: Optional[str] = None,
               block_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"Query dim mismatch: expected {self.dim}, got {q.shape[0]}")

        t0 = time.perf_counter()
        with self._lock:
            n = len(self._ids)
            if n == 0:
                self._n_searches += 1
                return []

            # GPU brute-force cosine similarity
            q_norm = _torch_normalize(torch.from_numpy(q).to(self.device))
            emb_norm = _torch_normalize(self._embeddings)
            sims = torch.mm(q_norm.unsqueeze(0), emb_norm.t()).squeeze(0)

            has_filter = agent_filter or block_type_filter
            fetch_k = min(k * 10, n) if has_filter else min(k, n)
            topk_sims, topk_idx = torch.topk(sims, fetch_k)

            results: List[Dict[str, Any]] = []
            for i in range(fetch_k):
                slot = topk_idx[i].item()
                mid = self._ids[slot] if slot < len(self._ids) else None
                if mid is None or mid not in self._meta_index:
                    continue
                _, meta = self._meta_index[mid]
                if agent_filter and meta.get("agent", "") != agent_filter:
                    continue
                if block_type_filter and meta.get("tier", "") != block_type_filter:
                    continue
                results.append({"memory_id": mid, "similarity": topk_sims[i].item(), "metadata": meta})
                if len(results) >= k:
                    break

            elapsed_us = (time.perf_counter() - t0) * 1_000_000
            self._n_searches += 1
            self._total_search_us += elapsed_us
            return results

    # -- Delete --------------------------------------------------------------

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            if memory_id not in self._meta_index:
                return False
            try:
                pos = self._ids.index(memory_id)
            except ValueError:
                self._meta_index.pop(memory_id, None)
                return False
            self._ids.pop(pos)
            self._meta_index.pop(memory_id, None)
            if self._embeddings is not None and pos < self._embeddings.shape[0]:
                mask = torch.ones(self._embeddings.shape[0], dtype=torch.bool, device=self.device)
                mask[pos] = False
                self._embeddings = self._embeddings[mask].contiguous()
            # Rebuild slot mapping (positions shifted)
            self._next_slot = len(self._ids)
            self._meta_index = {
                mid: (i, self._meta_index.get(mid, (i, {}))[1])
                for i, mid in enumerate(self._ids)
            }
            return True

    # -- Persistence (save / load) -------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist embeddings + metadata to disk as a .pt file."""
        resolved = Path(path).resolve() if path else Path(self.db_path.replace(".db", ".pt"))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            torch.save({
                "dim": self.dim,
                "embeddings": self._embeddings.cpu() if self._embeddings is not None else None,
                "ids": self._ids,
                "meta": dict(self._meta_index),
                "next_slot": self._next_slot,
            }, str(resolved))

    def load(self, path: Optional[str] = None) -> None:
        """Load previously saved state (weights_only=True for security)."""
        resolved = Path(path).resolve() if path else Path(self.db_path.replace(".db", ".pt"))
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        data = torch.load(str(resolved), map_location=self.device, weights_only=True)
        with self._lock:
            self.dim = data["dim"]
            self._embeddings = data["embeddings"].to(self.device) if data["embeddings"] is not None else None
            self._ids = data["ids"]
            self._meta_index = dict(data["meta"])
            self._next_slot = data["next_slot"]

    @property
    def db_path(self) -> str:
        return getattr(self, "_db_path", "mathir_gpu.db")

    @db_path.setter
    def db_path(self, val: str):
        self._db_path = val

    # -- Stats / lifecycle ---------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            n = len(self._ids)
            mem_bytes = self._embeddings.nelement() * self._embeddings.element_size() if self._embeddings is not None else 0
            avg_us = self._total_search_us / self._n_searches if self._n_searches > 0 else 0.0
            return {"count": n, "dim": self.dim, "device": str(self.device),
                    "is_gpu": self._is_gpu, "capacity": self.capacity,
                    "memory_bytes": mem_bytes, "memory_mb": round(mem_bytes / (1024**2), 2),
                    "n_stores": self._n_stores, "n_searches": self._n_searches,
                    "avg_search_us": round(avg_us, 1)}

    def reset(self) -> None:
        with self._lock:
            self._embeddings = None
            self._meta_index.clear()
            self._ids.clear()
            self._next_slot = 0
            self._n_stores = 0
            self._n_searches = 0
            self._total_search_us = 0.0

    def close(self) -> None:
        self.reset()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)

    def __repr__(self) -> str:
        return f"GPUVecMemory(dim={self.dim}, device={self.device}, count={len(self)}, capacity={self.capacity})"


__all__ = ["GPUVecMemory"]

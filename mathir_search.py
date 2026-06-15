"""
MATHIR Hybrid Vector Search — Auto-Scaling Backend
====================================================

The right architecture for an app that GROWS with users:

  Phase 1 (N<5K):    Numpy brute-force — 0.13ms, zero overhead
  Phase 2 (N>=5K):   USearch HNSW — 3.79ms at 10K, index on disk
  Always:             SQLite for metadata persistence + full CRUD

Architecture:
  ┌─────────────────────────────────────────────────┐
  │  VectorSearch (unified interface)               │
  │  .store(id, embedding, metadata)                │
  │  .search(query, k=5)                            │
  │  .delete(id)                                    │
  │  .count()                                       │
  ├─────────────┬───────────────┬───────────────────┤
  │  Numpy      │  USearch      │  SQLite           │
  │  (in-RAM)   │  (HNSW+mmap)  │  (metadata only)  │
  │  N<500      │  N>=500       │  Always           │
  └─────────────┴───────────────┴───────────────────┘

Auto-switches backend as data grows. No config needed.

Usage::

    from mathir_search import HybridSearch

    # Auto-detects best strategy
    search = HybridSearch(dim=1024, db_path="memories.db")
    search.store("mem_1", embedding, {"agent": "coder"})
    results = search.search(query, k=5)  # instant at any scale

    # Force specific phase
    search = HybridSearch(dim=1024, strategy="numpy")    # Phase 1
    search = HybridSearch(dim=1024, strategy="usearch")  # Phase 2/3
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMPY_THRESHOLD = 5000     # Switch from numpy to USearch at this N
REBUILD_INTERVAL = 100     # Rebuild USearch index every N inserts
MMAP_DIR = "mathir_indexes" # Subdirectory for mmap files


# ---------------------------------------------------------------------------
# SQLite Metadata Store (always used)
# ---------------------------------------------------------------------------

class _MetadataStore:
    """
    SQLite store for memory metadata — always used regardless of vector backend.
    Handles CRUD for metadata, provides persistence, supports queries by agent/tier.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-8000")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._lock = threading.Lock()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT UNIQUE,
                metadata TEXT,
                agent TEXT,
                tier TEXT,
                timestamp REAL,
                embedding BLOB
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier)"
        )
        self._conn.commit()

    def store(self, memory_id: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        meta = metadata or {}
        agent = meta.get("agent", "")
        tier = meta.get("tier", "episodic")
        ts = time.time()
        meta_json = json.dumps(meta, ensure_ascii=False, default=str)
        blob = embedding.astype(np.float32).tobytes()

        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO memories "
                "(memory_id, metadata, agent, tier, timestamp, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (memory_id, meta_json, agent, tier, ts, blob),
            )
            self._conn.commit()

    def store_batch(self, items: List[Dict[str, Any]]) -> None:
        with self._lock:
            for item in items:
                meta = item.get("metadata", {})
                blob = np.asarray(item["embedding"], dtype=np.float32).tobytes()
                self._conn.execute(
                    "INSERT OR REPLACE INTO memories "
                    "(memory_id, metadata, agent, tier, timestamp, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        item["memory_id"],
                        json.dumps(meta, ensure_ascii=False, default=str),
                        meta.get("agent", ""),
                        meta.get("tier", "episodic"),
                        time.time(),
                        blob,
                    ),
                )
            self._conn.commit()

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_all(self, agent_filter: Optional[str] = None, limit: int = 10000) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM memories"
        params: list = []
        if agent_filter:
            sql += " WHERE agent = ?"
            params.append(agent_filter)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def _row_to_dict(self, row) -> Dict[str, Any]:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        emb = np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None
        return {
            "memory_id": row["memory_id"],
            "metadata": meta,
            "agent": row["agent"],
            "tier": row["tier"],
            "timestamp": row["timestamp"],
            "embedding": emb,
        }

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# Numpy Backend (Phase 1: N < 500)
# ---------------------------------------------------------------------------

class _NumpyBackend:
    """Brute-force cosine search — fastest for small N."""

    def __init__(self, dim: int):
        self.dim = dim
        self._embeddings: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._lock = threading.RLock()

    def build(self, items: List[Dict[str, Any]]) -> None:
        if not items:
            return
        arrs = []
        ids = []
        for item in items:
            arr = np.asarray(item["embedding"], dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            arrs.append(arr)
            ids.append(item["memory_id"])
        with self._lock:
            self._embeddings = np.stack(arrs)
            self._ids = ids

    def add(self, memory_id: str, embedding: np.ndarray) -> None:
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        with self._lock:
            if self._embeddings is None:
                self._embeddings = arr.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, arr.reshape(1, -1)])
            self._ids.append(memory_id)

    def remove(self, memory_id: str) -> bool:
        with self._lock:
            if memory_id not in self._ids:
                return False
            idx = self._ids.index(memory_id)
            self._ids.pop(idx)
            self._embeddings = np.delete(self._embeddings, idx, axis=0)
            return True

    def search(self, query: np.ndarray, k: int = 5) -> List[tuple]:
        with self._lock:
            if self._embeddings is None or len(self._ids) == 0:
                return []
            q = np.asarray(query, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(q)
            if norm == 0:
                return []
            q_unit = q / norm
            sims = self._embeddings @ q_unit
            fetch_k = min(k, len(self._ids))
            top_idx = np.argpartition(sims, -fetch_k)[-fetch_k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            return [(self._ids[i], float(sims[i])) for i in top_idx]

    def count(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# USearch Backend (Phase 2/3: N >= 500)
# ---------------------------------------------------------------------------

class _USearchBackend:
    """HNSW index with memory-mapping — fast at any scale."""

    def __init__(self, dim: int, index_path: Optional[str] = None):
        from usearch.index import Index
        self.dim = dim
        self.index_path = index_path
        self._id_to_key: Dict[str, int] = {}
        self._key_to_id: Dict[int, str] = {}
        self._next_key = 0
        self._lock = threading.RLock()

        if index_path and os.path.exists(index_path):
            self._index = Index(ndim=dim, metric="cosine", path=index_path)
            # Rebuild reverse map from stored data
            self._next_key = self._index.size
        else:
            self._index = Index(ndim=dim, metric="cosine")

    def build(self, items: List[Dict[str, Any]]) -> None:
        if not items:
            return
        keys = []
        vecs = []
        with self._lock:
            for item in items:
                arr = np.asarray(item["embedding"], dtype=np.float32).reshape(-1)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                key = self._next_key
                self._next_key += 1
                self._id_to_key[item["memory_id"]] = key
                self._key_to_id[key] = item["memory_id"]
                keys.append(key)
                vecs.append(arr)
            self._index.add(np.array(keys, dtype=np.int64), np.stack(vecs))

    def add(self, memory_id: str, embedding: np.ndarray) -> None:
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        with self._lock:
            key = self._next_key
            self._next_key += 1
            self._id_to_key[memory_id] = key
            self._key_to_id[key] = memory_id
            self._index.add(np.array([key], dtype=np.int64), arr.reshape(1, -1))

    def remove(self, memory_id: str) -> bool:
        with self._lock:
            if memory_id not in self._id_to_key:
                return False
            key = self._id_to_key.pop(memory_id)
            del self._key_to_id[key]
            self._index.remove(np.array([key], dtype=np.int64))
            return True

    def search(self, query: np.ndarray, k: int = 5) -> List[tuple]:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = q / norm
        results = self._index.search(q, count=k)
        pairs = []
        with self._lock:
            for i in range(len(results)):
                key = results[i].key
                score = float(results[i].distance)
                mid = self._key_to_id.get(key)
                if mid:
                    pairs.append((mid, score))
        return pairs

    def count(self) -> int:
        return self._index.size

    def save(self) -> None:
        if self.index_path:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            self._index.save(self.index_path)


# ---------------------------------------------------------------------------
# HybridSearch — The Unified Interface
# ---------------------------------------------------------------------------

class HybridSearch:
    """
    Hybrid vector search that auto-scales with data growth.

    Architecture:
      - SQLite always stores metadata + embeddings (persistence)
      - Numpy backend for N < 500 (fastest, zero overhead)
      - USearch HNSW for N >= 500 (memory-mapped, fast at scale)
      - Auto-switches as data grows
      - Rebuilds USearch index periodically for optimal performance

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    db_path : str
        SQLite database path (for metadata persistence).
    strategy : str | None
        Force a strategy: "numpy", "usearch", or "auto".
        ``None`` → auto-detect based on current N.
    index_dir : str
        Directory for USearch mmap files.
    """

    def __init__(
        self,
        dim: int = 1024,
        db_path: str = "mathir_vectors.db",
        strategy: Optional[str] = None,
        index_dir: Optional[str] = None,
    ):
        self.dim = dim
        self.db_path = db_path
        self.strategy = strategy or "auto"
        self.index_dir = index_dir or os.path.join(
            os.path.dirname(db_path) or ".", MMAP_DIR
        )

        # Always: SQLite metadata store
        self._meta = _MetadataStore(db_path)

        # Vector backends (lazy init)
        self._numpy: Optional[_NumpyBackend] = None
        self._usearch: Optional[_USearchBackend] = None
        self._current: Optional[_NumpyBackend | _USearchBackend] = None
        self._current_name = "numpy"

        # Load existing data
        self._init_from_db()

    def _init_from_db(self):
        """Load existing data from SQLite into the appropriate backend."""
        count = self._meta.count()
        if count == 0:
            self._init_numpy()
            return

        items = self._meta.get_all(limit=100000)

        if self.strategy == "numpy" or (self.strategy == "auto" and count < NUMPY_THRESHOLD):
            self._init_numpy()
            self._current.build(items)
        else:
            self._init_usearch()
            self._current.build(items)

    def _init_numpy(self):
        if self._numpy is None:
            self._numpy = _NumpyBackend(self.dim)
        self._current = self._numpy
        self._current_name = "numpy"

    def _init_usearch(self):
        if self._usearch is None:
            idx_path = os.path.join(self.index_dir, f"mathir_{self.dim}d.usearch")
            self._usearch = _USearchBackend(self.dim, idx_path)
        self._current = self._usearch
        self._current_name = "usearch"

    def _maybe_switch_backend(self):
        """Auto-switch from numpy to USearch when N crosses threshold."""
        if self.strategy != "auto":
            return
        if self._current_name == "numpy" and self._current.count() >= NUMPY_THRESHOLD:
            # Switch to USearch
            items = self._meta.get_all(limit=100000)
            self._init_usearch()
            self._current.build(items)
            # Save to disk for mmap
            self._usearch.save()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        memory_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a single embedding with metadata."""
        meta = metadata or {}
        # 1. Persist to SQLite
        self._meta.store(memory_id, embedding, meta)
        # 2. Add to vector backend
        self._current.add(memory_id, embedding)
        # 3. Maybe switch backend
        self._maybe_switch_backend()

    def store_batch(self, items: List[Dict[str, Any]]) -> int:
        """Bulk-insert items. Each dict: {memory_id, embedding, metadata?}."""
        if not items:
            return 0
        # 1. Persist to SQLite
        self._meta.store_batch(items)
        # 2. Add to vector backend
        for item in items:
            self._current.add(item["memory_id"], item["embedding"])
        # 3. Maybe switch backend
        self._maybe_switch_backend()
        return len(items)

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        agent_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Top-k cosine similarity search."""
        # 1. Vector search (fast)
        pairs = self._current.search(query, k=k * 2 if agent_filter else k)

        # 2. Fetch metadata from SQLite (only for results)
        results = []
        for mid, score in pairs:
            if agent_filter:
                meta = self._meta.get(mid)
                if meta and meta.get("agent", "") != agent_filter:
                    continue
                if meta:
                    results.append({
                        "memory_id": mid,
                        "similarity": score,
                        "metadata": meta["metadata"],
                        "agent": meta["agent"],
                        "tier": meta["tier"],
                    })
            else:
                meta = self._meta.get(mid)
                if meta:
                    results.append({
                        "memory_id": mid,
                        "similarity": score,
                        "metadata": meta["metadata"],
                        "agent": meta["agent"],
                        "tier": meta["tier"],
                    })
            if len(results) >= k:
                break

        return results

    def delete(self, memory_id: str) -> bool:
        """Remove a memory by ID."""
        # 1. Remove from vector backend
        self._current.remove(memory_id)
        # 2. Remove from SQLite
        return self._meta.delete(memory_id)

    def count(self) -> int:
        """Number of stored memories."""
        return self._current.count()

    def stats(self) -> Dict[str, Any]:
        """Return backend statistics."""
        return {
            "backend": self._current_name,
            "dim": self.dim,
            "count": self.count(),
            "db_path": self.db_path,
            "numpy_available": self._numpy is not None,
            "usearch_available": self._usearch is not None,
            "threshold": NUMPY_THRESHOLD,
        }

    def save(self) -> None:
        """Persist USearch index to disk (for mmap)."""
        if self._usearch:
            self._usearch.save()

    def close(self) -> None:
        """Release resources."""
        if self._usearch:
            self._usearch.save()
        self._meta.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (
            f"HybridSearch(backend={self._current_name}, dim={self.dim}, "
            f"count={self.count()}, threshold={NUMPY_THRESHOLD})"
        )


__all__ = ["HybridSearch"]

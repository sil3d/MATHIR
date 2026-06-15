"""MATHIR Hybrid Vector Search — Auto-Scaling Backend (Numpy / USearch / SQLite)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

NUMPY_THRESHOLD = 5000
MMAP_DIR = "mathir_indexes"


def _normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D float32 vector."""
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def _make_result(memory_id: str, score: float, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {"memory_id": memory_id, "similarity": score,
            "metadata": meta.get("metadata", {}),
            "agent": meta.get("agent", ""), "tier": meta.get("tier", "episodic")}


# ---------------------------------------------------------------------------
# Numpy Backend (N < NUMPY_THRESHOLD)
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
        with self._lock:
            self._embeddings = np.stack([_normalize(it["embedding"]) for it in items])
            self._ids = [it["memory_id"] for it in items]

    def add(self, memory_id: str, embedding: np.ndarray) -> None:
        arr = _normalize(embedding)
        with self._lock:
            self._embeddings = (
                arr.reshape(1, -1) if self._embeddings is None
                else np.vstack([self._embeddings, arr.reshape(1, -1)]))
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
            if self._embeddings is None or not self._ids:
                return []
            q = _normalize(query)
            sims = self._embeddings @ q
            top_idx = np.argpartition(sims, -k)[-k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            return [(self._ids[i], float(sims[i])) for i in top_idx]

    def count(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# USearch Backend (N >= NUMPY_THRESHOLD)
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
            self._next_key = self._index.size
        else:
            # Higher connectivity + expansion → better recall for 1024d embeddings
            self._index = Index(
                ndim=dim,
                metric="cosine",
                connectivity=32,        # M=32 (default ~16)
                expansion_add=256,      # ef_construction=256 (default ~128)
                expansion_search=128,   # ef=128 at search time (default ~64)
            )

    def build(self, items: List[Dict[str, Any]]) -> None:
        if not items:
            return
        keys, vecs = [], []
        with self._lock:
            for it in items:
                key = self._next_key; self._next_key += 1
                self._id_to_key[it["memory_id"]] = key
                self._key_to_id[key] = it["memory_id"]
                keys.append(key); vecs.append(_normalize(it["embedding"]))
            self._index.add(np.array(keys, dtype=np.int64), np.stack(vecs))

    def add(self, memory_id: str, embedding: np.ndarray) -> None:
        arr = _normalize(embedding)
        with self._lock:
            key = self._next_key; self._next_key += 1
            self._id_to_key[memory_id] = key; self._key_to_id[key] = memory_id
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
        results = self._index.search(_normalize(query), count=k)
        with self._lock:
            return [(self._key_to_id[results[i].key], float(results[i].distance))
                    for i in range(len(results)) if results[i].key in self._key_to_id]

    def count(self) -> int:
        return self._index.size

    def save(self) -> None:
        if self.index_path:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            self._index.save(self.index_path)


# ---------------------------------------------------------------------------
# HybridSearch — Unified Interface
# ---------------------------------------------------------------------------

class HybridSearch:
    """Auto-scaling vector search: numpy (small N) → USearch HNSW (large N) + SQLite.

    Parameters: dim, db_path, strategy ("auto"|"numpy"|"usearch"|None), index_dir.
    """

    _SQL = {
        "create": ("CREATE TABLE IF NOT EXISTS memories ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT UNIQUE, "
                    "metadata TEXT, agent TEXT, tier TEXT, timestamp REAL, embedding BLOB)"),
        "insert": ("INSERT OR REPLACE INTO memories "
                   "(memory_id, metadata, agent, tier, timestamp, embedding) VALUES (?,?,?,?,?,?)"),
        "get": "SELECT * FROM memories WHERE memory_id=?",
        "get_all": "SELECT * FROM memories",
        "delete": "DELETE FROM memories WHERE memory_id=?",
        "count": "SELECT COUNT(*) FROM memories",
        "idx_agent": "CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent)",
        "idx_tier": "CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier)",
    }

    def __init__(self, dim: int = 1024, db_path: str = "mathir_vectors.db",
                 strategy: Optional[str] = None, index_dir: Optional[str] = None):
        self.dim = dim
        self.db_path = db_path
        self.strategy = strategy or "auto"
        self.index_dir = index_dir or os.path.join(os.path.dirname(db_path) or ".", MMAP_DIR)
        self._numpy: Optional[_NumpyBackend] = None
        self._usearch: Optional[_USearchBackend] = None
        self._current: Optional[_NumpyBackend | _USearchBackend] = None
        self._current_name = "numpy"
        # SQLite — always-on metadata store
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL"); self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-8000"); self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self._SQL["create"])
        self._conn.execute(self._SQL["idx_agent"]); self._conn.execute(self._SQL["idx_tier"])
        self._conn.commit()
        self._lock = threading.Lock()
        self._init_from_db()

    def _meta_store(self, memory_id: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        meta = metadata or {}
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
        with self._lock:
            self._conn.execute(self._SQL["insert"],
                (memory_id, json.dumps(meta, ensure_ascii=False, default=str),
                 meta.get("agent", ""), meta.get("tier", "episodic"), time.time(), blob))
            self._conn.commit()

    def _meta_store_batch(self, items: List[Dict[str, Any]]):
        with self._lock:
            for it in items:
                meta = it.get("metadata", {})
                blob = np.asarray(it["embedding"], dtype=np.float32).tobytes()
                self._conn.execute(self._SQL["insert"],
                    (it["memory_id"], json.dumps(meta, ensure_ascii=False, default=str),
                     meta.get("agent", ""), meta.get("tier", "episodic"), time.time(), blob))
            self._conn.commit()

    def _meta_get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(self._SQL["get"], (memory_id,)).fetchone()
        if not row:
            return None
        return {"memory_id": row["memory_id"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "agent": row["agent"], "tier": row["tier"],
                "timestamp": row["timestamp"],
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None}

    def _meta_get_all(self, limit: int = 100_000) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            self._SQL["get_all"] + " ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [{"memory_id": r["memory_id"],
                 "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                 "agent": r["agent"], "tier": r["tier"], "timestamp": r["timestamp"],
                 "embedding": np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None}
                for r in rows]

    def _meta_delete(self, memory_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(self._SQL["delete"], (memory_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def _meta_count(self) -> int:
        return self._conn.execute(self._SQL["count"]).fetchone()[0]

    def _init_from_db(self):
        count = self._meta_count()
        use_numpy = self.strategy == "numpy" or (self.strategy == "auto" and count < NUMPY_THRESHOLD)
        if use_numpy or count == 0:
            if self._numpy is None:
                self._numpy = _NumpyBackend(self.dim)
            self._current = self._numpy; self._current_name = "numpy"
        else:
            if self._usearch is None:
                self._usearch = _USearchBackend(self.dim, os.path.join(self.index_dir, f"mathir_{self.dim}d.usearch"))
            self._current = self._usearch; self._current_name = "usearch"
        if count > 0:
            self._current.build(self._meta_get_all())

    def _maybe_switch(self):
        if self.strategy != "auto" or self._current_name != "numpy":
            return
        if self._current.count() >= NUMPY_THRESHOLD:
            if self._usearch is None:
                self._usearch = _USearchBackend(self.dim, os.path.join(self.index_dir, f"mathir_{self.dim}d.usearch"))
            self._current = self._usearch; self._current_name = "usearch"
            self._current.build(self._meta_get_all()); self._usearch.save()

    # -- Public API ----------------------------------------------------------

    def store(self, memory_id: str, embedding: np.ndarray,
              metadata: Optional[Dict[str, Any]] = None) -> None:
        self._meta_store(memory_id, embedding, metadata or {})
        self._current.add(memory_id, embedding)
        self._maybe_switch()

    def store_batch(self, items: List[Dict[str, Any]]) -> int:
        if not items:
            return 0
        self._meta_store_batch(items)
        for it in items:
            self._current.add(it["memory_id"], it["embedding"])
        self._maybe_switch()
        return len(items)

    def search(self, query: np.ndarray, k: int = 5,
               agent_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        fetch_k = min(k * 2, self._current.count()) if agent_filter else min(k, self._current.count())
        pairs = self._current.search(query, k=fetch_k)
        results = []
        for mid, score in pairs:
            meta = self._meta_get(mid)
            if meta and (not agent_filter or meta.get("agent", "") == agent_filter):
                results.append(_make_result(mid, score, meta))
                if len(results) >= k:
                    break
        return results

    def delete(self, memory_id: str) -> bool:
        self._current.remove(memory_id)
        return self._meta_delete(memory_id)

    def count(self) -> int:
        return self._current.count()

    def stats(self) -> Dict[str, Any]:
        return {"backend": self._current_name, "dim": self.dim, "count": self.count(),
                "db_path": self.db_path, "threshold": NUMPY_THRESHOLD}

    def save(self) -> None:
        if self._usearch:
            self._usearch.save()

    def close(self) -> None:
        if self._usearch:
            self._usearch.save()
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"HybridSearch(backend={self._current_name}, dim={self.dim}, count={self.count()})"


__all__ = ["HybridSearch"]

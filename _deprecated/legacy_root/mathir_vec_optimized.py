"""MATHIR VecMemory — Optimized sqlite-vec backend with WAL, batch insert, and query cache."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import sqlite_vec
except ImportError:
    raise ImportError("pip install sqlite-vec")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize a float32 vector."""
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    return (arr / norm).astype(np.float32) if norm > 0 else arr.copy()


# ---------------------------------------------------------------------------
# VecMemory — Optimized (same public interface)
# ---------------------------------------------------------------------------

class VecMemory:
    """High-performance vector memory backed by sqlite-vec.

    WAL mode + single connection (no pool overhead for sequential access)
    + LRU dict cache for repeated queries.

    Public API: store, store_batch, search, delete, get_all, stats, close.
    """

    def __init__(self, db_path: str, dim: int):
        self.db_path = db_path
        self.dim = dim
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._initialized = False
        # Simple LRU cache: dict key → (result, timestamp)
        self._cache: Dict[str, tuple] = {}
        self._cache_capacity = 512
        self._cache_ttl = 120.0
        self._cache_hits = 0
        self._cache_misses = 0

    # -- Lazy init -----------------------------------------------------------

    def _ensure_init(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")
            conn.row_factory = sqlite3.Row
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT UNIQUE, metadata TEXT,
                    agent TEXT, tier TEXT, timestamp REAL
                )""")
            conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(embedding float[{self.dim}])")
            conn.commit()
            self._conn = conn
            self._initialized = True

    # -- Cache helpers -------------------------------------------------------

    def _cache_get(self, key: str):
        entry = self._cache.get(key)
        if entry and (time.monotonic() - entry[1]) < self._cache_ttl:
            self._cache_hits += 1
            return entry[0]
        if entry:
            del self._cache[key]
        self._cache_misses += 1
        return None

    def _cache_put(self, key: str, value):
        if key in self._cache:
            del self._cache[key]
        elif len(self._cache) >= self._cache_capacity:
            # Evict oldest (first inserted)
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (value, time.monotonic())

    # -- Store ---------------------------------------------------------------

    def store(self, memory_id: str, embedding: np.ndarray,
              metadata: Optional[Dict[str, Any]] = None):
        self._ensure_init()
        unit = _normalize(embedding)
        meta = metadata or {}
        ts = time.time()
        meta_json = json.dumps(meta, ensure_ascii=False, default=str)
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR REPLACE INTO memories (memory_id, metadata, agent, tier, timestamp) VALUES (?,?,?,?,?)",
                (memory_id, meta_json, meta.get("agent", ""), meta.get("tier", "episodic"), ts))
            self._conn.execute("INSERT OR REPLACE INTO memory_vec (rowid, embedding) VALUES (?,?)",
                               (cur.lastrowid, unit.tobytes()))
            self._conn.commit()
        self._cache.clear()

    def store_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        self._ensure_init()
        if not items:
            return []
        with self._lock:
            for it in items:
                unit = _normalize(it["embedding"])
                meta = it.get("metadata") or {}
                meta_json = json.dumps(meta, ensure_ascii=False, default=str)
                cur = self._conn.execute(
                    "INSERT OR REPLACE INTO memories (memory_id, metadata, agent, tier, timestamp) VALUES (?,?,?,?,?)",
                    (it["memory_id"], meta_json, meta.get("agent", ""), meta.get("tier", "episodic"), time.time()))
                self._conn.execute("INSERT OR REPLACE INTO memory_vec (rowid, embedding) VALUES (?,?)",
                                   (cur.lastrowid, unit.tobytes()))
            self._conn.commit()
        self._cache.clear()
        return [it["memory_id"] for it in items]

    # -- Search --------------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 5,
               agent_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        self._ensure_init()
        q = _normalize(query)
        cache_key = f"{q.tobytes().hex()}|{k}|{agent_filter or ''}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if np.linalg.norm(q) == 0:
            return []
        fetch_k = min(k * 10, 10000) if agent_filter else k

        with self._lock:
            rows = self._conn.execute(
                "SELECT m.memory_id, m.metadata, m.agent, m.tier, v.distance "
                "FROM memory_vec v JOIN memories m ON m.id = v.rowid "
                "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
                (q.tobytes(), fetch_k)).fetchall()

        results = []
        for row in rows:
            if agent_filter and row["agent"] != agent_filter:
                continue
            cos_sim = 1.0 - (row["distance"] ** 2) / 2.0
            results.append({
                "memory_id": row["memory_id"],
                "similarity": float(cos_sim),
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "agent": row["agent"],
                "tier": row["tier"],
            })
            if len(results) >= k:
                break

        self._cache_put(cache_key, results)
        return results

    # -- Delete --------------------------------------------------------------

    def delete(self, memory_id: str) -> bool:
        self._ensure_init()
        with self._lock:
            row = self._conn.execute("SELECT id FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            if row is None:
                return False
            self._conn.execute("DELETE FROM memory_vec WHERE rowid=?", (row["id"],))
            self._conn.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
            self._conn.commit()
            self._cache.clear()
            return True

    # -- Read ----------------------------------------------------------------

    def get_all(self, agent_filter: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        self._ensure_init()
        sql, params = "SELECT memory_id, metadata, agent, tier, timestamp FROM memories", []
        if agent_filter:
            sql += " WHERE agent=?"
            params.append(agent_filter)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [{"memory_id": r["memory_id"],
                 "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                 "agent": r["agent"], "tier": r["tier"], "timestamp": r["timestamp"]} for r in rows]

    # -- Stats ---------------------------------------------------------------

    def stats(self) -> Dict:
        self._ensure_init()
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_agent = {r[0]: r[1] for r in self._conn.execute("SELECT agent, COUNT(*) FROM memories GROUP BY agent").fetchall()}
            by_tier = {r[0]: r[1] for r in self._conn.execute("SELECT tier, COUNT(*) FROM memories GROUP BY tier").fetchall()}
        return {"total": total, "by_agent": by_agent, "by_tier": by_tier,
                "dim": self.dim, "cache_hits": self._cache_hits, "cache_misses": self._cache_misses}

    # -- Cleanup -------------------------------------------------------------

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

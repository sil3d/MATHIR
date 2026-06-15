"""
MATHIR sqlite-vec backed vector memory
========================================

Stores float32 embeddings in SQLite and uses the ``vec0`` virtual table
from ``sqlite-vec`` for fast approximate nearest-neighbour search.

Vectors are L2-normalized at insert time so that vec0's Euclidean
distance metric yields cosine similarity via:  cos = 1 - d²/2

Interface mirrors the drop-in ``SQLiteStore.search_by_embedding`` so it
can be swapped in as a drop-in backend.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import sqlite_vec
except ImportError:
    raise ImportError("pip install sqlite-vec")


class VecMemory:
    """
    Vector memory backed by sqlite-vec for fast cosine search.

    Parameters
    ----------
    db_path : str
        SQLite database file path. Created if it doesn't exist.
    dim : int
        Dimensionality of stored embeddings.
    """

    def __init__(self, db_path: str, dim: int):
        self.db_path = db_path
        self.dim = dim
        self._conn = sqlite3.connect(db_path)
        # ── Performance: WAL mode + relaxed sync (3-5x write speedup) ──
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-8000")       # 8 MB page cache
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA mmap_size=268435456")    # 256 MB mmap
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create the memories table + vec0 virtual table."""
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT UNIQUE,
                metadata TEXT,
                agent TEXT,
                tier TEXT,
                timestamp REAL
            )
        """)
        cur.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec
            USING vec0(embedding float[{self.dim}])
        """)
        self._conn.commit()

    def store(
        self,
        memory_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store a single embedding with metadata."""
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        assert arr.shape[0] == self.dim, (
            f"Expected dim {self.dim}, got {arr.shape[0]}"
        )

        # L2-normalize so vec0 L2 distance → cosine similarity
        norm = np.linalg.norm(arr)
        if norm > 0:
            unit = (arr / norm).astype(np.float32)
        else:
            unit = arr.copy()
        blob = unit.tobytes()

        meta = metadata or {}
        agent = meta.get("agent", "")
        tier = meta.get("tier", "episodic")
        ts = time.time()
        meta_json = json.dumps(meta, ensure_ascii=False, default=str)

        cur = self._conn.cursor()
        # INSERT into metadata table, capture rowid
        cur.execute(
            "INSERT OR REPLACE INTO memories "
            "(memory_id, metadata, agent, tier, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (memory_id, meta_json, agent, tier, ts),
        )
        rowid = cur.lastrowid
        # INSERT into vec0 using rowid as FK
        cur.execute(
            "INSERT OR REPLACE INTO memory_vec (rowid, embedding) VALUES (?, ?)",
            (rowid, blob),
        )
        self._conn.commit()

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        agent_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Top-k nearest neighbours by cosine distance via sqlite-vec.

        Parameters
        ----------
        query : np.ndarray
            Query vector (float32, shape [dim]).
        k : int
            Number of results to return.
        agent_filter : str, optional
            If set, only return memories with matching agent.

        Returns
        -------
        List of dicts with keys: memory_id, similarity, metadata, agent, tier.
        """
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        assert q.shape[0] == self.dim

        # L2-normalize query
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q_unit = (q / q_norm).astype(np.float32)

        # Fetch extra rows if we need to post-filter by agent
        fetch_k = min(k * 10, 10000) if agent_filter else k

        cur = self._conn.cursor()
        rows = cur.execute("""
            SELECT m.memory_id, m.metadata, m.agent, m.tier, v.distance
            FROM memory_vec v
            JOIN memories m ON m.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
        """, (q_unit.tobytes(), fetch_k)).fetchall()

        results = []
        for row in rows:
            mid = row["memory_id"]
            dist = row["distance"]
            cos_sim = 1.0 - (dist * dist) / 2.0

            if agent_filter and row["agent"] != agent_filter:
                continue

            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            results.append({
                "memory_id": mid,
                "similarity": float(cos_sim),
                "metadata": meta,
                "agent": row["agent"],
                "tier": row["tier"],
            })

            if len(results) >= k:
                break

        return results

    def close(self):
        """Close the database connection."""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

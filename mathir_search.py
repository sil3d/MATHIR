"""MATHIR Hybrid Search — Vector + BM25 + RRF Fusion."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

NUMPY_THRESHOLD = 5000
MMAP_DIR = "mathir_indexes"


def _normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D float32 vector."""
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenization for BM25."""
    return re.findall(r'\w+', text.lower())


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
# BM25 Backend — Lexical search for hybrid mode
# ---------------------------------------------------------------------------

class _BM25Backend:
    """BM25 lexical search over memory content. Rebuilt on each query (cheap for N<100k)."""

    def __init__(self):
        self._corpus: List[str] = []
        self._ids: List[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self._lock = threading.RLock()

    def build(self, items: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._corpus = [it.get("text", "") for it in items]
            self._ids = [it["memory_id"] for it in items]
            tokenized = [_tokenize(t) for t in self._corpus]
            self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def add(self, memory_id: str, text: str) -> None:
        with self._lock:
            self._corpus.append(text)
            self._ids.append(memory_id)
            tokenized = [_tokenize(t) for t in self._corpus]
            self._bm25 = BM25Okapi(tokenized)

    def remove(self, memory_id: str) -> bool:
        with self._lock:
            if memory_id not in self._ids:
                return False
            idx = self._ids.index(memory_id)
            self._ids.pop(idx)
            self._corpus.pop(idx)
            tokenized = [_tokenize(t) for t in self._corpus]
            self._bm25 = BM25Okapi(tokenized) if tokenized else None
            return True

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        with self._lock:
            if not self._bm25 or not self._ids:
                return []
            tokens = _tokenize(query)
            scores = self._bm25.get_scores(tokens)
            if len(scores) == 0:
                return []
            k = min(k, len(scores))
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            return [(self._ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def count(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# RRF Fusion — Combine vector + BM25 rankings
# ---------------------------------------------------------------------------

def rrf_fusion(vector_results: List[Tuple[str, float]],
               bm25_results: List[Tuple[str, float]],
               k: int = 60,
               vector_weight: float = 1.0,
               bm25_weight: float = 1.0) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion: combines two ranked lists into one.

    RRF score = sum(weight / (k + rank)) for each list.
    k=60 is the standard constant from the original RRF paper.
    """
    scores: Dict[str, float] = {}

    for rank, (mid, _) in enumerate(vector_results):
        scores[mid] = scores.get(mid, 0) + vector_weight / (k + rank + 1)

    for rank, (mid, _) in enumerate(bm25_results):
        scores[mid] = scores.get(mid, 0) + bm25_weight / (k + rank + 1)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(mid, scores[mid]) for mid in sorted_ids]


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
    """Hybrid Search: Vector (cosine) + BM25 (lexical) + RRF Fusion.

    Parameters: dim, db_path, strategy ("auto"|"numpy"|"usearch"|None), index_dir.
    """

    _SQL = {
        "create": ("CREATE TABLE IF NOT EXISTS memories ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT UNIQUE, "
                    "metadata TEXT, agent TEXT, tier TEXT, timestamp REAL, "
                    "embedding BLOB, text TEXT)"),
        "insert": ("INSERT OR REPLACE INTO memories "
                   "(memory_id, metadata, agent, tier, timestamp, embedding, text) VALUES (?,?,?,?,?,?,?)"),
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
        self._bm25: Optional[_BM25Backend] = _BM25Backend()
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

    def _meta_store(self, memory_id: str, embedding: np.ndarray, metadata: Dict[str, Any], text: str = ""):
        meta = metadata or {}
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
        with self._lock:
            self._conn.execute(self._SQL["insert"],
                (memory_id, json.dumps(meta, ensure_ascii=False, default=str),
                 meta.get("agent", ""), meta.get("tier", "episodic"), time.time(), blob, text))
            self._conn.commit()

    def _meta_store_batch(self, items: List[Dict[str, Any]]):
        with self._lock:
            for it in items:
                meta = it.get("metadata", {})
                blob = np.asarray(it["embedding"], dtype=np.float32).tobytes()
                self._conn.execute(self._SQL["insert"],
                    (it["memory_id"], json.dumps(meta, ensure_ascii=False, default=str),
                     meta.get("agent", ""), meta.get("tier", "episodic"), time.time(), blob,
                     it.get("text", "")))
            self._conn.commit()

    def _meta_get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(self._SQL["get"], (memory_id,)).fetchone()
        if not row:
            return None
        return {"memory_id": row["memory_id"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "agent": row["agent"], "tier": row["tier"],
                "timestamp": row["timestamp"],
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None,
                "text": row["text"] if "text" in row.keys() else ""}

    def _meta_get_all(self, limit: int = 100_000) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            self._SQL["get_all"] + " ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [{"memory_id": r["memory_id"],
                 "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                 "agent": r["agent"], "tier": r["tier"], "timestamp": r["timestamp"],
                 "embedding": np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None,
                 "text": r["text"] if "text" in r.keys() else ""}
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
            all_items = self._meta_get_all()
            self._current.build(all_items)
            # Build BM25 index from text content
            if self._bm25:
                self._bm25.build(all_items)

    def _maybe_switch(self):
        if self.strategy != "auto" or self._current_name != "numpy":
            return
        if self._current.count() >= NUMPY_THRESHOLD:
            if self._usearch is None:
                self._usearch = _USearchBackend(self.dim, os.path.join(self.index_dir, f"mathir_{self.dim}d.usearch"))
            self._current = self._usearch; self._current_name = "usearch"
            all_items = self._meta_get_all()
            self._current.build(all_items); self._usearch.save()
            if self._bm25:
                self._bm25.build(all_items)

    # -- Public API ----------------------------------------------------------

    def store(self, memory_id: str, embedding: np.ndarray,
              metadata: Optional[Dict[str, Any]] = None, text: str = "") -> None:
        self._meta_store(memory_id, embedding, metadata or {}, text)
        self._current.add(memory_id, embedding)
        if self._bm25:
            self._bm25.add(memory_id, text)
        self._maybe_switch()

    def store_batch(self, items: List[Dict[str, Any]]) -> int:
        if not items:
            return 0
        self._meta_store_batch(items)
        for it in items:
            self._current.add(it["memory_id"], it["embedding"])
            if self._bm25:
                self._bm25.add(it["memory_id"], it.get("text", ""))
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

    def hybrid_search(self, query_text: str, query_embedding: np.ndarray,
                      k: int = 5, vector_weight: float = 1.0,
                      bm25_weight: float = 1.0,
                      agent_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Hybrid search: Vector cosine + BM25 lexical + RRF fusion.

        Combines semantic understanding (vector) with exact keyword matching (BM25)
        using Reciprocal Rank Fusion. Better than either alone.
        """
        fetch_k = min(k * 3, self._current.count())

        # Vector results
        vector_pairs = self._current.search(query_embedding, k=fetch_k)
        vector_results = []
        for mid, score in vector_pairs:
            meta = self._meta_get(mid)
            if meta and (not agent_filter or meta.get("agent", "") == agent_filter):
                vector_results.append((mid, score))

        # BM25 results
        bm25_results = []
        if self._bm25:
            bm25_pairs = self._bm25.search(query_text, k=fetch_k)
            for mid, score in bm25_pairs:
                if not agent_filter or (self._meta_get(mid) or {}).get("agent", "") == agent_filter:
                    bm25_results.append((mid, score))

        # RRF fusion
        fused = rrf_fusion(vector_results, bm25_results,
                           vector_weight=vector_weight, bm25_weight=bm25_weight)

        # Build final results with metadata
        results = []
        for mid, rrf_score in fused[:k]:
            meta = self._meta_get(mid)
            if meta:
                results.append({
                    "memory_id": mid,
                    "rrf_score": rrf_score,
                    "metadata": meta.get("metadata", {}),
                    "agent": meta.get("agent", ""),
                    "tier": meta.get("tier", "episodic"),
                    "text": meta.get("text", ""),
                })
        return results

    def delete(self, memory_id: str) -> bool:
        self._current.remove(memory_id)
        if self._bm25:
            self._bm25.remove(memory_id)
        return self._meta_delete(memory_id)

    def count(self) -> int:
        return self._current.count()

    def stats(self) -> Dict[str, Any]:
        return {"backend": self._current_name, "dim": self.dim, "count": self.count(),
                "bm25_count": self._bm25.count() if self._bm25 else 0,
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

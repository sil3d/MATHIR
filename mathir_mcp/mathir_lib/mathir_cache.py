"""MATHIR Auto-Cache — 3-layer caching for embeddings, recalls, and sessions.

L1 Embedding Cache:  LRU on encode() calls. Deterministic, never expires.
L2 Recall Cache:     TTL-based on recall results. Invalidated on writes.
L3 Session Cache:    Pre-warmed top-N memories per project. TTL + write invalidation.

Thread-safe. All counters are atomic via threading.Lock.
"""

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


class EmbeddingCache:
    """L1 — LRU cache for embedder.encode() results.

    Embeddings are deterministic (same input → same output), so entries
    never expire. Only evicted when the cache is full (LRU order).
    """

    def __init__(self, maxsize: int = 1024):
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def get(self, text: str) -> Optional[Any]:
        k = self._key(text)
        with self._lock:
            if k in self._cache:
                self._cache.move_to_end(k)
                self._hits += 1
                return self._cache[k]
            self._misses += 1
            return None

    def put(self, text: str, embedding: Any) -> None:
        k = self._key(text)
        with self._lock:
            if k in self._cache:
                self._cache.move_to_end(k)
                self._cache[k] = embedding
                return
            self._cache[k] = embedding
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": round(self._hits / total, 4) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class RecallCache:
    """L2 — TTL-based cache for recall/search results.

    Keyed by (query, k, project, agent_filter, block_type_filter).
    Invalidated globally on any write operation (save/delete/promote/consolidate).
    """

    def __init__(self, maxsize: int = 256, ttl_seconds: float = 60.0):
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._invalidations = 0

    def _key(self, query: str, k: int, project: str = None,
             agent: str = None, block_type: str = None) -> str:
        raw = f"{query}|{k}|{project}|{agent}|{block_type}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def get(self, query: str, k: int, project: str = None,
            agent: str = None, block_type: str = None) -> Optional[Any]:
        key = self._key(query, k, project, agent, block_type)
        now = time.monotonic()
        with self._lock:
            if key in self._cache:
                ts, val = self._cache[key]
                if now - ts < self._ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return val
                del self._cache[key]
            self._misses += 1
            return None

    def put(self, query: str, k: int, result: Any,
            project: str = None, agent: str = None, block_type: str = None) -> None:
        key = self._key(query, k, project, agent, block_type)
        now = time.monotonic()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (now, result)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
            self._invalidations += 1

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": round(self._hits / total, 4) if total else 0.0,
                "invalidations": self._invalidations,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class SessionCache:
    """L3 — Pre-warmed top-N memories per project for session_start/context.

    Stores the most relevant memories for each project so session_start
    and memory_context can return instantly on repeat calls within the TTL.
    Invalidated per-project on save, or globally on delete/consolidate.
    """

    def __init__(self, top_n: int = 20, ttl_seconds: float = 300.0):
        self._top_n = top_n
        self._ttl = ttl_seconds
        self._cache: Dict[str, Tuple[float, List[Any]]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, project: str, task: str = "") -> Optional[List[Any]]:
        key = project or "__default__"
        now = time.monotonic()
        with self._lock:
            if key in self._cache:
                ts, val = self._cache[key]
                if now - ts < self._ttl:
                    self._hits += 1
                    return val
                del self._cache[key]
            self._misses += 1
            return None

    def put(self, project: str, memories: List[Any]) -> None:
        key = project or "__default__"
        now = time.monotonic()
        with self._lock:
            self._cache[key] = (now, memories[:self._top_n])

    def invalidate_project(self, project: str = None) -> None:
        with self._lock:
            key = project or "__default__"
            self._cache.pop(key, None)

    def invalidate_all(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "projects_cached": len(self._cache),
                "top_n": self._top_n,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": round(self._hits / total, 4) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ---------------------------------------------------------------------------
# Singleton instances (imported by mathir_server.py)
# ---------------------------------------------------------------------------
embedding_cache = EmbeddingCache(maxsize=1024)
recall_cache = RecallCache(maxsize=256, ttl_seconds=60.0)
session_cache = SessionCache(top_n=20, ttl_seconds=300.0)


def cache_stats() -> Dict[str, Any]:
    """Aggregate stats across all 3 cache layers."""
    return {
        "L1_embedding": embedding_cache.stats(),
        "L2_recall": recall_cache.stats(),
        "L3_session": session_cache.stats(),
    }


def invalidate_on_write(project: str = None) -> None:
    """Call after any write operation (save/delete/promote/consolidate)."""
    recall_cache.invalidate()
    if project:
        session_cache.invalidate_project(project)
    else:
        session_cache.invalidate_all()

"""
mathir_cache.py — Lightweight LRU cache for the mathir_adapter HTTP calls.

Confrank's 31 roundtrips-per-query bottleneck (1058ms) is dominated by
repeated HTTP calls to /hybrid_search(term) and /get_links(memory_id),
where term probes and per-candidate neighbor lookups are highly
redundant across queries within a benchmark run. This module wraps
hybrid_search and get_links with thread-safe LRU caches keyed on the
request signature (project, query_text, k) and (project, memory_id,
depth, decay) respectively.

Cache hit accounting: each MathirAdapter instance exposes `.cache_stats`
which the runner logs into the JSONL output for transparency.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, Any


class TTLCache:
    """Thread-safe LRU+TTL cache.

    - maxsize: number of entries (LRU eviction beyond that)
    - ttl_seconds: 0 means "no TTL"; >0 means evict entries older than this
    - Hits are counted; misses are counted; evictions are counted (per LRU
      and per TTL separately). This lets the runner verify the cache
      actually served the workload.
    """

    def __init__(self, maxsize: int = 1024, ttl_seconds: float = 0.0):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._data: "OrderedDict[Any, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0,
                      "lru_evictions": 0, "ttl_evictions": 0,
                      "writes": 0}

    def get(self, key):
        with self._lock:
            now = time.monotonic()
            entry = self._data.get(key)
            if entry is None:
                self.stats["misses"] += 1
                return None
            ts, value = entry
            if self.ttl_seconds > 0 and (now - ts) > self.ttl_seconds:
                del self._data[key]
                self.stats["ttl_evictions"] += 1
                self.stats["misses"] += 1
                return None
            # LRU touch
            self._data.move_to_end(key)
            self.stats["hits"] += 1
            return value

    def put(self, key, value):
        with self._lock:
            now = time.monotonic()
            # refresh existing
            if key in self._data:
                self._data[key] = (now, value)
                self._data.move_to_end(key)
                return
            # evict if at capacity
            while len(self._data) >= self.maxsize:
                self._data.popitem(last=False)
                self.stats["lru_evictions"] += 1
            self._data[key] = (now, value)
            self.stats["writes"] += 1

    def clear(self):
        with self._lock:
            self._data.clear()

    def summary(self) -> str:
        s = self.stats
        n = s["hits"] + s["misses"]
        rate = (s["hits"] / n * 100.0) if n else 0.0
        return (f"hits={s['hits']} misses={s['misses']} rate={rate:.1f}% "
                f"writes={s['writes']} lru_ev={s['lru_evictions']} "
                f"ttl_ev={s['ttl_evictions']}")


# Key helpers — kept here so cache logic is testable in isolation.
def hybrid_search_key(project: str, query: str, k: int,
                      vector_weight: float, bm25_weight: float,
                      agent: str | None, entity_weight: float = 0.0) -> tuple:
    return ("hybrid_search", project, query, k, vector_weight,
            bm25_weight, agent or "", entity_weight)


def get_links_key(project: str, memory_id: str, depth: int,
                  decay: float) -> tuple:
    return ("get_links", project, memory_id, depth, decay)


def recall_key(project: str, query: str, k: int, agent: str | None,
               include_embeddings: bool) -> tuple:
    return ("recall", project, query, k, agent or "", include_embeddings)
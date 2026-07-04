"""Tests for MATHIR 3-layer auto-cache system (mathir_cache.py)."""

import time
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mathir_lib"))
from mathir_cache import (
    EmbeddingCache,
    RecallCache,
    SessionCache,
    cache_stats,
    invalidate_on_write,
    embedding_cache,
    recall_cache,
    session_cache,
)


# ---------------------------------------------------------------------------
# L1 — Embedding Cache
# ---------------------------------------------------------------------------

class TestEmbeddingCache:

    def test_put_get_hit(self):
        c = EmbeddingCache(maxsize=8)
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        c.put("hello world", vec)
        got = c.get("hello world")
        assert got is not None
        np.testing.assert_array_equal(got, vec)

    def test_miss_returns_none(self):
        c = EmbeddingCache(maxsize=8)
        assert c.get("nonexistent") is None

    def test_lru_eviction(self):
        c = EmbeddingCache(maxsize=2)
        c.put("a", np.array([1.0]))
        c.put("b", np.array([2.0]))
        c.put("c", np.array([3.0]))  # evicts "a"
        assert c.get("a") is None
        assert c.get("b") is not None
        assert c.get("c") is not None

    def test_lru_access_refreshes(self):
        c = EmbeddingCache(maxsize=2)
        c.put("a", np.array([1.0]))
        c.put("b", np.array([2.0]))
        c.get("a")  # refresh "a"
        c.put("c", np.array([3.0]))  # evicts "b" (least recent)
        assert c.get("a") is not None
        assert c.get("b") is None

    def test_stats_counters(self):
        c = EmbeddingCache(maxsize=8)
        c.put("x", np.array([1.0]))
        c.get("x")  # hit
        c.get("y")  # miss
        s = c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["size"] == 1
        assert s["hit_ratio"] == 0.5

    def test_clear(self):
        c = EmbeddingCache(maxsize=8)
        c.put("x", np.array([1.0]))
        c.clear()
        assert c.get("x") is None
        assert c.stats()["size"] == 0

    def test_deterministic_key(self):
        c = EmbeddingCache(maxsize=8)
        vec = np.array([1.0, 2.0])
        c.put("test text", vec)
        got = c.get("test text")
        np.testing.assert_array_equal(got, vec)


# ---------------------------------------------------------------------------
# L2 — Recall Cache
# ---------------------------------------------------------------------------

class TestRecallCache:

    def test_put_get_hit(self):
        c = RecallCache(maxsize=8, ttl_seconds=60)
        result = {"results": [{"id": "1"}], "total": 1}
        c.put("query", 5, result, project="p1")
        got = c.get("query", 5, project="p1")
        assert got == result

    def test_miss_returns_none(self):
        c = RecallCache(maxsize=8, ttl_seconds=60)
        assert c.get("nope", 5) is None

    def test_different_params_different_keys(self):
        c = RecallCache(maxsize=8, ttl_seconds=60)
        r1 = {"results": [], "total": 0}
        r2 = {"results": [1], "total": 1}
        c.put("query", 5, r1, project="p1")
        c.put("query", 10, r2, project="p1")
        assert c.get("query", 5, project="p1") == r1
        assert c.get("query", 10, project="p1") == r2

    def test_ttl_expiry(self):
        c = RecallCache(maxsize=8, ttl_seconds=0.05)
        c.put("query", 5, {"data": 1})
        assert c.get("query", 5) is not None
        time.sleep(0.06)
        assert c.get("query", 5) is None

    def test_invalidate_clears_all(self):
        c = RecallCache(maxsize=8, ttl_seconds=60)
        c.put("q1", 5, {"data": 1})
        c.put("q2", 5, {"data": 2})
        c.invalidate()
        assert c.get("q1", 5) is None
        assert c.get("q2", 5) is None
        assert c.stats()["invalidations"] == 1

    def test_lru_eviction(self):
        c = RecallCache(maxsize=2, ttl_seconds=60)
        c.put("a", 5, {"a": 1})
        c.put("b", 5, {"b": 1})
        c.put("c", 5, {"c": 1})  # evicts "a"
        assert c.get("a", 5) is None
        assert c.get("b", 5) is not None

    def test_stats(self):
        c = RecallCache(maxsize=8, ttl_seconds=60)
        c.put("q", 5, {"x": 1})
        c.get("q", 5)  # hit
        c.get("miss", 5)  # miss
        s = c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_ratio"] == 0.5


# ---------------------------------------------------------------------------
# L3 — Session Cache
# ---------------------------------------------------------------------------

class TestSessionCache:

    def test_put_get(self):
        c = SessionCache(top_n=5, ttl_seconds=60)
        memories = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        c.put("project_x", memories)
        got = c.get("project_x")
        assert got == memories

    def test_top_n_truncation(self):
        c = SessionCache(top_n=2, ttl_seconds=60)
        memories = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        c.put("p", memories)
        got = c.get("p")
        assert len(got) == 2

    def test_ttl_expiry(self):
        c = SessionCache(top_n=5, ttl_seconds=0.05)
        c.put("p", [{"id": "1"}])
        assert c.get("p") is not None
        time.sleep(0.06)
        assert c.get("p") is None

    def test_invalidate_project(self):
        c = SessionCache(top_n=5, ttl_seconds=60)
        c.put("p1", [{"id": "1"}])
        c.put("p2", [{"id": "2"}])
        c.invalidate_project("p1")
        assert c.get("p1") is None
        assert c.get("p2") is not None

    def test_invalidate_all(self):
        c = SessionCache(top_n=5, ttl_seconds=60)
        c.put("p1", [{"id": "1"}])
        c.put("p2", [{"id": "2"}])
        c.invalidate_all()
        assert c.get("p1") is None
        assert c.get("p2") is None

    def test_stats(self):
        c = SessionCache(top_n=5, ttl_seconds=60)
        c.put("p", [{"id": "1"}])
        c.get("p")  # hit
        c.get("missing")  # miss
        s = c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["projects_cached"] == 1


# ---------------------------------------------------------------------------
# Integration — invalidate_on_write
# ---------------------------------------------------------------------------

class TestInvalidateOnWrite:

    def setup_method(self):
        recall_cache.clear()
        session_cache.clear()

    def test_invalidate_clears_recall_and_session(self):
        recall_cache.put("q", 5, {"data": 1})
        session_cache.put("proj", [{"id": "1"}])
        invalidate_on_write(project="proj")
        assert recall_cache.get("q", 5) is None
        assert session_cache.get("proj") is None

    def test_invalidate_no_project_clears_all_sessions(self):
        session_cache.put("p1", [{"id": "1"}])
        session_cache.put("p2", [{"id": "2"}])
        invalidate_on_write()
        assert session_cache.get("p1") is None
        assert session_cache.get("p2") is None

    def test_invalidate_specific_project_keeps_others(self):
        session_cache.put("p1", [{"id": "1"}])
        session_cache.put("p2", [{"id": "2"}])
        invalidate_on_write(project="p1")
        assert session_cache.get("p1") is None
        assert session_cache.get("p2") is not None


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def test_cache_stats_has_all_layers():
    s = cache_stats()
    assert "L1_embedding" in s
    assert "L2_recall" in s
    assert "L3_session" in s
    assert "hits" in s["L1_embedding"]
    assert "hit_ratio" in s["L2_recall"]
    assert "projects_cached" in s["L3_session"]

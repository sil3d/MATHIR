"""Tests for the BM25 index cache in /api/memory/hybrid_search.

Before this fix, every single hybrid_search call rebuilt a fresh BM25Okapi
index from every row in `memories` -- O(corpus size) tokenization per query.
Measured on real benchmark data: 566s for scifact's query set (5183 docs)
vs 17s for an equivalent cached reference implementation. These tests verify
the BM25 index is now reused across calls when the corpus hasn't changed,
and rebuilt when it has (new memories saved) -- using a real Flask test
client and real VecMemory/SQLite backing store (no mocking of the search
logic itself), following the existing test_auth.py pattern for exercising
mathir_server.py's routes without starting a real network server.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_MCP_ROOT = _HERE.parent
if str(_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT))

try:
    from mathir_lib import mathir_server
    from mathir_lib.mathir_vec import VecMemory
except ImportError:
    import mathir_server  # type: ignore[no-redef]
    from mathir_vec import VecMemory  # type: ignore[no-redef]


class _FakeEmbedder:
    """Deterministic embedder: same text -> same vector, so hybrid_search's
    vector component behaves predictably without loading a real model."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text: str):
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        return rng.randn(self.dim).astype(np.float32)


def _make_client(tmp_path, monkeypatch, db_name="hybrid_cache_test.db"):
    db = tmp_path / db_name
    vec_mem = VecMemory(db, embedding_dim=384)
    embedder = _FakeEmbedder(dim=384)
    monkeypatch.setattr(
        mathir_server, "_resolve_db",
        lambda project=None, cwd=None: (vec_mem, db, embedder),
    )
    monkeypatch.setattr(mathir_server, "_risk_enabled", False)
    monkeypatch.setattr(mathir_server, "_bm25_cache", {})
    client = mathir_server.app.test_client()
    return client, vec_mem, db


def test_bm25_index_is_reused_across_repeated_queries(tmp_path, monkeypatch):
    """Two hybrid_search calls against an unchanged corpus must only build
    BM25Okapi once."""
    client, vec_mem, db = _make_client(tmp_path, monkeypatch)

    for i in range(20):
        client.post("/api/memory/save", json={
            "content": f"fluid mechanics turbulence boundary layer document {i}",
            "agent": "test", "block_type": "episodic", "label": "", "priority": 5,
        })

    import rank_bm25
    build_count = {"n": 0}
    original_init = rank_bm25.BM25Okapi.__init__

    def counting_init(self, *args, **kwargs):
        build_count["n"] += 1
        return original_init(self, *args, **kwargs)

    with patch.object(rank_bm25.BM25Okapi, "__init__", counting_init):
        resp1 = client.post("/api/memory/hybrid_search", json={"query": "turbulence", "k": 5})
        resp2 = client.post("/api/memory/hybrid_search", json={"query": "boundary layer", "k": 5})
        resp3 = client.post("/api/memory/hybrid_search", json={"query": "document 5", "k": 5})

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp3.status_code == 200
    assert build_count["n"] == 1, (
        f"expected BM25Okapi to be built exactly once across 3 calls with an "
        f"unchanged corpus, got {build_count['n']} builds"
    )

    # Results still make sense: querying for "turbulence" should surface the
    # memories containing that term ranked above the ones that don't.
    body1 = resp1.get_json()
    assert body1["total"] > 0
    assert any("turbulence" in r["content"] for r in body1["results"])


def test_bm25_index_rebuilds_after_new_memory_saved(tmp_path, monkeypatch):
    """Adding a new memory must invalidate the cache and trigger exactly one
    rebuild on the next call, not zero (stale) and not one-per-call."""
    client, vec_mem, db = _make_client(tmp_path, monkeypatch)

    for i in range(10):
        client.post("/api/memory/save", json={
            "content": f"cavitation pump inlet pressure document {i}",
            "agent": "test", "block_type": "episodic", "label": "", "priority": 5,
        })

    import rank_bm25
    build_count = {"n": 0}
    original_init = rank_bm25.BM25Okapi.__init__

    def counting_init(self, *args, **kwargs):
        build_count["n"] += 1
        return original_init(self, *args, **kwargs)

    with patch.object(rank_bm25.BM25Okapi, "__init__", counting_init):
        # First call: cold cache -> 1 build.
        client.post("/api/memory/hybrid_search", json={"query": "cavitation", "k": 5})
        assert build_count["n"] == 1

        # Second call, same corpus: cache hit -> still 1 build.
        client.post("/api/memory/hybrid_search", json={"query": "pump", "k": 5})
        assert build_count["n"] == 1

        # New memory saved -> corpus row count changes.
        client.post("/api/memory/save", json={
            "content": "cavitation erosion damage on impeller blades new entry",
            "agent": "test", "block_type": "episodic", "label": "", "priority": 5,
        })

        # Next search must detect the row-count change and rebuild once.
        resp = client.post("/api/memory/hybrid_search", json={"query": "impeller", "k": 5})
        assert build_count["n"] == 2, (
            "expected exactly one rebuild after a new memory was saved, "
            f"got {build_count['n']} total builds"
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert any("impeller" in r["content"] for r in body["results"])

        # And a third call with the corpus unchanged again must NOT rebuild.
        client.post("/api/memory/hybrid_search", json={"query": "erosion", "k": 5})
        assert build_count["n"] == 2

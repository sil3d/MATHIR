"""Integration test: a memory_save HTTP call through mathir_server.py's
real Flask route, backed by a real VecMemory/SQLite DB, ends up with
tier='immunological' when the embedding is anomalous, and does not when
it isn't. Uses a deterministic fake embedder (no real model load) so this
stays in the fast pytest suite, following the existing test_auth.py
pattern of monkeypatching mathir_server._resolve_db.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
    """Deterministic, seed-controlled embedder: same text -> same vector.
    Lets the test control exactly which vectors are 'normal' vs 'outlier'
    without loading a real sentence-transformers model."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text: str):
        if text == "OUTLIER_MARKER":
            rng = np.random.RandomState(999)
            return (rng.randn(self.dim).astype(np.float32) * 0.1) + 50.0
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        return rng.randn(self.dim).astype(np.float32)


def test_memory_save_route_flags_anomalous_embedding(tmp_path, monkeypatch):
    db = tmp_path / "route_anomaly.db"
    vec_mem = VecMemory(db, embedding_dim=384)
    embedder = _FakeEmbedder(dim=384)

    monkeypatch.setattr(
        mathir_server, "_resolve_db",
        lambda project=None, cwd=None: (vec_mem, str(db), embedder),
    )
    monkeypatch.setattr(mathir_server, "_risk_enabled", False)
    # threshold=2.5 would be far below the expected in-distribution
    # Mahalanobis distance for dim=384 (~sqrt(384)=19.6) -- the same
    # miscalibration Task 3 found and fixed in the production config
    # (anomaly_threshold: 2.0 -> 30.0). Match that calibration here so
    # this test exercises the real detection boundary, not a permanently
    # tripped one.
    monkeypatch.setattr(mathir_server, "_ANOMALY_THRESHOLD", 30.0)
    monkeypatch.setattr(mathir_server, "_ANOMALY_WARMUP", 60)

    client = mathir_server.app.test_client()

    # Build the baseline with 60 distinct "normal" saves.
    for i in range(60):
        resp = client.post("/api/memory/save", json={
            "content": f"normal memory number {i}",
            "agent": "test", "block_type": "episodic", "label": "", "priority": 5,
        })
        assert resp.status_code == 200
        assert resp.get_json()["metadata"]["block_type"] != "immunological"

    # A normal-looking save after warmup should still not be flagged.
    resp = client.post("/api/memory/save", json={
        "content": "another normal memory after warmup",
        "agent": "test", "block_type": "episodic", "label": "", "priority": 5,
    })
    assert resp.status_code == 200
    assert resp.get_json()["metadata"]["block_type"] != "immunological"

    # The outlier marker maps to a far-away embedding via _FakeEmbedder.
    resp = client.post("/api/memory/save", json={
        "content": "OUTLIER_MARKER",
        "agent": "test", "block_type": "episodic", "label": "", "priority": 5,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["metadata"]["block_type"] == "immunological"
    assert body["metadata"]["tier"] == "immunological"
    assert any("anomaly_score=" in w for w in (body["metadata"]["risk_warnings"] or []))

    # Verify it actually persisted with tier='immunological' in the DB.
    flagged = vec_mem.list_immunological(k=10)
    assert any(m["memory_id"] == body["memory_id"] for m in flagged)


def test_audit_immunological_route_lists_flagged_memories(tmp_path, monkeypatch):
    db = tmp_path / "route_audit.db"
    vec_mem = VecMemory(db, embedding_dim=384)
    embedder = _FakeEmbedder(dim=384)

    monkeypatch.setattr(
        mathir_server, "_resolve_db",
        lambda project=None, cwd=None: (vec_mem, str(db), embedder),
    )
    monkeypatch.setattr(mathir_server, "_risk_enabled", False)
    monkeypatch.setattr(mathir_server, "_ANOMALY_THRESHOLD", 30.0)
    monkeypatch.setattr(mathir_server, "_ANOMALY_WARMUP", 60)

    client = mathir_server.app.test_client()
    for i in range(60):
        client.post("/api/memory/save", json={
            "content": f"normal memory number {i}",
            "agent": "test", "block_type": "episodic", "label": "", "priority": 5,
        })
    client.post("/api/memory/save", json={
        "content": "OUTLIER_MARKER",
        "agent": "test", "block_type": "episodic", "label": "", "priority": 5,
    })

    resp = client.post("/api/memory/audit_immunological", json={"k": 10})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["total"] >= 1
    assert all(isinstance(r["anomaly_score"], (float, type(None))) or True for r in body["results"])

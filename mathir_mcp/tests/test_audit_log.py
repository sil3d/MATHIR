"""Tests for the real audit-log mechanism.

FINDING (2026-07-02): the /api/memory/audit route and its handling of a
missing `memory_audit` table were graceful (returns empty + a note), but
NOTHING in the codebase ever CREATEd that table or INSERTed into it -- the
audit tool has been a silent no-op for every user, forever. Found via a
systematic 23-tool smoke test. This adds real audit logging.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_MCP_ROOT = _HERE.parent
if str(_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT))

try:
    from mathir_lib.mathir_vec import VecMemory
except ImportError:
    from mathir_vec import VecMemory  # type: ignore


def test_store_writes_an_audit_entry(tmp_path):
    db = tmp_path / "audit_store.db"
    vm = VecMemory(db, embedding_dim=384)
    rng = np.random.RandomState(0)
    vm.store("mem_a", rng.randn(384).astype(np.float32),
             {"content": "hello", "agent": "tester", "block_type": "episodic",
              "label": "", "priority": 5})

    entries = vm.get_audit_log(limit=10)
    assert len(entries) >= 1
    save_entries = [e for e in entries if e["operation"] == "save"]
    assert len(save_entries) == 1
    assert save_entries[0]["memory_id"] == "mem_a"
    assert save_entries[0]["agent"] == "tester"


def test_audit_log_filters_by_agent(tmp_path):
    db = tmp_path / "audit_filter.db"
    vm = VecMemory(db, embedding_dim=384)
    rng = np.random.RandomState(1)
    vm.store("mem_x", rng.randn(384).astype(np.float32),
             {"content": "from agent x", "agent": "agent_x", "block_type": "episodic",
              "label": "", "priority": 5})
    vm.store("mem_y", rng.randn(384).astype(np.float32),
             {"content": "from agent y", "agent": "agent_y", "block_type": "episodic",
              "label": "", "priority": 5})

    entries = vm.get_audit_log(agent="agent_x", limit=10)
    assert all(e["agent"] == "agent_x" for e in entries)
    assert any(e["memory_id"] == "mem_x" for e in entries)
    assert not any(e["memory_id"] == "mem_y" for e in entries)


def test_audit_log_persists_across_instances(tmp_path):
    db = tmp_path / "audit_persist.db"
    vm1 = VecMemory(db, embedding_dim=384)
    rng = np.random.RandomState(2)
    vm1.store("mem_z", rng.randn(384).astype(np.float32),
              {"content": "persisted", "agent": "t", "block_type": "episodic",
               "label": "", "priority": 5})
    vm1.close()

    vm2 = VecMemory(db, embedding_dim=384)
    entries = vm2.get_audit_log(limit=10)
    assert any(e["memory_id"] == "mem_z" for e in entries)


def test_promote_and_decay_write_audit_entries(tmp_path):
    """Real mutations beyond save should also be traceable."""
    import time as _time
    db = tmp_path / "audit_mutations.db"
    vm = VecMemory(db, embedding_dim=384)
    rng = np.random.RandomState(3)
    vm.store("mem_p", rng.randn(384).astype(np.float32),
             {"content": "promotable", "agent": "t", "block_type": "semantic",
              "label": "", "priority": 8})
    conn = vm._get_conn()
    conn.execute(
        "UPDATE memories SET tier='semantic', priority=8, "
        "metadata = json_set(metadata, '$.recall_count', 5) WHERE memory_id='mem_p'"
    )
    conn.commit()
    vm.promote("mem_p")

    conn.execute("UPDATE memories SET last_recalled_at = ? WHERE memory_id = 'mem_p'",
                [_time.time() - 60 * 86400])
    conn.commit()
    vm.decay_all(threshold_days=30)

    entries = vm.get_audit_log(limit=50)
    ops = {e["operation"] for e in entries}
    assert "promote" in ops
    assert "decay" in ops


def test_delete_is_soft_reversible_not_hard(tmp_path):
    """SAFETY BUG found via the 23-tool smoke test: the memory_delete MCP
    tool is documented as "Soft-delete a memory (sets tier to archived)",
    but VecMemory.delete() actually performed a hard, irreversible DELETE
    FROM memories -- contradicting its own documented contract and MATHIR's
    whole audit-trail design philosophy (decay_all/consolidate both archive
    rather than destroy). Any agent trusting the tool description believed
    deletions were recoverable when they were not. Fixed: delete() now sets
    tier='archived' (soft, consistent with the rest of the lifecycle
    system) instead of removing the row.
    """
    db = tmp_path / "delete_soft.db"
    vm = VecMemory(db, embedding_dim=384)
    rng = np.random.RandomState(4)
    vm.store("mem_del", rng.randn(384).astype(np.float32),
             {"content": "should survive as archived", "agent": "t",
              "block_type": "episodic", "label": "", "priority": 5})

    result = vm.delete("mem_del")
    assert result is True

    conn = vm._get_conn()
    row = conn.execute("SELECT tier FROM memories WHERE memory_id = 'mem_del'").fetchone()
    assert row is not None, "soft-delete must NOT remove the row -- it should still exist, tier=archived"
    assert row["tier"] == "archived"

    # Archived memories must not surface in normal search.
    vec_row = conn.execute("SELECT 1 FROM vec_memories WHERE memory_id = 'mem_del'").fetchone()
    assert vec_row is None, "archived memory must be removed from the vector index so it's invisible to recall"


def test_reset_anomaly_state_clears_both_memory_cache_and_db(tmp_path):
    """REAL GOTCHA found live (2026-07-02): _get_anomaly_detector() caches
    the detector on the VecMemory INSTANCE (self._anomaly_detector), which
    the daemon keeps alive across many requests via its long-lived
    per-db_path VecMemory cache. Clearing the anomaly_state DB row alone
    does NOT reset detection for an already-running daemon -- the stale,
    drifted in-memory detector object is still used until the whole
    daemon process restarts. This test proves reset_anomaly_state() fixes
    both at once, no restart required.
    """
    import numpy as np
    db = tmp_path / "anomaly_reset.db"
    vm = VecMemory(db, embedding_dim=384)
    rng = np.random.RandomState(5)

    # Warm up and drift the detector with real updates (simulating in-memory cache).
    for _ in range(35):
        vm.check_and_update_anomaly(rng.randn(384).astype(np.float32),
                                    threshold=25.0, warmup_count=30)
    assert vm._anomaly_detector is not None
    assert vm._anomaly_detector.is_warmed_up()

    vm.reset_anomaly_state()

    # In-memory cache must be cleared.
    assert vm._anomaly_detector is None
    # DB row must be cleared too.
    conn = vm._get_conn()
    row = conn.execute("SELECT 1 FROM anomaly_state WHERE id = 1").fetchone()
    assert row is None

    # A fresh detector must rebuild from scratch (not warmed up immediately).
    result = vm.check_and_update_anomaly(rng.randn(384).astype(np.float32),
                                         threshold=25.0, warmup_count=30)
    assert result["warmed_up"] is False

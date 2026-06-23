"""Smoke test for the new decay/boost methods on a real DB."""
import tempfile, time, sqlite3
from pathlib import Path
import numpy as np
from mathir_vec import VecMemory


def main():
    db_path = Path(tempfile.gettempdir()) / "mathir_decay_smoke.db"
    if db_path.exists():
        db_path.unlink()

    # Pre-create legacy-schema DB so VecMemory picks it up on open.
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE memories (
            memory_id      TEXT PRIMARY KEY,
            modality       TEXT NOT NULL DEFAULT 'text',
            embedding      BLOB,
            embedding_dim  INTEGER,
            metadata       TEXT,
            modality_text  TEXT,
            timestamp      REAL,
            tier           TEXT,
            stability      REAL DEFAULT 1.0,
            recall_count   INTEGER DEFAULT 0,
            provider       TEXT DEFAULT 'unknown',
            model          TEXT DEFAULT 'unknown',
            last_recalled_at REAL DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()

    vm = VecMemory(db_path, embedding_dim=384)
    assert vm._schema_kind() == "legacy", f"expected legacy, got {vm._schema_kind()}"

    cols = [r[1] for r in vm._get_conn().execute("PRAGMA table_info(memories)").fetchall()]
    assert "last_recalled_at" in cols
    assert "stability" in cols

    vec = np.zeros(384, dtype=np.float32)
    now = time.time()
    c = vm._get_conn()
    c.execute(
        """INSERT INTO memories
           (memory_id, modality, embedding, embedding_dim, metadata,
            modality_text, timestamp, tier, stability, recall_count,
            provider, model)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("mem_old", "text", vec.tobytes(), 384, "{}",
         "old memory", now - 100 * 86400, "episodic", 0.5, 5, "test", "test"),
    )
    c.execute(
        """INSERT INTO memories
           (memory_id, modality, embedding, embedding_dim, metadata,
            modality_text, timestamp, tier, stability, recall_count,
            provider, model)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("mem_fresh", "text", vec.tobytes(), 384, "{}",
         "fresh memory", now, "episodic", 1.0, 0, "test", "test"),
    )
    c.commit()

    # Test 1: boost_on_recall at cap stays at 1.0
    r = vm.boost_on_recall("mem_fresh")
    print("Test 1 boost_on_recall(mem_fresh):", r)
    assert r["found"] is True
    assert r["old_stability"] == 1.0
    assert abs(r["new_stability"] - 1.0) < 1e-6

    # Test 2: boost_on_recall bumps 0.5 -> 0.6
    r = vm.boost_on_recall("mem_old")
    print("Test 2 boost_on_recall(mem_old):", r)
    assert r["old_stability"] == 0.5
    assert abs(r["new_stability"] - 0.6) < 1e-6

    # Test 3: touch_recall also boosts stability (legacy schema)
    pre = c.execute(
        "SELECT stability, recall_count FROM memories WHERE memory_id=?",
        ("mem_fresh",),
    ).fetchone()
    r = vm.touch_recall("mem_fresh")
    print("Test 3 touch_recall(mem_fresh):", r)
    post = c.execute(
        "SELECT stability, recall_count FROM memories WHERE memory_id=?",
        ("mem_fresh",),
    ).fetchone()
    assert post["recall_count"] == pre["recall_count"] + 1
    assert post["stability"] == 1.0  # already at cap
    assert "old_stability" in r and "new_stability" in r

    # Test 4: get_decay_candidates only returns old memories
    c.execute(
        "UPDATE memories SET last_recalled_at = 0 WHERE memory_id = ?",
        ("mem_old",),
    )
    c.commit()
    candidates = vm.get_decay_candidates(threshold_days=30)
    print("Test 4 get_decay_candidates(30):", candidates)
    assert len(candidates) == 1
    assert candidates[0]["memory_id"] == "mem_old"

    # Test 5: decay_all decays mem_old; mem_old was 0.6, days_since ~100
    # decay_amount = 100 * 0.05/30 = 0.1667 -> new_stability = 0.6 - 0.1667 = 0.4333
    # NOT archived (above 0.05 floor)
    r = vm.decay_all(threshold_days=30, archive_floor=0.05)
    print("Test 5 decay_all:", r)
    assert r["decayed"] == 1
    assert r["archived"] == 0
    assert "by_tier" in r
    post = c.execute(
        "SELECT stability, tier FROM memories WHERE memory_id = ?",
        ("mem_old",),
    ).fetchone()
    # mem_old was boost'd to 0.6 above; check stability decayed
    assert post["stability"] < 0.6, f"stability should have decayed, got {post['stability']}"
    assert post["tier"] != "archived"

    # Test 6: low-stability memory gets archived
    c.execute(
        "UPDATE memories SET stability = 0.04, last_recalled_at = 0 WHERE memory_id = ?",
        ("mem_old",),
    )
    c.commit()
    r = vm.decay_all(threshold_days=30, archive_floor=0.05)
    print("Test 6 decay_all archive case:", r)
    state = c.execute("SELECT memory_id, stability, tier FROM memories").fetchall()
    archived = [dict(row) for row in state if row["tier"] == "archived"]
    print("  archived rows:", archived)
    assert len(archived) >= 1, "expected mem_old to be archived"
    assert archived[0]["memory_id"] == "mem_old"

    # Test 7: not_found returns found=False
    r = vm.boost_on_recall("nonexistent_id")
    print("Test 7 boost_on_recall not_found:", r)
    assert r["found"] is False

    # Test 8: new-schema no-op
    db2 = Path(tempfile.gettempdir()) / "mathir_decay_smoke_new.db"
    if db2.exists():
        db2.unlink()
    vm2 = VecMemory(db2, embedding_dim=384)  # auto-creates NEW schema
    assert vm2._schema_kind() == "new"
    # Store a memory via the new path
    vm2.store(
        "new1", np.random.randn(384).astype(np.float32),
        {"content": "hi", "agent": "test"},
    )
    r = vm2.boost_on_recall("new1")
    print("Test 8 boost_on_recall on new schema:", r)
    assert r.get("skipped") is True
    c2 = vm2.get_decay_candidates(threshold_days=30)
    assert c2 == []
    r = vm2.decay_all(threshold_days=30)
    print("Test 8 decay_all on new schema:", r)
    assert r.get("skipped") is True
    vm2.close()
    db2.unlink()

    vm.close()
    db_path.unlink()
    print("\nSMOKE TEST OK — all 8 test cases passed")


if __name__ == "__main__":
    main()
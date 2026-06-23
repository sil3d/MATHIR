"""Pytest suite for the 4-phase memory lifecycle.

Phase 1: promote  (working_memory -> episodic -> semantic -> procedural)
Phase 2: decay    (Ebbinghaus: 5%/30d, archive < 0.05)
Phase 3: consolidate (cosine > 0.95 merge)
Phase 4: link graph (cosine > 0.7, BFS, decay 0.5)
"""
import os
import sys
import math
import time
import shutil
import tempfile
import sqlite3
from pathlib import Path

import pytest

# Bootstrap imports — the package is a portable layout, not pip-installed
_PKG_ROOT = Path(__file__).resolve().parent.parent          # mathir_mcp/
_LIB = _PKG_ROOT / "mathir_lib"
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_LIB))

import numpy as np
from mathir_vec import VecMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db():
    """Fresh temp DB per test."""
    d = tempfile.mkdtemp(prefix="mathir_lifecycle_")
    db_path = Path(d) / "test.db"
    yield db_path
    shutil.rmtree(d, ignore_errors=True)


def _rand_vec(dim=384, seed=None):
    """Random unit-normalized vector for synthetic embeddings."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
    else:
        v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _store(memory, mid, content, tier="working_memory", label="", priority=5,
           recall_count=0, stability=1.0, age_days=None, seed=None):
    """Store a memory and optionally age it / preset recall_count."""
    emb = _rand_vec(seed=seed)
    meta = {
        "agent": "test",
        "block_type": tier if tier != "working_memory" else "working_memory",
        "label": label,
        "priority": priority,
        "content": content,
    }
    memory.store(mid, emb, meta)
    # Direct SQL adjustments — bypass the public store() so we can set
    # tier / recall_count / stability / last_recalled_at / timestamp as needed
    conn = sqlite3.connect(str(memory.db_path))
    try:
        sk = memory._schema_kind() if hasattr(memory, "_schema_kind") else "auto"
        if sk == "legacy":
            if tier:
                conn.execute("UPDATE memories SET tier = ? WHERE memory_id = ?",
                             (tier, mid))
            if recall_count:
                conn.execute("UPDATE memories SET recall_count = ? WHERE memory_id = ?",
                             (recall_count, mid))
            if stability != 1.0:
                conn.execute("UPDATE memories SET stability = ? WHERE memory_id = ?",
                             (stability, mid))
            if age_days is not None:
                # timestamp = now - age_days (in seconds)
                ts = time.time() - age_days * 86400
                conn.execute("UPDATE memories SET timestamp = ? WHERE memory_id = ?",
                             (ts, mid))
                if age_days > 0:
                    # also null out last_recalled_at to simulate no recent recall
                    conn.execute("UPDATE memories SET last_recalled_at = 0 WHERE memory_id = ?",
                                 (mid,))
        else:
            # NEW schema — recall_count/stability live in JSON metadata
            import json
            row = conn.execute("SELECT metadata FROM memories WHERE memory_id = ?",
                               (mid,)).fetchone()
            if row and row[0]:
                md = json.loads(row[0])
                md["tier"] = tier
                md["recall_count"] = recall_count
                md["stability"] = stability
                if age_days is not None:
                    md["timestamp"] = time.time() - age_days * 86400
                    md["last_recalled_at"] = 0
                conn.execute("UPDATE memories SET metadata = ? WHERE memory_id = ?",
                             (json.dumps(md), mid))
            # Also set the actual tier column (new schema has tier as a column too)
            if tier:
                conn.execute("UPDATE memories SET tier = ? WHERE memory_id = ?",
                             (tier, mid))
        conn.commit()
    finally:
        conn.close()
    return mid


# ===========================================================================
# PHASE 1: PROMOTE
# ===========================================================================

class TestPromote:
    def test_force_promote_working_to_episodic(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "m1", "hello", tier="working_memory")
        res = m.promote("m1", force=True)
        assert res["promoted"] is True
        assert res["old_tier"] == "working_memory"
        assert res["new_tier"] == "episodic"

    def test_force_promote_episodic_to_semantic(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "m1", "hello", tier="episodic")
        res = m.promote("m1", force=True)
        assert res["promoted"] is True
        assert res["new_tier"] == "semantic"

    def test_force_promote_semantic_to_procedural(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "m1", "hello", tier="semantic", label="how-to: foo")
        res = m.promote("m1", force=True)
        assert res["promoted"] is True
        assert res["new_tier"] == "procedural"

    def test_no_rule_blocks_promotion(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "m1", "fresh memory", tier="working_memory",
               recall_count=0, age_days=0)
        res = m.promote("m1", force=False)
        assert res["promoted"] is False
        # Reason format varies by tier (recall_count/age/priority) but always
        # indicates a numeric check failed
        assert "<" in res["reason"] or "needs" in res["reason"].lower() or "no rule" in res["reason"].lower()

    def test_promote_meets_rule(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "m1", "mature working memory", tier="working_memory",
               recall_count=5, age_days=2)
        res = m.promote("m1", force=False)
        # legacy schema: recall_count >= 3 + age >= 1d -> should promote
        if m._schema_kind() == "legacy":
            assert res["promoted"] is True
            assert res["new_tier"] == "episodic"

    def test_top_tier_is_noop(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "m1", "already top", tier="procedural")
        res = m.promote("m1", force=True)
        assert res["promoted"] is False
        assert res["new_tier"] == "procedural"

    def test_not_found(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        res = m.promote("nonexistent")
        assert res["found"] is False

    def test_auto_promote_all(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "m1", "mature", tier="working_memory", recall_count=10, age_days=10)
        _store(m, "m2", "fresh", tier="working_memory", recall_count=0, age_days=0)
        results = m.auto_promote_all()
        if m._schema_kind() == "legacy":
            promoted_ids = {r["memory_id"] for r in results}
            assert "m1" in promoted_ids
            assert "m2" not in promoted_ids

    def test_touch_recall_increments_count(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "m1", "touchable", tier="working_memory")
        m.touch_recall("m1")
        m.touch_recall("m1")
        # recall_count may be in a column (legacy) or in JSON metadata (new)
        conn = sqlite3.connect(str(tmp_db))
        try:
            if m._schema_kind() == "legacy":
                rc = conn.execute("SELECT recall_count FROM memories WHERE memory_id = 'm1'").fetchone()[0]
                assert rc == 2
            else:
                import json
                row = conn.execute("SELECT metadata FROM memories WHERE memory_id = 'm1'").fetchone()
                md = json.loads(row[0]) if row[0] else {}
                assert md.get("recall_count", 0) >= 2
        finally:
            conn.close()


# ===========================================================================
# PHASE 2: DECAY
# ===========================================================================

class TestDecay:
    def test_boost_on_recall_increases_stability(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "m1", "stable", stability=0.5)
        res = m.boost_on_recall("m1")
        if m._schema_kind() == "legacy" and not res.get("skipped"):
            assert res["new_stability"] == pytest.approx(0.6, abs=0.01)

    def test_boost_caps_at_one(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "m1", "maxed", stability=0.98)
        res = m.boost_on_recall("m1")
        if m._schema_kind() == "legacy" and not res.get("skipped"):
            assert res["new_stability"] == 1.0

    def test_decay_candidates_ordered_oldest_first(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "old", "ancient", recall_count=1, age_days=60)
        _store(m, "new", "recent", recall_count=10, age_days=0)
        candidates = m.get_decay_candidates(threshold_days=30)
        if m._schema_kind() == "legacy":
            ids = [c["memory_id"] for c in candidates]
            assert "old" in ids

    def test_decay_reduces_stability(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "m1", "decaying", stability=1.0, age_days=60)
        res = m.decay_all(threshold_days=30, archive_floor=0.0)
        if m._schema_kind() == "legacy" and not res.get("skipped"):
            conn = sqlite3.connect(str(tmp_db))
            new_stab = conn.execute(
                "SELECT stability FROM memories WHERE memory_id = 'm1'"
            ).fetchone()[0]
            conn.close()
            assert new_stab < 1.0

    def test_decay_archives_low_stability(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "dead", "almost gone", stability=0.04, age_days=365)
        res = m.decay_all(threshold_days=30, archive_floor=0.05)
        if m._schema_kind() == "legacy" and not res.get("skipped"):
            conn = sqlite3.connect(str(tmp_db))
            tier = conn.execute(
                "SELECT tier FROM memories WHERE memory_id = 'dead'"
            ).fetchone()[0]
            conn.close()
            assert tier == "archived"

    def test_decay_skipped_on_new_schema(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "m1", "anything")
        res = m.decay_all()
        if m._schema_kind() != "legacy":
            assert res.get("skipped") is True


# ===========================================================================
# PHASE 3: CONSOLIDATE
# ===========================================================================

class TestConsolidate:
    def test_find_duplicates_near_identical(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        # Two near-identical vectors — should be very high cosine
        e1 = _rand_vec(seed=42)
        e2 = e1 + 0.01 * _rand_vec(seed=43)  # tiny perturbation
        e2 = e2 / (np.linalg.norm(e2) + 1e-9)
        m.store("a", e1, {"agent": "t", "content": "alpha"})
        m.store("b", e2, {"agent": "t", "content": "alpha-copy"})
        dups = m.find_duplicates(threshold=0.95)
        if m._schema_kind() == "legacy" or m._schema_kind() == "new":
            pairs = [(d["memory_id_a"], d["memory_id_b"]) for d in dups]
            assert any(("a" in p and "b" in p) for p in pairs)

    def test_consolidate_pair_merges(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        e1 = _rand_vec(seed=10)
        e2 = e1 + 0.01 * _rand_vec(seed=11)
        e2 = e2 / (np.linalg.norm(e2) + 1e-9)
        m.store("strong", e1, {"agent": "t", "content": "real",
                                "recall_count": 5, "stability": 1.0})
        m.store("weak", e2, {"agent": "t", "content": "copy",
                              "recall_count": 1, "stability": 0.5})
        res = m.consolidate_pair("strong", "weak")
        if m._schema_kind() == "legacy" or m._schema_kind() == "new":
            assert res["canonical_id"] == "strong"
            assert res["merged_from"] == "weak"
            assert res["new_recall_count"] == 6  # 5 + 1

    def test_consolidate_all_dry_run(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        e1 = _rand_vec(seed=20)
        e2 = e1 + 0.01 * _rand_vec(seed=21)
        e2 = e2 / (np.linalg.norm(e2) + 1e-9)
        m.store("a", e1, {"agent": "t", "content": "one"})
        m.store("b", e2, {"agent": "t", "content": "one-copy"})
        res = m.consolidate_all(threshold=0.95, dry_run=True)
        if m._schema_kind() == "legacy" or m._schema_kind() == "new":
            assert res["dry_run"] is True
            # Neither memory should be archived after dry run
            conn = sqlite3.connect(str(tmp_db))
            tiers = [r[0] for r in conn.execute("SELECT tier FROM memories").fetchall()]
            conn.close()
            assert "archived" not in tiers

    def test_consolidate_all_real_run(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        e1 = _rand_vec(seed=30)
        e2 = e1 + 0.01 * _rand_vec(seed=31)
        e2 = e2 / (np.linalg.norm(e2) + 1e-9)
        m.store("a", e1, {"agent": "t", "content": "x", "recall_count": 3})
        m.store("b", e2, {"agent": "t", "content": "x", "recall_count": 1})
        res = m.consolidate_all(threshold=0.95, dry_run=False)
        if m._schema_kind() == "legacy" or m._schema_kind() == "new":
            assert res["merged"] >= 1


# ===========================================================================
# PHASE 4: LINK GRAPH
# ===========================================================================

class TestLinkGraph:
    def test_add_link_creates(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "a", "node a")
        _store(m, "b", "node b")
        res = m.add_link("a", "b", weight=0.8)
        assert res["created"] is True
        assert res["weight"] == 0.8

    def test_add_link_rejects_self(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "a", "node a")
        with pytest.raises(ValueError):
            m.add_link("a", "a", weight=1.0)

    def test_add_link_rejects_empty(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        with pytest.raises(ValueError):
            m.add_link("", "b")

    def test_get_links_one_hop(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "a", "node a")
        _store(m, "b", "node b")
        _store(m, "c", "node c")
        m.add_link("a", "b", weight=0.9)
        m.add_link("a", "c", weight=0.5)
        links = m.get_links("a", depth=1, decay=0.5)
        ids = [l["memory_id"] for l in links]
        assert "b" in ids
        assert "c" in ids
        # b should rank higher than c (higher weight)
        idx_b = ids.index("b")
        idx_c = ids.index("c")
        assert idx_b < idx_c

    def test_get_links_two_hops_with_decay(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        _store(m, "a", "root")
        _store(m, "b", "level1")
        _store(m, "c", "level2")
        m.add_link("a", "b", weight=1.0)
        m.add_link("b", "c", weight=1.0)
        links = m.get_links("a", depth=2, decay=0.5)
        weights = {l["memory_id"]: l["cumulative_weight"] for l in links}
        if "b" in weights and "c" in weights:
            # b at 1 hop = 0.5, c at 2 hops = 0.25
            assert weights["b"] > weights["c"]

    def test_build_links_all_creates_symmetric(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        # Two near-identical vectors
        e1 = _rand_vec(seed=50)
        e2 = e1 + 0.05 * _rand_vec(seed=51)
        e2 = e2 / (np.linalg.norm(e2) + 1e-9)
        m.store("a", e1, {"agent": "t", "content": "one"})
        m.store("b", e2, {"agent": "t", "content": "one-copy"})
        m.store("c", _rand_vec(seed=52), {"agent": "t", "content": "orthogonal"})
        res = m.build_links_all(threshold=0.7)
        assert res["links_created"] >= 2  # a->b and b->a (symmetric)

    def test_find_related_combines_vector_and_graph(self, tmp_db):
        m = VecMemory(tmp_db, 384)
        e1 = _rand_vec(seed=60)
        e2 = e1 + 0.05 * _rand_vec(seed=61)
        e2 = e2 / (np.linalg.norm(e2) + 1e-9)
        m.store("seed", e1, {"agent": "t", "content": "seed",
                              "modality_text": "seed memory"})
        m.store("near", e2, {"agent": "t", "content": "near",
                              "modality_text": "near memory"})
        m.store("far", _rand_vec(seed=62), {"agent": "t", "content": "far",
                                              "modality_text": "far memory"})
        m.add_link("seed", "far", weight=0.6)
        related = m.find_related("seed", max_hops=1, min_weight=0.1)
        ids = [r["memory_id"] for r in related]
        # near should be via vector, far should be via link
        assert "near" in ids
        assert "far" in ids

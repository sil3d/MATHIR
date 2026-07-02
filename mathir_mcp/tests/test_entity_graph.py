"""Tests for entity extraction (mathir_entity_graph) and the entity-linked
graph layer (VecMemory.build_entity_links_all)."""
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
    from mathir_lib.mathir_entity_graph import extract_entities
    from mathir_lib.mathir_vec import VecMemory
except ImportError:
    from mathir_entity_graph import extract_entities  # type: ignore
    from mathir_vec import VecMemory  # type: ignore


def test_extract_entities_finds_shared_bridge_entity():
    """The two sides of a HotpotQA-style bridge share an entity even though
    the sentences are about different topics."""
    a = "Kiss and Tell is a 1945 American comedy film in which Shirley Temple played Corliss Archer."
    b = "Shirley Temple later served as United States Chief of Protocol."
    ea = extract_entities(a)
    eb = extract_entities(b)
    shared = ea & eb
    assert any("shirley temple" in s for s in shared), (
        f"expected 'shirley temple' shared between the two, got a={ea} b={eb}"
    )


def test_extract_entities_empty_and_generic():
    assert extract_entities("") == set()
    assert extract_entities("   ") == set()
    # Very short / generic tokens shouldn't dominate.
    ents = extract_entities("it was a good day")
    assert all(len(e) >= 3 for e in ents)


def test_build_entity_links_creates_edge_between_entity_sharing_memories(tmp_path):
    """Two memories sharing an entity but with LOW embedding similarity still
    get an entity edge -- the exact case the cosine graph misses."""
    db = tmp_path / "entity_links.db"
    vm = VecMemory(db, embedding_dim=384)

    # Deliberately near-orthogonal embeddings so a cosine graph would NOT link
    # these two -- only the shared entity should.
    rng = np.random.RandomState(0)
    emb_a = rng.randn(384).astype(np.float32)
    emb_b = rng.randn(384).astype(np.float32)  # independent -> ~orthogonal

    vm.store("mem_a", emb_a, {
        "content": "Kiss and Tell is a 1945 film starring Shirley Temple.",
        "agent": "t", "block_type": "episodic", "label": "", "priority": 5})
    vm.store("mem_b", emb_b, {
        "content": "Shirley Temple became United States Chief of Protocol.",
        "agent": "t", "block_type": "episodic", "label": "", "priority": 5})
    vm.store("mem_c", rng.randn(384).astype(np.float32), {
        "content": "Photosynthesis converts sunlight into chemical energy in plants.",
        "agent": "t", "block_type": "episodic", "label": "", "priority": 5})

    stats = vm.build_entity_links_all()
    assert stats["links_created"] >= 2, f"expected an entity edge, got {stats}"

    # mem_a and mem_b (share 'shirley temple') must be linked; mem_c (no
    # shared entity) must NOT be linked to either.
    links = vm.get_links("mem_a", depth=1)
    linked_ids = {n.get("memory_id") for n in links.get("nodes", [])} if isinstance(links, dict) else set()
    # get_links shape can vary; fall back to a direct DB check for robustness.
    conn = vm._get_conn()
    a_targets = {r[0] for r in conn.execute(
        "SELECT target_id FROM memory_links WHERE source_id = 'mem_a'").fetchall()}
    assert "mem_b" in a_targets, f"mem_a should link to mem_b via shared entity; targets={a_targets}"
    assert "mem_c" not in a_targets, f"mem_a should NOT link to unrelated mem_c; targets={a_targets}"


def test_ensure_embedding_model_records_model_on_new_db(tmp_path):
    """A brand-new DB (no stored model yet) records whichever model is
    passed in and returns it -- this is how a NEW project picks up
    whatever the current configured default is (e.g. e5-small)."""
    db = tmp_path / "model_track_new.db"
    vm = VecMemory(db, embedding_dim=384)
    assert vm.get_stored_embedding_model() is None
    resolved = vm.ensure_embedding_model("intfloat/multilingual-e5-small")
    assert resolved == "intfloat/multilingual-e5-small"
    assert vm.get_stored_embedding_model() == "intfloat/multilingual-e5-small"


def test_ensure_embedding_model_keeps_existing_model_on_old_db(tmp_path):
    """An EXISTING DB (model already recorded) keeps its original model
    even if the configured default changes later -- this is the whole
    point: swapping MATHIR's default embedder must not silently corrupt
    retrieval on projects embedded with the old model."""
    db = tmp_path / "model_track_old.db"
    vm = VecMemory(db, embedding_dim=384)
    first = vm.ensure_embedding_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    assert first == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Simulate a config change to a new default -- must NOT affect this DB.
    second = vm.ensure_embedding_model("intfloat/multilingual-e5-small")
    assert second == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", (
        "existing DB must keep its original model, not silently switch"
    )


def test_ensure_embedding_model_persists_across_instances(tmp_path):
    """The recorded model survives a daemon restart (new VecMemory instance
    on the same db_path)."""
    db = tmp_path / "model_track_persist.db"
    vm1 = VecMemory(db, embedding_dim=384)
    vm1.ensure_embedding_model("intfloat/multilingual-e5-small")
    vm1.close()

    vm2 = VecMemory(db, embedding_dim=384)
    assert vm2.get_stored_embedding_model() == "intfloat/multilingual-e5-small"


try:
    from mathir_lib.mathir_mcp_server import get_model_prefixes
except ImportError:
    from mathir_mcp_server import get_model_prefixes  # type: ignore


def test_get_model_prefixes_e5_small_has_query_passage_prefixes():
    q, p = get_model_prefixes("intfloat/multilingual-e5-small")
    assert q == "query: "
    assert p == "passage: "


def test_get_model_prefixes_default_minilm_has_no_prefixes():
    q, p = get_model_prefixes("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    assert q == ""
    assert p == ""


def test_get_model_prefixes_unknown_model_defaults_to_no_prefix():
    q, p = get_model_prefixes("some/other-model-nobody-registered")
    assert q == ""
    assert p == ""


def test_ensure_embedding_model_backfills_legacy_default_for_pre_existing_data(tmp_path):
    """CRITICAL: a DB created BEFORE the db_meta table existed (has real
    memories but no db_meta row) must be backfilled with the LEGACY
    default model, never silently adopt whatever the CURRENT configured
    default is -- otherwise changing MATHIR's default embedder corrupts
    every pre-existing project's retrieval on first access after upgrade.
    This is a real bug caught live: locomo_conv_2 (1326 real memories,
    embedded with the old default) got mis-pinned to a newly-configured
    default model before this fix.
    """
    db = tmp_path / "legacy_backfill.db"
    vm = VecMemory(db, embedding_dim=384)
    rng = np.random.RandomState(0)
    # Simulate pre-existing content (as if saved before db_meta existed).
    vm.store("mem_old1", rng.randn(384).astype(np.float32),
             {"content": "old memory", "agent": "t", "block_type": "episodic",
              "label": "", "priority": 5})
    # Manually clear db_meta to simulate "created before this migration".
    conn = vm._get_conn()
    conn.execute("DELETE FROM db_meta")
    conn.commit()

    resolved = vm.ensure_embedding_model("intfloat/multilingual-e5-small")
    assert resolved == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", (
        f"pre-existing DB with real content must backfill the legacy default, "
        f"not adopt the new configured default -- got {resolved}"
    )


def test_ensure_embedding_model_uses_new_default_for_genuinely_empty_db(tmp_path):
    """A genuinely empty DB (no memories, no db_meta row) is a brand-new
    project -- it should adopt whatever model is passed (the current
    configured default), NOT the legacy fallback."""
    db = tmp_path / "genuinely_new.db"
    vm = VecMemory(db, embedding_dim=384)
    resolved = vm.ensure_embedding_model("intfloat/multilingual-e5-small")
    assert resolved == "intfloat/multilingual-e5-small"


def test_touch_recall_boosts_stability_on_new_schema(tmp_path):
    """CRITICAL LIFECYCLE BUG: on the 'new' schema (used by every real
    MATHIR project -- confirmed live on locomo_conv_2 and MATHIR's own
    project), touch_recall() incremented recall_count but left stability
    untouched, so decay_all() could only ever decrease stability -- a
    memory recalled 20 times decayed identically to one never recalled at
    all, defeating the entire reinforce-on-use lifecycle design. The
    stability column exists on the new schema too (added by the
    idempotent migration in _ensure_db), touch_recall just never wrote to
    it. This test proves the fix: frequent recall must measurably protect
    against decay.
    """
    db = tmp_path / "lifecycle_boost.db"
    vm = VecMemory(db, embedding_dim=384)
    assert vm._schema_kind() == "new"
    rng = np.random.RandomState(0)
    emb = rng.randn(384).astype(np.float32)
    vm.store("mem_used", emb, {"content": "frequently recalled", "agent": "t",
                               "block_type": "episodic", "label": "", "priority": 5})
    vm.store("mem_unused", rng.randn(384).astype(np.float32),
             {"content": "never recalled", "agent": "t",
              "block_type": "episodic", "label": "", "priority": 5})

    for _ in range(20):
        vm.touch_recall("mem_used")

    conn = vm._get_conn()
    row = conn.execute("SELECT stability FROM memories WHERE memory_id = 'mem_used'").fetchone()
    assert row["stability"] > 1.0 - 1e-9 or row["stability"] == 1.0, (
        "stability should be boosted (capped at 1.0) after 20 recalls, not left untouched"
    )
    # More discriminating: recall a fresh memory once and check stability
    # actually moved from whatever its floor/start value is.
    vm.store("mem_probe", rng.randn(384).astype(np.float32),
             {"content": "probe", "agent": "t", "block_type": "episodic",
              "label": "", "priority": 5})
    conn.execute("UPDATE memories SET stability = 0.5 WHERE memory_id = 'mem_probe'")
    conn.commit()
    r = vm.touch_recall("mem_probe")
    row2 = conn.execute("SELECT stability FROM memories WHERE memory_id = 'mem_probe'").fetchone()
    assert row2["stability"] > 0.5, f"expected stability boost above 0.5, got {row2['stability']}"
    assert r["old_stability"] == 0.5
    assert r["new_stability"] == pytest.approx(0.6)


def test_decay_after_frequent_recall_protects_more_than_no_recall(tmp_path):
    """The end-to-end proof: a heavily-recalled memory must decay LESS
    than an identical never-recalled memory over repeated decay/recall
    cycles -- this is the actual reinforce-vs-forget contract the whole
    lifecycle system exists to provide.

    Both memories start at stability=1.0 (the schema default), so a
    single recall's boost is invisible while at the ceiling -- this test
    runs an initial decay pass to bring both below 1.0 first (simulating
    a month of disuse for both), THEN recalls mem_used repeatedly
    (partially reinforcing it back up) while mem_unused stays cold, THEN
    applies a second decay pass and compares.
    """
    import time as _time
    db = tmp_path / "lifecycle_e2e.db"
    vm = VecMemory(db, embedding_dim=384)
    rng = np.random.RandomState(1)

    vm.store("mem_used", rng.randn(384).astype(np.float32),
             {"content": "used", "agent": "t", "block_type": "episodic",
              "label": "", "priority": 5})
    vm.store("mem_unused", rng.randn(384).astype(np.float32),
             {"content": "unused", "agent": "t", "block_type": "episodic",
              "label": "", "priority": 5})

    conn = vm._get_conn()
    stale_ts = _time.time() - 60 * 86400
    conn.execute("UPDATE memories SET last_recalled_at = ? WHERE memory_id IN ('mem_used','mem_unused')",
                [stale_ts])
    conn.commit()
    vm.decay_all(threshold_days=30)  # both drop below 1.0 equally

    for _ in range(5):
        vm.touch_recall("mem_used")  # only mem_used gets reinforced

    conn.execute("UPDATE memories SET last_recalled_at = ? WHERE memory_id = 'mem_unused'",
                [stale_ts])
    conn.commit()
    vm.decay_all(threshold_days=30)

    used_stability = conn.execute(
        "SELECT stability FROM memories WHERE memory_id = 'mem_used'").fetchone()["stability"]
    unused_stability = conn.execute(
        "SELECT stability FROM memories WHERE memory_id = 'mem_unused'").fetchone()["stability"]
    assert used_stability > unused_stability, (
        f"a heavily-recalled memory must retain MORE stability after decay than an "
        f"unused one -- got used={used_stability} unused={unused_stability}"
    )

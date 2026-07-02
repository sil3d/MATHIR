"""Tests for entity extraction (mathir_entity_graph) and the entity-linked
graph layer (VecMemory.build_entity_links_all)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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

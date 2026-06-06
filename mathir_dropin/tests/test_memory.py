"""
MATHIR Drop-in — Tests.

10 critical tests covering:

    1.  test_init_default
    2.  test_init_with_db_path
    3.  test_store_and_recall
    4.  test_recall_with_modality_filter
    5.  test_persistence
    6.  test_dimension_mismatch_raises
    7.  test_forget
    8.  test_get_stats
    9.  test_concurrent_stores
    10. test_text_search_via_fts5

Run with::

    python -m pytest mathir_dropin/tests/test_memory.py -v

or::

    cd mathir_dropin && python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

import pytest
import torch

# Make the package importable when running from the project root.
# Tests live in mathir_dropin/tests/, so the package itself is the
# parent of `tests/`. We need the GRAND-parent on sys.path because the
# package is named `mathir_dropin`, not `tests`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DROPIN = os.path.dirname(_HERE)            # .../mathir_dropin
_PARENT = os.path.dirname(_DROPIN)          # project root
for p in (_PARENT, _DROPIN):
    if p not in sys.path:
        sys.path.insert(0, p)

# `mathir_dropin` is the parent directory of `tests/`, so the project
# root must be on sys.path for `import mathir_dropin` to work when
# running `pytest tests/` from inside the package.
import mathir_dropin  # noqa: E402

from mathir_dropin import (  # noqa: E402
    DEFAULT_CONFIG,
    DimensionMismatchError,
    MATHIRMemory,
    StorageError,
    configure,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db():
    """Yield a fresh SQLite path in a tempdir, clean up afterwards."""
    d = tempfile.mkdtemp(prefix="mathir_test_")
    path = os.path.join(d, "test.db")
    yield path
    # Best-effort cleanup; Windows holds the file briefly.
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def memory_factory(tmp_db):
    """Factory that returns a fresh MATHIRMemory bound to ``tmp_db``."""
    created = []

    def _make(dim: int = 64, **kwargs) -> MATHIRMemory:
        cfg = configure({
            "memory": {
                "embedding_dim": dim,
                "working_capacity": 16,
                "episodic_capacity": 32,
                "semantic_prototypes": 8,
                "immunological_capacity": 16,
            },
        })
        # Allow caller to pass storage config.
        if "db_path" not in kwargs:
            kwargs["db_path"] = tmp_db
        m = MATHIRMemory(embedding_dim=dim, config=cfg, **kwargs)
        created.append(m)
        return m

    yield _make

    for m in created:
        try:
            m._store.close() if m._store else None
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1. test_init_default
# ---------------------------------------------------------------------------

def test_init_default(memory_factory):
    """A new MATHIRMemory with no config has the default capacities."""
    mem = memory_factory(dim=128, db_path=None)
    stats = mem.get_stats()
    assert stats["embedding_dim"] == 128
    assert stats["tier_working"]["capacity"] == 16
    assert stats["tier_episodic"]["capacity"] == 32
    assert stats["tier_semantic"]["num_prototypes"] == 8
    assert stats["tier_immune"]["capacity"] == 16
    # All tiers start empty.
    assert stats["tier_working"]["usage"] == 0
    assert stats["tier_episodic"]["usage"] == 0
    assert stats["tier_immune"]["usage"] == 0


# ---------------------------------------------------------------------------
# 2. test_init_with_db_path
# ---------------------------------------------------------------------------

def test_init_with_db_path(memory_factory, tmp_db):
    """Passing ``db_path`` should create a SQLite file and a Store object."""
    mem = memory_factory(dim=64, db_path=tmp_db)
    assert mem._store is not None
    assert mem._store.db_path == tmp_db
    # The file should exist (sqlite creates it lazily on connect).
    assert os.path.exists(tmp_db)
    # The schema should be in place.
    assert mem._store.count() == 0


# ---------------------------------------------------------------------------
# 3. test_store_and_recall
# ---------------------------------------------------------------------------

def test_store_and_recall(memory_factory):
    """Stored embeddings can be retrieved with high similarity."""
    torch.manual_seed(42)
    mem = memory_factory(dim=64, db_path=None)

    # Store 5 distinct vectors.
    base = torch.randn(5, 64)
    ids = []
    for i in range(5):
        mid = mem.store(base[i].unsqueeze(0), {"text": f"item-{i}"})
        ids.append(mid)
    assert all(isinstance(i, str) and i.startswith("mem_") for i in ids)

    # Recall with the *same* vector should produce similarity 1.0
    # (cosine of a vector with itself).
    hits = mem.recall(base[0].unsqueeze(0), k=3)
    assert len(hits) >= 1
    assert hits[0]["similarity"] > 0.99
    # The top hit should be the stored one.
    assert hits[0]["memory_id"] in ids or hits[0]["memory_id"].startswith("live_")


# ---------------------------------------------------------------------------
# 4. test_recall_with_modality_filter
# ---------------------------------------------------------------------------

def test_recall_with_modality_filter(memory_factory, tmp_db):
    """``modality`` filter narrows the SQLite result set."""
    torch.manual_seed(0)
    mem = memory_factory(dim=64, db_path=tmp_db)

    # Insert a mix of modalities directly into SQLite to control the
    # ground truth exactly.
    store = mem._store
    assert store is not None
    for i in range(4):
        emb = torch.randn(64)
        store.insert(
            memory_id=f"text_{i}",
            embedding=emb,
            metadata={"text": f"text doc {i}"},
            modality="text",
            modality_text=f"text doc {i}",
        )
    for i in range(3):
        emb = torch.randn(64)
        store.insert(
            memory_id=f"image_{i}",
            embedding=emb,
            metadata={"text": f"image doc {i}"},
            modality="image",
            modality_text=f"image doc {i}",
        )

    q = torch.randn(64)
    text_hits = mem.recall(q, k=10, modality="text")
    image_hits = mem.recall(q, k=10, modality="image")
    all_hits = mem.recall(q, k=10)

    assert all(h["modality"] == "text" for h in text_hits)
    assert all(h["modality"] == "image" for h in image_hits)
    assert len(text_hits) == 4
    assert len(image_hits) == 3
    assert len(all_hits) == 7


# ---------------------------------------------------------------------------
# 5. test_persistence
# ---------------------------------------------------------------------------

def test_persistence(memory_factory, tmp_db):
    """Memories saved to SQLite are reloaded in a fresh instance."""
    torch.manual_seed(1)
    cfg = configure({
        "memory": {
            "embedding_dim": 64,
            "working_capacity": 16,
            "episodic_capacity": 32,
        },
    })

    # ---- Phase 1: write -----------------------------------------------
    mem1 = MATHIRMemory(embedding_dim=64, config=cfg, db_path=tmp_db)
    for i in range(5):
        mem1.store(torch.randn(1, 64), {"text": f"persist {i}"})
    # force-flush (already auto-saved but be explicit)
    mem1.save()
    assert mem1._store.count() == 5
    mem1._store.close()

    # ---- Phase 2: reopen ----------------------------------------------
    mem2 = MATHIRMemory(embedding_dim=64, config=cfg, db_path=tmp_db)
    mem2.load()
    # Now episodic in-memory tier should contain all 5.
    assert mem2.get_stats()["tier_episodic"]["usage"] == 5
    # Recall should find them.
    hits = mem2.recall(torch.randn(1, 64), k=10)
    assert len(hits) == 5


# ---------------------------------------------------------------------------
# 6. test_dimension_mismatch_raises
# ---------------------------------------------------------------------------

def test_dimension_mismatch_raises(memory_factory):
    """Wrong-dim embeddings raise a clear ``DimensionMismatchError``."""
    mem = memory_factory(dim=64, db_path=None)

    with pytest.raises(DimensionMismatchError) as excinfo:
        mem.store(torch.randn(1, 128), {"text": "wrong dim"})
    assert excinfo.value.expected == 64
    assert excinfo.value.got == 128

    with pytest.raises(DimensionMismatchError):
        mem.recall(torch.randn(1, 256))

    with pytest.raises(DimensionMismatchError):
        mem.perceive(torch.randn(1, 32))


# ---------------------------------------------------------------------------
# 7. test_forget
# ---------------------------------------------------------------------------

def test_forget(memory_factory):
    """``forget()`` returns the number of dropped memories and doesn't crash."""
    torch.manual_seed(7)
    mem = memory_factory(dim=32, db_path=None)

    # Store 10 random vectors.
    for _ in range(10):
        mem.store(torch.randn(1, 32))

    before = mem.get_stats()["tier_episodic"]["usage"]
    assert before == 10

    # Very high threshold => most will look "uninformative" => drop many.
    dropped = mem.forget(threshold=0.99)
    after = mem.get_stats()["tier_episodic"]["usage"]
    assert dropped >= 0
    assert after == before - dropped
    # Always keep at least 1.
    assert after >= 1


# ---------------------------------------------------------------------------
# 8. test_get_stats
# ---------------------------------------------------------------------------

def test_get_stats(memory_factory):
    """``get_stats()`` returns a complete and consistent snapshot."""
    mem = memory_factory(dim=48, db_path=None)

    # Empty.
    s0 = mem.get_stats()
    assert s0["tier_working"]["usage"] == 0
    assert s0["tier_episodic"]["usage"] == 0
    assert s0["tier_semantic"]["used_prototypes"] == 0
    assert s0["tier_immune"]["usage"] == 0
    assert s0["router"]["type"] in ("kl_constrained", "uniform")

    # After a few stores, all relevant counters increase.
    for _ in range(5):
        mem.store(torch.randn(1, 48), {"text": "x"})
    s1 = mem.get_stats()
    assert s1["tier_working"]["usage"] > 0
    assert s1["tier_episodic"]["usage"] == 5
    assert s1["tier_semantic"]["used_prototypes"] > 0
    assert s1["tier_immune"]["usage"] > 0
    assert s1["storage"]["type"] in ("sqlite", "memory")


# ---------------------------------------------------------------------------
# 9. test_concurrent_stores
# ---------------------------------------------------------------------------

def test_concurrent_stores(memory_factory):
    """Stores from multiple threads must not corrupt in-memory state.

    We spawn 4 threads, each inserting 25 vectors, and verify that
    the final episodic count is exactly 100. SQLite's per-thread
    transactions are serialised by the internal lock, so the
    invariant holds. The fixture defaults to capacity=32; we override
    here so 100 fits.
    """
    cfg = configure({
        "memory": {
            "embedding_dim": 32,
            "working_capacity": 16,
            "episodic_capacity": 200,
            "semantic_prototypes": 16,
            "immunological_capacity": 16,
        },
    })
    mem = MATHIRMemory(embedding_dim=32, config=cfg, db_path=None)

    def worker(start: int) -> None:
        for i in range(25):
            mem.store(torch.randn(1, 32), {"text": f"t{start}-{i}"})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert mem.get_stats()["tier_episodic"]["usage"] == 100


# ---------------------------------------------------------------------------
# 10. test_text_search_via_fts5
# ---------------------------------------------------------------------------

def test_text_search_via_fts5(memory_factory, tmp_db):
    """FTS5 BM25 search returns rows matching the query text."""
    mem = memory_factory(dim=32, db_path=tmp_db)
    assert mem._store is not None

    # Insert rows with distinguishable text.
    for i in range(6):
        text = (
            "transformer attention mechanism" if i < 3
            else "convolutional neural network filter"
        )
        mem.store(torch.randn(1, 32), {"text": text})
    mem.save()

    # Search for "transformer" — should return the first 3.
    transformer_hits = mem.recall_text("transformer", k=10)
    assert len(transformer_hits) == 3
    for h in transformer_hits:
        assert "transformer" in h.get("modality_text", "").lower()

    # Search for "convolutional" — should return the other 3.
    conv_hits = mem.recall_text("convolutional", k=10)
    assert len(conv_hits) == 3
    for h in conv_hits:
        assert "convolutional" in h.get("modality_text", "").lower()

    # Search with an obviously non-matching query — should return 0.
    none_hits = mem.recall_text("quantum-gravity-astrophysics", k=10)
    assert len(none_hits) == 0

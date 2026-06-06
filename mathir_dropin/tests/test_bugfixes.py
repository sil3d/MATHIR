"""
MATHIR Drop-in — Regression tests for three robustness bugs.

These cover behaviours that were silently broken:

    1. test_save_persists_after_autosave_off
       ``save()`` must actually flush in-memory rows to SQLite when
       ``auto_save`` is disabled (was a no-op → data loss).

    2. test_forget_prunes_sqlite_rows
       After ``forget()`` drops in-memory rows, the SQLite store must
       not keep serving the pruned memories (the two stores diverged).

    3. test_concurrent_perceive_and_store
       ``perceive()`` must take the same lock as ``store()`` so the
       working-buffer pointer is not corrupted under concurrency.

    4. test_delete_returns_true_only_when_found
       ``delete()`` must report False for an unknown id and True for a
       real one (the old return logic had a dead clause).

Run with::

    python -m pytest mathir_dropin/tests/test_bugfixes.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DROPIN = os.path.dirname(_HERE)
_PARENT = os.path.dirname(_DROPIN)
for p in (_PARENT, _DROPIN):
    if p not in sys.path:
        sys.path.insert(0, p)

from mathir_dropin import MATHIRMemory, configure  # noqa: E402


@pytest.fixture
def tmp_db():
    d = tempfile.mkdtemp(prefix="mathir_bugfix_")
    path = os.path.join(d, "test.db")
    yield path
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def _cfg(dim: int = 64):
    return configure({
        "memory": {
            "embedding_dim": dim,
            "working_capacity": 16,
            "episodic_capacity": 32,
            "semantic_prototypes": 8,
            "immunological_capacity": 16,
        },
    })


# ---------------------------------------------------------------------------
# Bug 1 — save() was a no-op
# ---------------------------------------------------------------------------

def test_save_persists_after_autosave_off(tmp_db):
    """With auto_save disabled, store() skips the DB but save() must flush."""
    torch.manual_seed(1)
    cfg = _cfg()
    cfg["storage"]["auto_save"] = False
    mem = MATHIRMemory(embedding_dim=64, config=cfg, db_path=tmp_db)

    vec = torch.randn(1, 64)
    mem.store(vec, {"text": "important note"})

    # auto_save is off → nothing on disk yet.
    assert mem._store.count() == 0

    mem.save()

    # After an explicit save(), the row must be persisted and recallable.
    assert mem._store.count() == 1
    hits = mem.recall(vec, k=1)
    assert len(hits) == 1
    assert hits[0]["metadata"].get("text") == "important note"
    mem._store.close()


# ---------------------------------------------------------------------------
# Bug 2 — forget() left SQLite rows behind
# ---------------------------------------------------------------------------

def test_forget_prunes_sqlite_rows(tmp_db):
    """forget() must remove dropped memories from SQLite too, so recall
    does not keep returning 'forgotten' rows."""
    torch.manual_seed(2)
    mem = MATHIRMemory(embedding_dim=64, config=_cfg(), db_path=tmp_db)

    for i in range(10):
        mem.store(torch.randn(1, 64), {"text": f"item-{i}"})
    before = mem._store.count()
    assert before == 10

    # Aggressive threshold prunes almost everything.
    dropped = mem.forget(threshold=0.99)
    assert dropped > 0

    # SQLite must reflect the prune: the on-disk count drops by exactly
    # the number forgotten, and never exceeds the in-memory usage.
    after = mem._store.count()
    assert after == before - dropped
    assert after == mem.episodic.usage
    mem._store.close()


# ---------------------------------------------------------------------------
# Bug 3 — perceive() bypassed the op lock
# ---------------------------------------------------------------------------

def test_concurrent_perceive_and_store():
    """Concurrent perceive()/store() must not corrupt the working buffer
    pointer. Usage must never exceed capacity and the buffer stays valid."""
    torch.manual_seed(3)
    mem = MATHIRMemory(embedding_dim=64, config=_cfg(), db_path=None)
    cap = mem.working.capacity

    errors = []

    def hammer(do_store):
        try:
            for _ in range(200):
                v = torch.randn(1, 64)
                if do_store:
                    mem.store(v, {"text": "x"})
                else:
                    mem.perceive(v)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=hammer, args=(i % 2 == 0,))
               for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"races raised: {errors}"
    # Pointer/count invariants must hold.
    assert 0 <= int(mem.working.ptr.item()) < cap
    assert 0 <= mem.working.usage <= cap


# ---------------------------------------------------------------------------
# Bug 4 — delete() return logic
# ---------------------------------------------------------------------------

def test_delete_returns_true_only_when_found(tmp_db):
    torch.manual_seed(4)
    mem = MATHIRMemory(embedding_dim=64, config=_cfg(), db_path=tmp_db)

    mid = mem.store(torch.randn(1, 64), {"text": "deleteme"})
    assert mem.delete(mid) is True
    # Second delete of the same id: already gone.
    assert mem.delete(mid) is False
    # Never-existed id.
    assert mem.delete("mem_doesnotexist") is False
    mem._store.close()

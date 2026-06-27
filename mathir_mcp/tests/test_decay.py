"""Tests for VecMemory.run_maintenance() — the single canonical lifecycle.

run_maintenance() composes decay, promote, dedupe, and link-build into one
call.  These tests verify it returns the expected shape and is idempotent.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports — try package-qualified first, fall back to flat
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MCP_ROOT = _HERE.parent
if str(_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT))

try:
    from mathir_lib.mathir_vec import VecMemory
except ImportError:
    from mathir_vec import VecMemory  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_two_memories(vm: VecMemory) -> None:
    """Insert two arbitrary memories so the lifecycle has work to do."""
    emb_a = np.random.RandomState(42).randn(vm.embedding_dim).astype(np.float32)
    emb_b = np.random.RandomState(43).randn(vm.embedding_dim).astype(np.float32)

    vm.store("mem_alpha", emb_a, {
        "content": "first test memory",
        "agent": "test-agent",
        "block_type": "episodic",
        "label": "alpha",
        "priority": 5,
    })
    vm.store("mem_beta", emb_b, {
        "content": "second test memory",
        "agent": "test-agent",
        "block_type": "semantic",
        "label": "beta",
        "priority": 7,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunMaintenance:
    """VecMemory.run_maintenance() — shape and idempotency."""

    def test_run_maintenance_returns_expected_keys(self, tmp_path):
        """run_maintenance() on a 2-memory DB returns the five canonical keys."""
        db = tmp_path / "test.db"
        vm = VecMemory(db, embedding_dim=384)
        _save_two_memories(vm)

        result = vm.run_maintenance()

        assert isinstance(result, dict)
        for key in ("decay", "promoted", "deduped", "links", "errors"):
            assert key in result, f"missing key: {key}"
        assert isinstance(result["decay"], dict)
        assert isinstance(result["promoted"], int)
        assert isinstance(result["deduped"], int)
        assert isinstance(result["links"], int)
        assert isinstance(result["errors"], list)

    def test_run_maintenance_idempotent(self, tmp_path):
        """Calling run_maintenance() twice should not produce errors on the second pass."""
        db = tmp_path / "test_idempotent.db"
        vm = VecMemory(db, embedding_dim=384)
        _save_two_memories(vm)

        first = vm.run_maintenance()
        second = vm.run_maintenance()

        assert second["errors"] == [], (
            f"Second maintenance pass produced errors: {second['errors']}"
        )
        # The second pass should be a no-op for decay (nothing changed)
        assert second["decay"].get("decayed", 0) == 0


class TestConsolidateApplyDecayDelegates:
    """mathir_consolidate.apply_decay delegates to VecMemory.decay_all()."""

    def test_consolidate_apply_decay_delegates_to_vecmemory(self, tmp_path, monkeypatch):
        """apply_decay() imports VecMemory and calls decay_all()."""
        try:
            from mathir_lib.mathir_consolidate import apply_decay
        except ImportError:
            from mathir_consolidate import apply_decay  # type: ignore[no-redef]

        db = tmp_path / "consolidate_test.db"
        vm = VecMemory(db, embedding_dim=384)
        _save_two_memories(vm)

        # apply_decay() imports VecMemory inside its function body via:
        #   from mathir_vec import VecMemory
        # Patch the class at the mathir_vec module level so the re-import
        # inside apply_decay picks up our spy subclass.
        try:
            import mathir_lib.mathir_vec as vec_mod
        except ImportError:
            import mathir_vec as vec_mod  # type: ignore[no-redef]

        original_cls = vec_mod.VecMemory
        calls: list = []

        class SpyVecMemory(original_cls):
            def decay_all(self, *args, **kwargs):
                calls.append(1)
                return super().decay_all(*args, **kwargs)

        monkeypatch.setattr(vec_mod, "VecMemory", SpyVecMemory)

        result = apply_decay(db)

        assert isinstance(result, int)
        assert len(calls) >= 1, "VecMemory.decay_all() was never called by apply_decay()"

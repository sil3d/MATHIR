"""Verifies the v9 module tree is clean.

Checks that all core modules import without error, the legacy ``brain``
namespace is gone, path resolvers agree, and the client block-type allowlist
is correct.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure mathir_lib is importable
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MCP_ROOT = _HERE.parent
if str(_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModuleTree:
    """v9 module-tree health checks."""

    def test_imports_all_core_modules(self):
        """Every core module under mathir_lib imports without error."""
        core_modules = [
            "mathir_lib.mathir_paths",
            "mathir_lib.mathir_mcp_server",
            "mathir_lib.mathir_server",
            "mathir_lib.mathir_stats_server",
            "mathir_lib.mathir_vec",
            "mathir_lib.mathir_search",
            "mathir_lib.mathir_consolidate",
            "mathir_lib.mathir_spread",
            "mathir_lib.mathir_brain",
            "mathir_lib.mathir_watchdog",
            "mathir_lib.mathir_inject_proxy",
        ]
        for mod_name in core_modules:
            try:
                mod = importlib.import_module(mod_name)
            except Exception as exc:
                pytest.fail(f"Failed to import {mod_name}: {exc}")
            assert mod is not None, f"importlib returned None for {mod_name}"

    def test_brain_namespace_removed(self):
        """The legacy top-level 'brain' directory should not be a proper package.

        A bare directory (namespace package without __init__.py) still
        resolves in Python 3 — verify that ``brain`` either does not exist
        OR has no ``__init__.py`` (meaning it is an inert leftover, not a
        real importable package).
        """
        try:
            brain_mod = importlib.import_module("brain")
        except ModuleNotFoundError:
            # Fully removed — best case
            return

        # It resolved (namespace package).  Verify it has no __init__.py.
        brain_path = Path(brain_mod.__path__[0])
        assert not (brain_path / "__init__.py").exists(), (
            f"brain package has __init__.py at {brain_path} — "
            "should have been removed as a top-level namespace"
        )

    def test_paths_unified(self):
        """get_data_dir() and mathir_paths.DATA_DIR resolve to the same path."""
        from mathir_lib import get_data_dir
        from mathir_lib.mathir_paths import DATA_DIR

        resolved_func = get_data_dir()
        resolved_const = DATA_DIR

        assert resolved_func == resolved_const, (
            f"Path mismatch: get_data_dir()={resolved_func} "
            f"!= DATA_DIR={resolved_const}"
        )

    def test_client_block_types(self):
        """_CLIENT_BLOCK_TYPES must include episodic but exclude immunological."""
        from mathir_lib.mathir_mcp_server import _CLIENT_BLOCK_TYPES

        assert "immunological" not in _CLIENT_BLOCK_TYPES, (
            "immunological must NOT be in _CLIENT_BLOCK_TYPES (it is an "
            "internal detection tier, not client-writable)"
        )
        assert "episodic" in _CLIENT_BLOCK_TYPES

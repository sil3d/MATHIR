"""MATHIR - 4-tier cognitive memory library (V7 backward-compat shim).

The real, portable implementation lives in
``D:\\SECRET_PROJECT\\MATHIR\\mathir_mcp\\mathir_lib\\`` (the ``mathir-mcp`` Python
package). This shim provides:

1. The canonical 4-tier taxonomy (single source of truth)
2. Backward-compat re-exports for V7 scripts that do ``from mathir_lib import X``

Tier taxonomy (matches production code in mathir_mcp/mathir_lib/mathir_mcp_server.py):
    working_memory | episodic | semantic | procedural

Note: ``immunological`` is NOT a block_type tier - it is an internal
anomaly-detection capacity slot (see mathir_mcp/mathir_lib/memory_risks.py).
"""
import os
import sys
from pathlib import Path

__version__ = "8.3.0"

# Tier taxonomy - single source of truth
TIERS = ("working_memory", "episodic", "semantic", "procedural")
BLOCK_TYPES = TIERS

# ---------------------------------------------------------------------------
# Portable path helpers (used by both the legacy shim and the mcp package).
# Override defaults via MATHIR_DATA_DIR / MATHIR_CONFIG_DIR env vars.
# ---------------------------------------------------------------------------
_DEFAULT_DATA = Path.home() / ".local" / "share" / "mathir"
_DEFAULT_CONFIG = Path.home() / ".config" / "mathir"


def get_lib_dir() -> Path:
    """Return the install location of the real mathir_lib (the mathir_mcp/ copy)."""
    # If we're the mathir_mcp/mathir_lib/ package (portable install), return this dir.
    here = Path(__file__).parent.resolve()
    if (here / "mathir_vec.py").exists() and (here / "mathir_daemon.py").exists():
        return here
    # Otherwise, walk up to project root and into mathir_mcp/mathir_lib/.
    project_root = here.parent
    candidate = project_root / "mcp" / "mathir_lib"
    if candidate.is_dir():
        return candidate
    return here


def get_data_dir() -> Path:
    """User data dir. Override with MATHIR_DATA_DIR env var."""
    custom = os.environ.get("MATHIR_DATA_DIR")
    if custom:
        return Path(custom).expanduser().resolve()
    _DEFAULT_DATA.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_DATA


def get_config_dir() -> Path:
    """User config dir. Override with MATHIR_CONFIG_DIR env var."""
    custom = os.environ.get("MATHIR_CONFIG_DIR")
    if custom:
        return Path(custom).expanduser().resolve()
    _DEFAULT_CONFIG.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Optional V8 re-exports. The portable mcp package may or may not be on
# sys.path depending on the install method, so we do a best-effort import
# and degrade gracefully.
# ---------------------------------------------------------------------------
_V8_AVAILABLE = False

def _try_import_v8():
    """Best-effort import of the portable mcp package. Adds it to sys.path
    if it lives next to this file (typical source-tree layout)."""
    global _V8_AVAILABLE
    if _V8_AVAILABLE:
        return
    candidate = get_lib_dir()
    if str(candidate.parent) not in sys.path:
        sys.path.insert(0, str(candidate.parent))
    try:
        # Import the mcp subpackage, not this shim
        if "mathir_lib" in sys.modules and sys.modules["mathir_lib"].__file__ == __file__:
            # We're the legacy shim, not the real package - clear it so the
            # real one can be imported as mathir_lib.
            del sys.modules["mathir_lib"]
        # If the real mathir_lib (in mcp/) is importable now, it will replace
        # the shim. This is a development convenience.
    except Exception:
        pass


# Run the import helper at import time so callers can do
# ``from mathir_lib.mathir_vec import VecMemory``.
_try_import_v8()


__all__ = [
    "__version__",
    "TIERS",
    "BLOCK_TYPES",
    "get_lib_dir",
    "get_data_dir",
    "get_config_dir",
    "_V8_AVAILABLE",
]

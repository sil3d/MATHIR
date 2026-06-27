"""MATHIR Memory MCP Server - portable cognitive memory for LLM agents.

This is the canonical, portable Python package (``mathir-mcp``). Install with:

    pip install -e ./mathir_mcp

Then use from anywhere:

    import mathir_lib                          # this package
    from mathir_lib import get_db_path, TIERS
    from mathir_lib.mathir_mcp_server import TOOLS

Or, when importing from outside the mathir_mcp/ directory (e.g. via
``mathir_mcp.mathir_lib.X``), the fully-qualified form works too:

    from mathir_mcp.mathir_lib.mathir_mcp_server import TOOLS
"""
from pathlib import Path
import os

__version__ = "8.4.2"

# Tier taxonomy - SINGLE SOURCE OF TRUTH. Matches the enum in
# mathir_mcp_server.py (memory_save tool schema) and the JSON schema in TOOLS.
#
# MATHIR has 5 tiers total:
#   - 4 user-facing storage tiers that follow the lifecycle promotion chain
#     (working_memory -> episodic -> semantic -> procedural)
#   - 1 detection tier (immunological) for anomaly storage: prompt injections,
#     threat signatures, suspicious patterns. It is first-class: queryable,
#     consolidatable, and linkable, but does NOT participate in the standard
#     promotion chain (it is terminal, like procedural).
TIERS = ("working_memory", "episodic", "semantic", "procedural", "immunological")  # all 5 tiers

# Subset of TIERS that participate in the user-facing promotion chain.
# Use this for: argparse choices, TIER_ORDER in mathir_vec, lifecycle tests.
TIERS_STORAGE = ("working_memory", "episodic", "semantic", "procedural")

# Anomaly-detection tier. Use this for: threat-signature storage, prompt-
# injection quarantine, immune-response pattern matching.
TIERS_DETECTION = ("immunological",)

BLOCK_TYPES = TIERS

# Backward-compat alias - some old V7 code references this name.
# NOTE: this returns the FULL 5-tier list. For the 4-tier promotion chain,
# use TIERS_STORAGE explicitly.
TIER_NAMES = TIERS

__all__ = [
    "__version__",
    "TIERS",
    "TIERS_STORAGE",
    "TIERS_DETECTION",
    "TIER_NAMES",
    "BLOCK_TYPES",
    "get_lib_dir",
    "get_config_dir",
    "get_data_dir",
    "get_db_path",
]

# Portable paths — no hardcoded D:\SECRET_PROJECT
_LIB_DIR = Path(__file__).parent.resolve()


def get_lib_dir() -> Path:
    """Get the mathir_lib install directory (portable)."""
    return _LIB_DIR


def get_config_dir() -> Path:
    """Get user config dir. Override with MATHIR_CONFIG_DIR env var."""
    custom = os.environ.get("MATHIR_CONFIG_DIR")
    if custom:
        return Path(custom).expanduser().resolve()
    try:
        from .mathir_paths import HOME as _HOME
    except ImportError:
        from mathir_paths import HOME as _HOME
    p = _HOME / "config"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_data_dir() -> Path:
    """Get user data dir. Override with MATHIR_DATA_DIR env var.

    Resolves onto the same base as mathir_paths (single resolver) so the
    unified server and this helper agree on where project DBs live."""
    custom = os.environ.get("MATHIR_DATA_DIR")
    if custom:
        return Path(custom).expanduser().resolve()
    try:
        from .mathir_paths import DATA_DIR as _DATA_DIR
    except ImportError:
        from mathir_paths import DATA_DIR as _DATA_DIR
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


def get_db_path(project: str = "default") -> Path:
    """Get DB path for a project. CWD-first, then the unified data dir.

    Discovery order:
      1. ``.mathir/mathir.db`` in the current working directory (per-project isolation)
      2. ``$MATHIR_DATA_DIR/projects/<project>/mathir.db``
      3. ``<MATHIR_HOME>/data/projects/<project>/mathir.db`` (fallback via mathir_paths)
    """
    cwd_db = Path.cwd() / ".mathir" / "mathir.db"
    if cwd_db.exists():
        return cwd_db
    data = get_data_dir() / "projects" / project
    data.mkdir(parents=True, exist_ok=True)
    return data / "mathir.db"
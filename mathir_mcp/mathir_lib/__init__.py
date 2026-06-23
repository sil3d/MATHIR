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

__version__ = "8.4.0"

# Tier taxonomy - SINGLE SOURCE OF TRUTH. Matches the enum in
# mathir_mcp_server.py line ~260 and the JSON schema in TOOLS.
TIERS = ("working_memory", "episodic", "semantic", "procedural")
BLOCK_TYPES = TIERS

# Backward-compat alias - some old V7 code references this name
TIER_NAMES = TIERS

__all__ = [
    "__version__",
    "TIERS",
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
    p = Path.home() / ".config" / "mathir"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_data_dir() -> Path:
    """Get user data dir. Override with MATHIR_DATA_DIR env var."""
    custom = os.environ.get("MATHIR_DATA_DIR")
    if custom:
        return Path(custom).expanduser().resolve()
    p = Path.home() / ".local" / "share" / "mathir"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_db_path(project: str = "default") -> Path:
    """Get DB path for a project. CWD-first, then data dir.

    Discovery order:
      1. ``.mathir/mathir.db`` in the current working directory (per-project isolation)
      2. ``$MATHIR_DATA_DIR/projects/<project>/mathir.db`` (XDG_DATA_HOME)
      3. ``~/.local/share/mathir/projects/<project>/mathir.db`` (fallback)
    """
    cwd_db = Path.cwd() / ".mathir" / "mathir.db"
    if cwd_db.exists():
        return cwd_db
    data = get_data_dir() / "projects" / project
    data.mkdir(parents=True, exist_ok=True)
    return data / "mathir.db"
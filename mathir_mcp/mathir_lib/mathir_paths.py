"""Centralized, portable filesystem paths for MATHIR.

No IDE-specific hardcodes. The base directory resolves as:

  1. ``$MATHIR_HOME`` if set
  2. ``~/.config/mathir`` if it already exists (migrated or fresh install)
  3. ``~/.config/opencode`` if it exists (backward compat for existing data)
  4. ``~/.config/mathir`` (default for fresh installs)

Sub-paths (``config/``, ``data/``, ``logs/``) derive from the resolved base.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_home() -> Path:
    env = os.environ.get("MATHIR_HOME")
    if env:
        return Path(env).expanduser()
    new = Path.home() / ".config" / "mathir"
    legacy = Path.home() / ".config" / "opencode"
    if new.exists():
        return new
    if legacy.exists():
        return legacy
    return new


HOME = _resolve_home()
CONFIG_DIR = HOME / "config"
DATA_DIR = HOME / "data"
LOG_DIR = HOME / "logs"

CONFIG_PATH = CONFIG_DIR / "mathir.json"
PROJECTS_DIR = DATA_DIR / "projects"
LEGACY_DB_PATH = DATA_DIR / "mathir.db"
REGISTRY_PATH = DATA_DIR / "mathir_registry.json"


def ensure_dirs() -> None:
    """Create the config/data/logs directories if missing."""
    for d in (CONFIG_DIR, DATA_DIR, LOG_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

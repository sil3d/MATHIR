"""
MATHIR Drop-in
==============

A minimal, self-contained memory plugin you can copy into any project
and use in five minutes.

The full V7 research codebase (``mathir_lib/``) has 18+ files and many
flags. This package re-implements the canonical 4-tier memory in 7
files, with one well-defined API and one obvious storage format (a
SQLite database file you can ``sqlite3``-inspect).

Quickstart
----------

    from mathir_dropin import MATHIRMemory, configure
    import torch

    memory = MATHIRMemory(embedding_dim=384, db_path="my_memory.db")
    mid = memory.store(torch.randn(1, 384), {"text": "hello world"})
    hits = memory.recall(torch.randn(1, 384), k=3)
    print(memory.get_stats())

Where is my data?
-----------------

By default, in a single SQLite file at the path you passed to
``db_path``. Open it with any SQLite browser
(DB Browser for SQLite, sqlite3 CLI, DataGrip, etc.) and you'll see
one row per memory in the ``memories`` table, with the embedding as a
BLOB, the metadata as JSON, and an FTS5 index for text search.

What if I don't want a file?
----------------------------

Pass ``db_path=None`` (or set ``storage.type = "memory"``) and the
plugin keeps everything in RAM. Faster, no persistence, lost on exit.

What's the difference from ``mathir_lib``?
------------------------------------------

This is a *strict subset* with simpler defaults. The full library has
8 novel memory algorithms, 4 episodic variants, FAISS backends, hybrid
BM25+CE retrieval, and 6 theoretical theorems. The drop-in ships the
core 4-tier model only. For new research or ablations, use
``mathir_lib``. For shipping a feature, use this.
"""

from .config import DEFAULT_CONFIG, configure, get_default_config, validate_config
from .exceptions import (
    DimensionMismatchError,
    MATHIRError,
    MemoryFullError,
    StorageError,
)
from .memory import MATHIRMemory, load, save
from .store import SQLiteStore

# SimpleMemory: FTS5-only, no torch required (new in 7.7.1)
try:
    from .simple import SimpleMemory
except Exception:
    SimpleMemory = None  # type: ignore[assignment]

# UniversalBridge is optional; if the module failed to import (e.g. on
# a deployment that excluded the file) the rest of the package still
# works -- ``MATHIRMemory.universal_recall`` falls back to the union
# of the original ``recall`` and ``recall_text`` methods.
try:  # pragma: no cover - import-time only
    from .universal_bridge import UniversalBridge
except Exception:  # pragma: no cover
    UniversalBridge = None  # type: ignore[assignment]

# Latin name / technical nomenclature handler (new in 7.3.0).
# Optional for the same reason as UniversalBridge: if the module
# is missing the rest of the package imports cleanly.
try:  # pragma: no cover - import-time only
    import mathir_dropin.latin_names as latin_names
except Exception:  # pragma: no cover
    latin_names = None  # type: ignore[assignment]

__version__ = "7.7.1"
__author__ = "MATHIR Research Team"

__all__ = [
    # Main API (full 4-tier, requires torch)
    "MATHIRMemory",
    # Simple API (FTS5-only, no torch) — new in 7.7.1
    "SimpleMemory",
    # Convenience helpers (functional style)
    "save",
    "load",
    # Configuration
    "configure",
    "get_default_config",
    "validate_config",
    "DEFAULT_CONFIG",
    # Storage (exposed for advanced users who want to query the DB directly)
    "SQLiteStore",
    # Cross-provider / cross-lingual bridge (new in 7.2.0)
    "UniversalBridge",
    # Latin / scientific / technical name handling (new in 7.3.0)
    "latin_names",
    # Exceptions
    "MATHIRError",
    "DimensionMismatchError",
    "MemoryFullError",
    "StorageError",
]

#!/usr/bin/env python3
"""One-shot schema migration: legacy MATHIR memories → canonical new schema.

Scans LEGACY_DB_PATH and per-project DBs under PROJECTS_DIR.  Detects legacy
schemas (``modality`` column present, ``content`` column absent) and migrates
them in-place with a safety backup (``<db>.legacy.bak``).

Usage::

    python -m mathir_lib.mathir_migrate --dry-run   # preview
    python -m mathir_lib.mathir_migrate --apply      # execute

Idempotent — running on an already-new DB is a no-op.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

try:
    from .mathir_paths import LEGACY_DB_PATH, PROJECTS_DIR
except ImportError:
    from mathir_paths import LEGACY_DB_PATH, PROJECTS_DIR


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_NEW_COLUMNS = {
    "memory_id", "content", "agent", "block_type", "label", "priority",
    "tier", "project", "created_at", "last_recalled_at", "metadata",
}


def _schema_kind(db_path: Path) -> str:
    """Detect schema flavour for *db_path*.

    Returns ``"new"`` if the ``memories`` table has a ``content`` column,
    ``"legacy"`` if it has ``modality``, or ``"empty"`` if the table does
    not exist at all.
    """
    if not db_path.exists():
        return "empty"
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = {col[1] for col in cursor.fetchall()}
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return "empty"

    if not columns:
        return "empty"
    if "content" in columns:
        return "new"
    if "modality" in columns:
        return "legacy"
    return "unknown"


# ---------------------------------------------------------------------------
# Core migration
# ---------------------------------------------------------------------------

_CREATE_NEW_TABLE = """\
CREATE TABLE IF NOT EXISTS _memories_new (
    memory_id   TEXT PRIMARY KEY,
    content     TEXT,
    agent       TEXT,
    block_type  TEXT,
    label       TEXT,
    priority    INTEGER DEFAULT 5,
    tier        TEXT DEFAULT 'episodic',
    project     TEXT,
    created_at  TEXT,
    last_recalled_at REAL DEFAULT 0,
    metadata    JSON
)
"""

_INSERT_SELECT = """\
INSERT INTO _memories_new
    (memory_id, content, agent, block_type, label, priority,
     tier, project, created_at, last_recalled_at, metadata)
SELECT
    memory_id,
    modality_text                                                        AS content,
    json_extract(metadata, '$.agent')                                    AS agent,
    COALESCE(json_extract(metadata, '$.block_type'), 'episodic')         AS block_type,
    json_extract(metadata, '$.label')                                    AS label,
    COALESCE(json_extract(metadata, '$.priority'), 5)                    AS priority,
    tier,
    json_extract(metadata, '$.project')                                  AS project,
    strftime('%Y-%m-%dT%H:%M:%S', timestamp, 'unixepoch')                AS created_at,
    last_recalled_at,
    json_patch(
        COALESCE(metadata, '{}'),
        json_object('recall_count', COALESCE(recall_count, 0),
                    'stability',    COALESCE(stability, 1.0))
    )                                                                    AS metadata
FROM memories
"""


def migrate_db(db_path: Path, *, dry_run: bool = True) -> dict:
    """Migrate a single legacy DB to the new schema.

    Parameters
    ----------
    db_path:
        Absolute path to the ``mathir.db`` file.
    dry_run:
        If *True* only report what would happen; no writes.

    Returns
    -------
    dict with keys: ``path``, ``status``, ``rows``, ``message``.
    """
    kind = _schema_kind(db_path)
    if kind == "new":
        return {"path": str(db_path), "status": "no-op",
                "rows": 0, "message": "Already new schema — nothing to do."}
    if kind != "legacy":
        return {"path": str(db_path), "status": "skip",
                "rows": 0, "message": f"Unknown/empty schema ({kind}) — skipping."}

    # Count rows before migration.
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        row_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        conn.close()

    if dry_run:
        return {
            "path": str(db_path),
            "status": "would-migrate",
            "rows": row_count,
            "message": (
                f"Legacy schema detected ({row_count} rows). "
                "Embeddings will be re-computed on next save/recall."
            ),
        }

    # --- Actual migration ------------------------------------------------
    bak = db_path.with_suffix(".db.legacy.bak")
    if not bak.exists():
        shutil.copy2(str(db_path), str(bak))

    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_CREATE_NEW_TABLE)
        conn.execute(_INSERT_SELECT)
        conn.execute("DROP TABLE memories")
        conn.execute("ALTER TABLE _memories_new RENAME TO memories")
        conn.commit()
    finally:
        conn.close()

    return {
        "path": str(db_path),
        "status": "migrated",
        "rows": row_count,
        "message": (
            f"Migrated {row_count} rows. Backup at {bak.name}. "
            "Embeddings will be re-computed on next save/recall."
        ),
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_dbs() -> list[Path]:
    """Return all candidate ``mathir.db`` paths to inspect."""
    paths: list[Path] = []

    # 1. Global legacy DB (canonical)
    if LEGACY_DB_PATH.exists():
        paths.append(LEGACY_DB_PATH)

    # 2. Per-project DBs (canonical)
    if PROJECTS_DIR.exists():
        for project_dir in PROJECTS_DIR.iterdir():
            db = project_dir / "mathir.db"
            if db.exists():
                paths.append(db)

    # 3. Deployed daemon's own .mathir/ (MATHIR_HOME/mathir_mcp/.mathir/)
    #    Auto-detected for users who installed via pip install -e (editable).
    deployed = Path.home() / ".config" / "MATHIR" / "mathir_mcp" / ".mathir" / "mathir.db"
    if deployed.exists() and deployed not in paths:
        paths.append(deployed)

    # 4. Wildcard: any *.mathir/mathir.db under common dev areas (best-effort).
    import os as _os
    for root in (Path.home() / "Documents", Path.home() / "Desktop"):
        if not root.exists():
            continue
        # bounded walk (3 levels deep) to avoid huge home-dir scans
        try:
            for dirpath, dirnames, _ in _os.walk(root):
                depth = Path(dirpath).relative_to(root).parts
                if len(depth) > 3:
                    dirnames.clear()
                    continue
                if ".mathir" in dirnames:
                    candidate = Path(dirpath) / ".mathir" / "mathir.db"
                    if candidate.exists() and candidate not in paths:
                        paths.append(candidate)
        except (PermissionError, OSError):
            continue

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy MATHIR schemas to the canonical new schema.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true",
                       help="Preview migration without writing.")
    group.add_argument("--apply", action="store_true",
                       help="Execute the migration (backups created automatically).")
    args = parser.parse_args(argv)

    dry_run = not args.apply
    dbs = discover_dbs()

    if not dbs:
        print("No MATHIR databases found.")
        return 0

    migrated = 0
    for db in sorted(dbs):
        result = migrate_db(db, dry_run=dry_run)
        tag = {
            "no-op":       "[OK]",
            "skip":        "[SKIP]",
            "would-migrate": "[DRY]",
            "migrated":    "[DONE]",
        }.get(result["status"], "[??]")
        print(f"  {tag} {result['path']}  {result['message']}")
        if result["status"] in ("migrated", "would-migrated"):
            migrated += 1

    action = "would be migrated" if dry_run else "migrated"
    print(f"\n{len(dbs)} DB(s) scanned, {migrated} {action}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

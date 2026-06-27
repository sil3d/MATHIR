"""Tests for mathir_lib.mathir_migrate — one-shot schema migration."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mathir_lib.mathir_migrate import migrate_db, _schema_kind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LEGACY_DDL = """\
CREATE TABLE memories (
    memory_id   TEXT PRIMARY KEY,
    modality    TEXT,
    embedding   BLOB,
    embedding_dim INTEGER,
    metadata    TEXT,
    modality_text TEXT,
    timestamp   REAL,
    tier        TEXT,
    stability   REAL DEFAULT 1.0,
    recall_count INTEGER DEFAULT 0,
    provider    TEXT,
    model       TEXT,
    last_recalled_at REAL DEFAULT 0
)
"""


def _make_legacy_db(tmp_path: Path) -> Path:
    """Create a synthetic legacy DB with 3 rows and return its path."""
    db_path = tmp_path / "mathir.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(_LEGACY_DDL)

    rows = [
        (
            "mem-001",
            "text",
            b"\x00" * 16,
            4,
            json.dumps({"agent": "alpha", "block_type": "episodic",
                        "label": "first", "priority": 8, "project": "projA"}),
            "Hello world",
            1700000000.0,
            "episodic",
            0.95,
            12,
            "openai",
            "gpt-4",
            0.0,
        ),
        (
            "mem-002",
            "text",
            b"\x00" * 16,
            4,
            json.dumps({"agent": "beta", "label": "second"}),
            "Second memory",
            1700100000.0,
            "semantic",
            1.0,
            3,
            "anthropic",
            "claude-3",
            1700050000.0,
        ),
        (
            "mem-003",
            "text",
            b"\x00" * 16,
            4,
            json.dumps({}),       # minimal metadata — COALESCE paths
            "Third memory",
            1700200000.0,
            "working_memory",
            0.5,
            0,
            None,
            None,
            0.0,
        ),
    ]
    conn.executemany(
        "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


def _read_new_schema(db_path: Path) -> tuple[list[str], list[dict]]:
    """Return (column_names, rows_as_dicts) for the new memories table."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("PRAGMA table_info(memories)")
    columns = [col[1] for col in cursor.fetchall()]
    rows = [dict(r) for r in conn.execute("SELECT * FROM memories").fetchall()]
    conn.close()
    return columns, rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSchemaDetection:
    def test_legacy_detected(self, tmp_path):
        db = _make_legacy_db(tmp_path)
        assert _schema_kind(db) == "legacy"

    def test_new_detected(self, tmp_path):
        db = tmp_path / "mathir.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""\
            CREATE TABLE memories (
                memory_id TEXT PRIMARY KEY, content TEXT, agent TEXT,
                block_type TEXT, label TEXT, priority INTEGER,
                tier TEXT, project TEXT, created_at TEXT,
                last_recalled_at REAL, metadata JSON
            )
        """)
        conn.close()
        assert _schema_kind(db) == "new"

    def test_missing_db(self, tmp_path):
        assert _schema_kind(tmp_path / "nope.db") == "empty"


class TestMigration:
    def test_backup_created(self, tmp_path):
        db = _make_legacy_db(tmp_path)
        result = migrate_db(db, dry_run=False)
        assert result["status"] == "migrated"
        assert db.with_suffix(".db.legacy.bak").exists()

    def test_row_count_preserved(self, tmp_path):
        db = _make_legacy_db(tmp_path)
        migrate_db(db, dry_run=False)
        _, rows = _read_new_schema(db)
        assert len(rows) == 3

    def test_new_schema_columns_present(self, tmp_path):
        db = _make_legacy_db(tmp_path)
        migrate_db(db, dry_run=False)
        columns, _ = _read_new_schema(db)
        for expected in ("content", "agent", "block_type", "created_at"):
            assert expected in columns, f"Missing column: {expected}"

    def test_content_from_modality_text(self, tmp_path):
        db = _make_legacy_db(tmp_path)
        migrate_db(db, dry_run=False)
        _, rows = _read_new_schema(db)
        by_id = {r["memory_id"]: r for r in rows}
        assert by_id["mem-001"]["content"] == "Hello world"
        assert by_id["mem-002"]["content"] == "Second memory"
        assert by_id["mem-003"]["content"] == "Third memory"

    def test_agent_extracted_from_metadata(self, tmp_path):
        db = _make_legacy_db(tmp_path)
        migrate_db(db, dry_run=False)
        _, rows = _read_new_schema(db)
        by_id = {r["memory_id"]: r for r in rows}
        assert by_id["mem-001"]["agent"] == "alpha"
        assert by_id["mem-002"]["agent"] == "beta"
        # mem-003 had empty metadata — agent should be NULL/empty
        assert by_id["mem-003"]["agent"] in (None, "")

    def test_created_at_is_iso(self, tmp_path):
        db = _make_legacy_db(tmp_path)
        migrate_db(db, dry_run=False)
        _, rows = _read_new_schema(db)
        for row in rows:
            ts = row["created_at"]
            # ISO 8601 format: YYYY-MM-DDTHH:MM:SS
            assert "T" in ts, f"Not ISO: {ts}"

    def test_metadata_contains_recall_count_and_stability(self, tmp_path):
        db = _make_legacy_db(tmp_path)
        migrate_db(db, dry_run=False)
        _, rows = _read_new_schema(db)
        by_id = {r["memory_id"]: r for r in rows}
        meta = json.loads(by_id["mem-001"]["metadata"])
        assert meta["recall_count"] == 12
        assert meta["stability"] == pytest.approx(0.95)
        meta2 = json.loads(by_id["mem-002"]["metadata"])
        assert meta2["recall_count"] == 3

    def test_original_backed_up(self, tmp_path):
        db = _make_legacy_db(tmp_path)
        # Read original content for comparison
        orig_conn = sqlite3.connect(str(db))
        orig_rows = orig_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        orig_conn.close()

        migrate_db(db, dry_run=False)
        bak = db.with_suffix(".db.legacy.bak")
        assert bak.exists()

        # Backup should still have the old schema
        bak_conn = sqlite3.connect(str(bak))
        cols = {c[1] for c in bak_conn.execute("PRAGMA table_info(memories)").fetchall()}
        bak_conn.close()
        assert "modality" in cols, "Backup should have legacy schema"
        assert "content" not in cols

    def test_idempotent_noop(self, tmp_path):
        db = _make_legacy_db(tmp_path)
        migrate_db(db, dry_run=False)
        # Run again — should be no-op
        result = migrate_db(db, dry_run=False)
        assert result["status"] == "no-op"

    def test_dry_run_does_not_modify(self, tmp_path):
        db = _make_legacy_db(tmp_path)
        result = migrate_db(db, dry_run=True)
        assert result["status"] == "would-migrate"
        # Schema should still be legacy
        assert _schema_kind(db) == "legacy"
        # No backup should be created
        assert not db.with_suffix(".db.legacy.bak").exists()


class TestEdgeCases:
    def test_block_type_coalesce_default(self, tmp_path):
        """When metadata has no block_type, it should default to 'episodic'."""
        db = tmp_path / "mathir.db"
        conn = sqlite3.connect(str(db))
        conn.execute(_LEGACY_DDL)
        conn.execute(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("mem-x", "text", b"", 4, json.dumps({"agent": "x"}),
             "content x", 1700000000.0, "episodic", 1.0, 0, None, None, 0.0),
        )
        conn.commit()
        conn.close()

        migrate_db(db, dry_run=False)
        _, rows = _read_new_schema(db)
        assert rows[0]["block_type"] == "episodic"

    def test_priority_coalesce_default(self, tmp_path):
        """When metadata has no priority, it should default to 5."""
        db = tmp_path / "mathir.db"
        conn = sqlite3.connect(str(db))
        conn.execute(_LEGACY_DDL)
        conn.execute(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("mem-y", "text", b"", 4, json.dumps({}),
             "content y", 1700000000.0, "episodic", 1.0, 0, None, None, 0.0),
        )
        conn.commit()
        conn.close()

        migrate_db(db, dry_run=False)
        _, rows = _read_new_schema(db)
        assert rows[0]["priority"] == 5

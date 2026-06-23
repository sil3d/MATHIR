"""
mathir_dropin.py — MATHIR memory for vision_testing.

Tries to import from the mathir_dropin package (SimpleMemory).
Falls back to standalone FTS5 implementation if package not available.

Key findings (v7.7.1):
- FTS5 alone provides good recall for conversational memory
- get_last() is essential for context (always include recent memories)
- DB should NOT be deleted on restart (preserve memories)
- No torch/sentence_transformers needed for basic memory
"""

# Try importing from package first
try:
    from mathir_dropin.simple import SimpleMemory as _SimpleMemory
    _HAS_PACKAGE = True
except ImportError:
    _HAS_PACKAGE = False

# Standalone fallback (if package not available)
import sqlite3
import json
from datetime import datetime
from pathlib import Path


class MATHIRMemory:
    """Simple persistent memory store using SQLite FTS5.
    
    If mathir_dropin package is installed, delegates to SimpleMemory.
    Otherwise uses standalone FTS5 implementation.
    """

    def __init__(self, embedding_dim: int = 384, db_path: str = "memory/vision_test.db"):
        self.db_path = db_path
        if _HAS_PACKAGE:
            self._impl = _SimpleMemory(db_path=db_path)
        else:
            self._impl = None
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        # Main memories table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT,
                provider TEXT,
                model TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # FTS5 index for full-text search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                text, metadata,
                content='memories',
                content_rowid='id'
            )
        """)
        # Triggers to keep FTS in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, text, metadata) VALUES (new.id, new.text, new.metadata);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text, metadata) VALUES('delete', old.id, old.text, old.metadata);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text, metadata) VALUES('delete', old.id, old.text, old.metadata);
                INSERT INTO memories_fts(rowid, text, metadata) VALUES (new.id, new.text, new.metadata);
            END
        """)
        conn.commit()
        conn.close()

    def store(self, embedding=None, metadata: dict = None, provider: str = "fts5", model: str = "text"):
        """Store a memory. Embedding is optional (FTS5 doesn't need it)."""
        text = (metadata or {}).get("text", "")
        if not text:
            return
        if self._impl:
            self._impl.store(text=text, metadata=metadata, provider=provider, model=model)
        else:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO memories (text, metadata, provider, model) VALUES (?, ?, ?, ?)",
                (text, json.dumps(metadata or {}), provider, model)
            )
            conn.commit()
            conn.close()

    def universal_recall(self, query: str, k: int = 5) -> list:
        """Recall memories relevant to query using FTS5 full-text search."""
        if self._impl:
            results = self._impl.recall(query, k=k)
            # Normalize to expected format
            return [{
                "memory_id": r["memory_id"],
                "metadata": {
                    "text": r["text"],
                    "model": r["metadata"].get("model", ""),
                    "timestamp": r["metadata"].get("timestamp", r.get("created_at", "")),
                },
                "score": r["score"],
            } for r in results]
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute("""
                    SELECT m.id, m.text, m.metadata, m.created_at, rank
                    FROM memories_fts fts
                    JOIN memories m ON m.id = fts.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY rank LIMIT ?
                """, (query, k)).fetchall()
            except Exception:
                rows = conn.execute("""
                    SELECT id, text, metadata, created_at, 0 as rank
                    FROM memories WHERE text LIKE ?
                    ORDER BY id DESC LIMIT ?
                """, (f"%{query}%", k)).fetchall()
            conn.close()
            return [{"memory_id": r["id"], "metadata": {"text": r["text"], "model": json.loads(r["metadata"]).get("model","") if r["metadata"] else "", "timestamp": json.loads(r["metadata"]).get("timestamp","") if r["metadata"] else ""}, "score": abs(r["rank"]) if r["rank"] else 0.0} for r in rows]

    def get_last(self, n: int = 3) -> list:
        """Return the last N memories (most recent)."""
        if self._impl:
            results = self._impl.get_last(n=n)
            return [{
                "memory_id": r["memory_id"],
                "metadata": {
                    "text": r["text"],
                    "model": r["metadata"].get("model", ""),
                    "timestamp": r["metadata"].get("timestamp", r.get("created_at", "")),
                },
                "score": 0.0,
            } for r in results]
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, text, metadata, created_at FROM memories ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
            conn.close()
            return [{"memory_id": r["id"], "metadata": {"text": r["text"], "model": json.loads(r["metadata"]).get("model","") if r["metadata"] else "", "timestamp": json.loads(r["metadata"]).get("timestamp","") if r["metadata"] else ""}, "score": 0.0} for r in rows]

    def get_stats(self) -> dict:
        """Return memory statistics."""
        if self._impl:
            return self._impl.get_stats()
        else:
            conn = sqlite3.connect(self.db_path)
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            conn.close()
            return {"total_memories": count}

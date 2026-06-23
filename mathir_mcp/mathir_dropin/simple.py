"""
MATHIR Drop-in — Simple FTS5-only memory (no torch required).

This is a lightweight alternative to the full MATHIRMemory class.
It uses SQLite FTS5 for text search only — no embeddings, no torch,
no sentence_transformers. Works on any Python 3.8+ with zero deps.

Key findings from vision_testing:
- FTS5 alone provides good recall for conversational memory
- get_last() is essential for context (always include recent memories)
- DB should NOT be deleted on restart (preserve memories)
- Simple is better — 143 lines > 1390 lines for most use cases

Usage:
    from mathir_dropin.simple import SimpleMemory
    
    mem = SimpleMemory(db_path="memory.db")
    mem.store(text="The man is wearing a red shirt", metadata={"model": "gemma"})
    results = mem.recall("shirt", k=5)
    last = mem.get_last(n=3)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SimpleMemory:
    """FTS5-only persistent memory. No torch, no embeddings, no deps."""

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT,
                provider TEXT DEFAULT 'fts5',
                model TEXT DEFAULT 'text',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
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

    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None,
              provider: str = "fts5", model: str = "text") -> int:
        """Store a memory. Returns the memory id."""
        if not text:
            return 0
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            "INSERT INTO memories (text, metadata, provider, model) VALUES (?, ?, ?, ?)",
            (text, json.dumps(metadata or {}), provider, model)
        )
        mid = cur.lastrowid
        conn.commit()
        conn.close()
        return mid

    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Recall memories relevant to query using FTS5."""
        if not query:
            return []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("""
                SELECT m.id, m.text, m.metadata, m.created_at, rank
                FROM memories_fts fts
                JOIN memories m ON m.id = fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, k)).fetchall()
        except Exception:
            # Fallback to LIKE for special chars
            rows = conn.execute("""
                SELECT id, text, metadata, created_at, 0 as rank
                FROM memories
                WHERE text LIKE ?
                ORDER BY id DESC
                LIMIT ?
            """, (f"%{query}%", k)).fetchall()
        conn.close()
        return [self._row_to_result(r) for r in rows]

    def get_last(self, n: int = 3) -> List[Dict[str, Any]]:
        """Return the last N memories (most recent first)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, text, metadata, created_at, 0 as rank FROM memories ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
        conn.close()
        return [self._row_to_result(r) for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        return {"total_memories": count}

    def search_context(self, query: str, k: int = 5, last_n: int = 3) -> str:
        """Get context string for LLM injection (recall + last N, deduplicated)."""
        recalled = self.recall(query, k=k)
        last = self.get_last(n=last_n)
        # Deduplicate
        seen = set()
        all_mem = []
        for r in recalled + last:
            mid = r.get("memory_id", 0)
            if mid not in seen:
                seen.add(mid)
                all_mem.append(r)
        # Format
        lines = []
        for r in all_mem[:k]:
            txt = r.get("text", "")
            if txt:
                lines.append(f"- {txt}")
        return "\n".join(lines) if lines else ""

    def _row_to_result(self, row) -> Dict[str, Any]:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return {
            "memory_id": row["id"],
            "text": row["text"],
            "metadata": meta,
            "created_at": row["created_at"],
            "score": abs(row["rank"]) if row["rank"] else 0.0,
        }

    def __repr__(self):
        stats = self.get_stats()
        return f"SimpleMemory(db={self.db_path}, memories={stats['total_memories']})"

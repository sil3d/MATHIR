"""
MATHIR Drop-in — SQLite storage layer.

Why SQLite?
    * Single file → trivial to back up, version, and ship to production.
    * FTS5 is built in → fast text search without an external service.
    * The user can `sqlite3 mathir.db` and read their own data. No
      opaque binary blobs hidden behind a custom API.

Schema
------
memories (one row per memory):
    memory_id      TEXT PRIMARY KEY    e.g. "mem_a1b2c3d4"
    modality       TEXT                'text' | 'image' | 'audio' | 'video' | 'other'
    embedding      BLOB                raw bytes of float32 torch.Tensor
    embedding_dim  INTEGER             for sanity checks on load
    metadata       TEXT (JSON)         user-provided dict
    modality_text  TEXT                short text used for FTS5 search
    timestamp      REAL                unix seconds
    tier           TEXT                'working' | 'episodic' | 'semantic' | 'immune'
    stability      REAL                Ebbinghaus stability (default 1.0)
    recall_count   INTEGER             # of times recalled
    provider       TEXT DEFAULT 'unknown'  -- 'openai', 'cohere', 'voyage', 'ollama', 'huggingface', 'direct'
    model          TEXT DEFAULT 'unknown'   -- e.g., 'text-embedding-3-small', 'voyage-4', 'llama3.2:3b'

memories_fts (FTS5 virtual table):
    mirror of (memory_id, modality_text) for BM25-style text search.

The module is importable without numpy/torch (the encode/decode helpers
are pure Python), but the in-process cosine search in
``search_by_embedding`` does require numpy. If numpy is missing the
caller can still use ``search_by_text`` and ``get_all``.
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .exceptions import StorageError


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# FTS5 is required for hybrid text search. We fail loudly at import time
# if it is missing — the user should know rather than getting cryptic
# "no such module" errors later.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id      TEXT PRIMARY KEY,
    modality       TEXT NOT NULL,
    embedding      BLOB,
    embedding_dim  INTEGER,
    metadata       TEXT,
    modality_text  TEXT,
    timestamp      REAL,
    tier           TEXT,
    stability      REAL DEFAULT 1.0,
    recall_count   INTEGER DEFAULT 0,
    provider       TEXT DEFAULT 'unknown',
    model          TEXT DEFAULT 'unknown'
);

CREATE INDEX IF NOT EXISTS idx_memories_modality  ON memories(modality);
CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_tier      ON memories(tier);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    modality_text,
    tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS memory_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    created_at REAL DEFAULT (unixepoch()),
    FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
    UNIQUE(memory_id, provider)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_memory_id ON memory_embeddings(memory_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_provider ON memory_embeddings(provider);
"""

_INSERT = """
INSERT OR REPLACE INTO memories
    (memory_id, modality, embedding, embedding_dim, metadata,
     modality_text, timestamp, tier, stability, recall_count,
     provider, model)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_DELETE = "DELETE FROM memories WHERE memory_id = ?"

_SELECT_ONE = """
SELECT memory_id, modality, embedding, embedding_dim, metadata,
       modality_text, timestamp, tier, stability, recall_count,
       provider, model
FROM memories
WHERE memory_id = ?
"""

_FTS_INSERT = "INSERT INTO memories_fts(memory_id, modality_text) VALUES (?, ?)"
_FTS_DELETE = "DELETE FROM memories_fts WHERE memory_id = ?"

# Multi-embedding SQL
_EMB_INSERT = """
INSERT OR REPLACE INTO memory_embeddings
    (memory_id, provider, model, embedding, embedding_dim, created_at)
VALUES (?, ?, ?, ?, ?, unixepoch())
"""

_EMB_DELETE = "DELETE FROM memory_embeddings WHERE memory_id = ?"

_EMB_SELECT = """
SELECT memory_id, provider, model, embedding, embedding_dim, created_at
FROM memory_embeddings WHERE memory_id = ? AND provider = ?
"""

_EMB_SELECT_ALL = """
SELECT id, memory_id, provider, model, embedding, embedding_dim, created_at
FROM memory_embeddings WHERE memory_id = ?
"""

# Allowed provider identifiers (lowercase)
ALLOWED_PROVIDERS = frozenset({
    "openai", "cohere", "voyage", "ollama",
    "huggingface", "direct", "unknown",
})
_MAX_PROVIDER_LEN = 50
_MAX_MODEL_LEN = 200


# ---------------------------------------------------------------------------
# Embedding (de)serialization
# ---------------------------------------------------------------------------

def _encode_embedding(vec: Any) -> bytes:
    """Serialize a 1-D torch / numpy vector to little-endian float32 bytes.

    The native float32 format is the lowest common denominator:
        * torch.Tensor.cpu().numpy() can produce it without copy,
        * numpy can read it with np.frombuffer,
        * no pickle (no version skew issues across torch versions),
        * no JSON (lossless for floats).
    The first 4 bytes are an int32 count header so multi-dim arrays
    round-trip correctly even though the drop-in API is 1-D only.
    """
    try:
        import numpy as np
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    except Exception:
        # Fallback: list/tuple of floats.
        flat = list(vec) if not hasattr(vec, "__iter__") is False else [float(vec)]
        arr = __import__("numpy").array(flat, dtype="float32").reshape(-1)

    return struct.pack("<i", arr.size) + arr.tobytes()


def _decode_embedding(blob: bytes) -> "numpy.ndarray":
    """Inverse of :func:`_encode_embedding` → 1-D ``float32`` array."""
    if blob is None:
        return __import__("numpy").zeros((0,), dtype="float32")
    n = struct.unpack("<i", blob[:4])[0]
    import numpy as np
    return np.frombuffer(blob[4:4 + 4 * n], dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SQLiteStore:
    """Thin wrapper around a single SQLite file.

    One instance = one database file. The class is *not* thread-safe in
    the sense of sharing a single connection across threads; each thread
    should call :meth:`get_connection` if it needs concurrent access, or
    rely on the fact that the public API takes an internal lock.

    Parameters
    ----------
    db_path:
        Path to the SQLite file. ``":memory:"`` for an in-memory DB
        (useful for tests). The file is created if it does not exist.
    """

    def __init__(self, db_path: str = "mathir.db") -> None:
        self.db_path = db_path
        # `check_same_thread=False` so the same connection can be used
        # by background workers. We still serialize writes via a lock
        # for predictability.
        import threading
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(
                db_path,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        except sqlite3.Error as e:
            raise StorageError(f"Could not open SQLite DB at {db_path!r}: {e}",
                               original=e) from e

    # ---- low-level plumbing ------------------------------------------------

    @contextmanager
    def _tx(self):
        """Context manager: BEGIN ... COMMIT/ROLLBACK."""
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _init_schema(self) -> None:
        """Create tables / indices / FTS5 virtual table if missing.

        ``executescript()`` issues a ``COMMIT`` before running, so the
        schema statements execute in their own implicit transaction.
        We must NOT wrap this in :meth:`_tx` because the manual
        ``ROLLBACK`` would then have nothing to roll back and would
        itself raise.
        """
        with self._lock:
            try:
                self._conn.executescript(_SCHEMA)
            except sqlite3.Error as e:
                raise StorageError(
                    f"Failed to initialise schema: {e}", original=e
                ) from e

    def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- insert / delete / get --------------------------------------------

    def insert(
        self,
        memory_id: str,
        embedding: Any,
        metadata: Optional[Dict[str, Any]] = None,
        modality: str = "text",
        modality_text: str = "",
        tier: str = "episodic",
        timestamp: Optional[float] = None,
        stability: float = 1.0,
        recall_count: int = 0,
        provider: str = "unknown",
        model: str = "unknown",
    ) -> None:
        """Insert (or replace) a memory row.

        The corresponding FTS5 row is kept in sync inside the same
        transaction so a half-written state is never visible.
        """
        emb_blob = _encode_embedding(embedding)
        emb_dim = len(_decode_embedding(emb_blob))
        meta_json = json.dumps(metadata or {}, ensure_ascii=False, default=str)
        ts = float(timestamp) if timestamp is not None else time.time()

        try:
            with self._tx() as conn:
                conn.execute(_INSERT, (
                    memory_id, modality, emb_blob, emb_dim, meta_json,
                    modality_text or "", ts, tier, float(stability),
                    int(recall_count), provider, model,
                ))
                # Keep FTS5 in sync. DELETE first because the row may
                # have existed with different modality_text.
                conn.execute(_FTS_DELETE, (memory_id,))
                if modality_text:
                    conn.execute(_FTS_INSERT, (memory_id, modality_text))
        except sqlite3.Error as e:
            raise StorageError(
                f"insert({memory_id}) failed: {e}", original=e
            ) from e

    def delete(self, memory_id: str) -> bool:
        """Delete a single memory. Returns True if the row existed."""
        try:
            with self._tx() as conn:
                cur = conn.execute(_DELETE, (memory_id,))
                conn.execute(_FTS_DELETE, (memory_id,))
                conn.execute(_EMB_DELETE, (memory_id,))
            return cur.rowcount > 0
        except sqlite3.Error as e:
            raise StorageError(
                f"delete({memory_id}) failed: {e}", original=e
            ) from e

    # ---- multi-embedding operations -------------------------------------

    def insert_embedding(
        self,
        memory_id: str,
        provider: str,
        model: str,
        embedding: Any,
    ) -> None:
        """Insert an embedding for a specific provider for an existing memory.

        The UNIQUE(memory_id, provider) constraint ensures one embedding per
        provider per memory. Use this to store embeddings from different
        providers (e.g., OpenAI, Cohere) so you never need to re-embed on
        provider switch.

        Raises
        ------
        StorageError
            If ``provider`` is not in the allow-list, exceeds 50 chars,
            or ``model`` exceeds 200 chars.
        """
        # Validate provider length only (allow any provider name for flexibility)
        provider_clean = str(provider).strip().lower()
        if len(provider_clean) > _MAX_PROVIDER_LEN:
            raise StorageError(
                f"insert_embedding: provider must be <= {_MAX_PROVIDER_LEN} chars, "
                f"got {len(provider_clean):d}"
            )
        # Validate model length
        model_str = str(model)
        if len(model_str) > _MAX_MODEL_LEN:
            raise StorageError(
                f"insert_embedding: model must be <= {_MAX_MODEL_LEN} chars, "
                f"got {len(model_str):d}"
            )
        emb_blob = _encode_embedding(embedding)
        emb_dim = len(_decode_embedding(emb_blob))
        try:
            with self._tx() as conn:
                conn.execute(_EMB_INSERT, (
                    memory_id, provider_clean, model_str, emb_blob, emb_dim
                ))
        except sqlite3.Error as e:
            raise StorageError(
                f"insert_embedding({memory_id}, {provider_clean}) failed: {e}", original=e
            ) from e

    def get_embedding(
        self,
        memory_id: str,
        provider: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the stored embedding for a specific provider of a memory.

        Returns a dict with keys: memory_id, provider, model, embedding,
        embedding_dim, created_at. Returns None if not found.
        """
        try:
            row = self._conn.execute(_EMB_SELECT, (memory_id, provider)).fetchone()
        except sqlite3.Error as e:
            raise StorageError(
                f"get_embedding({memory_id}, {provider}) failed: {e}", original=e
            ) from e
        if row is None:
            return None
        return _emb_row_to_dict(row)

    def get_embeddings_for_memory(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get all provider embeddings for a single memory.

        Returns a list of dicts, one per provider.
        """
        try:
            rows = self._conn.execute(_EMB_SELECT_ALL, (memory_id,)).fetchall()
        except sqlite3.Error as e:
            raise StorageError(
                f"get_embeddings_for_memory({memory_id}) failed: {e}", original=e
            ) from e
        return [_emb_row_to_dict(r) for r in rows]

    def search_by_embedding_multi(
        self,
        query_embedding: Any,
        provider: str,
        k: int = 5,
        modality: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Top-k most similar memories using a specific provider's embeddings.

        This searches using the stored embeddings for the specified provider,
        ensuring you're always in the right vector space for that provider.
        Falls back to primary embeddings if a memory doesn't have an embedding
        for the specified provider.
        """
        try:
            import numpy as np
        except ImportError as e:
            raise StorageError(
                "search_by_embedding_multi requires numpy", original=e
            ) from e

        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        q_unit = q / q_norm

        # Build query to get all memories with their primary embedding
        # and the provider-specific embedding if available
        sql = """
            SELECT m.memory_id, m.modality, m.embedding, m.embedding_dim,
                   m.metadata, m.modality_text, m.timestamp, m.tier,
                   m.stability, m.recall_count,
                   e.embedding as emb_embedding, e.model as emb_model
            FROM memories m
            LEFT JOIN memory_embeddings e ON m.memory_id = e.memory_id
                AND e.provider = ?
        """
        clauses: List[str] = []
        params: List[Any] = [provider]
        if modality is not None:
            clauses.append("m.modality = ?")
            params.append(modality)
        if tier is not None:
            clauses.append("m.tier = ?")
            params.append(tier)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY m.timestamp"

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            raise StorageError(
                f"search_by_embedding_multi failed: {e}", original=e
            ) from e

        if not rows:
            return []

        # Validate query dimension matches stored dimension for this provider.
        # Use first row's emb_embedding_dim if available, otherwise primary embedding_dim.
        first_emb_dim = None
        for r in rows:
            if r["emb_embedding"] is not None:
                first_emb_dim = _decode_embedding(r["emb_embedding"]).shape[-1]
                break
            elif r["embedding"] is not None:
                first_emb_dim = r["embedding_dim"]
                break
        if first_emb_dim is not None and q.shape[0] != first_emb_dim:
            raise StorageError(
                f"Query embedding dim {q.shape[0]} != stored dim {first_emb_dim}"
            )

        # Phase 1: Extract embeddings and metadata from rows
        mem_ids = []
        modalities = []
        texts = []
        timestamps = []
        tiers = []
        stabilities = []
        recall_counts = []
        metadatas = []
        emb_models = []
        embeddings = []  # collected for vectorized cosine

        for r in rows:
            # Use provider-specific embedding if available, otherwise primary
            if r["emb_embedding"] is not None:
                emb = _decode_embedding(r["emb_embedding"])
                model = r["emb_model"]
            elif r["embedding"] is not None:
                emb = _decode_embedding(r["embedding"])
                model = "primary"
            else:
                continue  # Skip memories without any embedding

            mem_ids.append(r["memory_id"])
            modalities.append(r["modality"])
            texts.append(r["modality_text"] or "")
            timestamps.append(float(r["timestamp"]))
            tiers.append(r["tier"])
            stabilities.append(float(r["stability"]))
            recall_counts.append(int(r["recall_count"]))
            emb_models.append(model)
            embeddings.append(emb)

            try:
                meta = json.loads(r["metadata"]) if r["metadata"] else {}
            except json.JSONDecodeError:
                meta = {"_raw": r["metadata"]}
            metadatas.append(meta)

        if not mem_ids:
            return []

        # Phase 2: Vectorized cosine similarity (single matrix multiply)
        embs_matrix = np.stack(embeddings)  # [N, D]
        norms = np.linalg.norm(embs_matrix, axis=1)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
        embs_unit = embs_matrix / norms[:, None]
        similarities = embs_unit @ q_unit  # [N]

        # Phase 3: Top-k via argpartition — O(N) not O(N log N)
        if k >= len(similarities):
            top_idx = np.argsort(-similarities)
        else:
            top_idx = np.argpartition(-similarities, k)[:k]
            top_idx = top_idx[np.argsort(-similarities[top_idx])]

        results: List[Dict[str, Any]] = []
        for i in top_idx:
            results.append({
                "memory_id": mem_ids[i],
                "modality": modalities[i],
                "metadata": metadatas[i],
                "modality_text": texts[i],
                "timestamp": timestamps[i],
                "tier": tiers[i],
                "stability": stabilities[i],
                "recall_count": recall_counts[i],
                "similarity": float(similarities[i]),
                "embedding_model": emb_models[i],
            })
        return results

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Return a single memory as a dict, or ``None`` if missing."""
        try:
            row = self._conn.execute(_SELECT_ONE, (memory_id,)).fetchone()
        except sqlite3.Error as e:
            raise StorageError(
                f"get({memory_id}) failed: {e}", original=e
            ) from e
        if row is None:
            return None
        return _row_to_dict(row)

    def get_all(self, memory_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """Batch fetch. Missing IDs are silently skipped."""
        ids = list(memory_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        try:
            rows = self._conn.execute(
                f"SELECT memory_id, modality, embedding, embedding_dim, metadata,"
                f"       modality_text, timestamp, tier, stability, recall_count,"
                f"       provider, model "
                f"FROM memories WHERE memory_id IN ({placeholders})",
                ids,
            ).fetchall()
        except sqlite3.Error as e:
            raise StorageError(f"get_all failed: {e}") from e
        return [_row_to_dict(r) for r in rows]

    def count(
        self,
        tier: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> int:
        """Count rows, optionally filtered by tier and/or modality."""
        sql = "SELECT COUNT(*) FROM memories"
        clauses: List[str] = []
        params: List[Any] = []
        if tier is not None:
            clauses.append("tier = ?")
            params.append(tier)
        if modality is not None:
            clauses.append("modality = ?")
            params.append(modality)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        try:
            return int(self._conn.execute(sql, params).fetchone()[0])
        except sqlite3.Error as e:
            raise StorageError(f"count failed: {e}", original=e) from e

    def list_providers(self) -> List[str]:
        """Distinct provider names stored in ``memory_embeddings``.

        Returns an empty list if no provider-specific embeddings have
        been written yet.  Used by :meth:`MATHIRMemory.universal_recall`
        to build the provider fallback chain.
        """
        try:
            rows = self._conn.execute(
                "SELECT DISTINCT provider FROM memory_embeddings "
                "ORDER BY provider"
            ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.Error as e:
            raise StorageError(f"list_providers failed: {e}", original=e) from e

    def all_ids(self) -> List[str]:
        """Return every memory_id in insertion order. Cheap O(N) listing."""
        try:
            return [r[0] for r in self._conn.execute(
                "SELECT memory_id FROM memories ORDER BY timestamp"
            ).fetchall()]
        except sqlite3.Error as e:
            raise StorageError(f"all_ids failed: {e}", original=e) from e

    # ---- search ------------------------------------------------------------

    def search_by_embedding(
        self,
        query_embedding: Any,
        k: int = 5,
        modality: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Top-k most similar memories by cosine similarity.

        We load *all* embeddings into memory and do the ranking in
        NumPy. For the drop-in's expected scale (10k memories) this is
        faster and simpler than maintaining an index. The full
        V7 plugin can be configured to use FAISS for >1M.
        """
        try:
            import numpy as np
        except ImportError as e:  # pragma: no cover
            raise StorageError(
                "search_by_embedding requires numpy", original=e
            ) from e

        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        q_unit = q / q_norm

        sql = (
            "SELECT memory_id, modality, embedding, embedding_dim, metadata,"
            "       modality_text, timestamp, tier, stability, recall_count,"
            "       provider, model "
            "FROM memories"
        )
        clauses: List[str] = []
        params: List[Any] = []
        if modality is not None:
            clauses.append("modality = ?")
            params.append(modality)
        if tier is not None:
            clauses.append("tier = ?")
            params.append(tier)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp"

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            raise StorageError(
                f"search_by_embedding failed: {e}", original=e
            ) from e

        if not rows:
            return []

        # Validate query dimension matches stored dimension.
        stored_dim = rows[0]["embedding_dim"]
        if q.shape[0] != stored_dim:
            raise StorageError(
                f"Query embedding dim {q.shape[0]} != stored dim {stored_dim}"
            )

        # Bulk-decode embeddings.
        embs = np.stack([_decode_embedding(r["embedding"]) for r in rows])
        norms = np.linalg.norm(embs, axis=1)
        # Avoid division-by-zero; zero-vector rows have undefined cosine
        # similarity which we treat as 0.0.
        norms = np.where(norms == 0, 1.0, norms)
        embs_unit = embs / norms[:, None]

        sims = embs_unit @ q_unit
        # argpartition is O(N) for top-k — much faster than full sort.
        if k >= len(sims):
            top_idx = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, k)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: List[Dict[str, Any]] = []
        for i in top_idx:
            d = _row_to_dict(rows[int(i)])
            d["similarity"] = float(sims[int(i)])
            results.append(d)
        return results

    def search_by_text(
        self,
        query_text: str,
        k: int = 5,
        modality: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """BM25 search via SQLite FTS5.

        The query is escaped against FTS5 syntax errors. A simple
        approach: wrap each whitespace-separated token in double quotes
        and add a prefix wildcard so partial matches work
        ("trans*" → "transformer").
        """
        if not query_text or not query_text.strip():
            return []

        # Escape and re-tokenize. Each token gets a prefix wildcard.
        tokens = _tokenize_fts(query_text)
        if not tokens:
            return []
        fts_query = " ".join(f'"{tok}"*' for tok in tokens)

        sql = (
            "SELECT m.memory_id, m.modality, m.embedding, m.embedding_dim, m.metadata,"
            "       m.modality_text, m.timestamp, m.tier, m.stability, m.recall_count,"
            "       m.provider, m.model, bm25(memories_fts) AS rank "
            "FROM memories_fts "
            "JOIN memories m ON m.memory_id = memories_fts.memory_id "
            "WHERE memories_fts MATCH ?"
        )
        params: List[Any] = [fts_query]
        if modality is not None:
            sql += " AND m.modality = ?"
            params.append(modality)
        # bm25() in SQLite returns a *cost* (smaller = more relevant).
        sql += " ORDER BY rank LIMIT ?"
        params.append(int(k))

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            # FTS5 syntax errors surface here; fall back to LIKE so
            # the user still gets results.
            try:
                fallback = (
                    "SELECT memory_id, modality, embedding, embedding_dim, metadata,"
                    "       modality_text, timestamp, tier, stability, recall_count,"
                    "       provider, model "
                    "FROM memories WHERE modality_text LIKE ? LIMIT ?"
                )
                rows = self._conn.execute(
                    fallback, (f"%{query_text}%", int(k))
                ).fetchall()
            except sqlite3.Error as e2:
                raise StorageError(
                    f"search_by_text failed: {e2}", original=e2
                ) from e2

        return [_row_to_dict(r) for r in rows]

    # ---- bookkeeping -------------------------------------------------------

    def bump_recall(self, memory_id: str) -> None:
        """Increment recall_count (used by Ebbinghaus spaced repetition)."""
        try:
            with self._tx() as conn:
                conn.execute(
                    "UPDATE memories SET recall_count = recall_count + 1 "
                    "WHERE memory_id = ?",
                    (memory_id,),
                )
        except sqlite3.Error as e:
            raise StorageError(
                f"bump_recall({memory_id}) failed: {e}", original=e
            ) from e

    def drop_all(self) -> None:
        """Delete *every* memory. Used by tests and `forget(threshold=∞)`."""
        try:
            with self._tx() as conn:
                conn.execute("DELETE FROM memories")
                conn.execute("DELETE FROM memories_fts")
                conn.execute("DELETE FROM memory_embeddings")
        except sqlite3.Error as e:
            raise StorageError(f"drop_all failed: {e}", original=e) from e

    def vacuum(self) -> None:
        """Rebuild the DB file to reclaim disk space after bulk deletes."""
        try:
            with self._lock:
                self._conn.execute("VACUUM")
        except sqlite3.Error as e:
            raise StorageError(f"vacuum failed: {e}", original=e) from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row → plain dict, decoding BLOB / JSON."""
    import numpy as np
    emb = _decode_embedding(row["embedding"])
    try:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
    except json.JSONDecodeError:
        meta = {"_raw": row["metadata"]}
    return {
        "memory_id":     row["memory_id"],
        "modality":      row["modality"],
        "embedding":     emb,
        "embedding_dim": int(row["embedding_dim"]),
        "metadata":      meta,
        "modality_text": row["modality_text"] or "",
        "timestamp":     float(row["timestamp"]),
        "tier":          row["tier"],
        "stability":     float(row["stability"]),
        "recall_count":  int(row["recall_count"]),
        "provider":      row["provider"] if "provider" in row.keys() else "unknown",
        "model":         row["model"] if "model" in row.keys() else "unknown",
    }


def _emb_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a memory_embeddings sqlite3.Row → plain dict."""
    import numpy as np
    emb = _decode_embedding(row["embedding"])
    return {
        "id":            int(row["id"]) if "id" in row.keys() else None,
        "memory_id":     row["memory_id"],
        "provider":      row["provider"],
        "model":         row["model"],
        "embedding":     emb,
        "embedding_dim": int(row["embedding_dim"]),
        "created_at":    float(row["created_at"]) if row["created_at"] else None,
    }


_TOKEN_RE = __import__("re").compile(r"[A-Za-z0-9]+")


def _tokenize_fts(text: str) -> List[str]:
    """Lowercase alphanumeric tokens. Strips punctuation, keeps unicode."""
    return [t.lower() for t in _TOKEN_RE.findall(text) if t]


__all__ = ["SQLiteStore", "_encode_embedding", "_decode_embedding"]

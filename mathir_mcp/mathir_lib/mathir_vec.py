"""
MATHIR Vec — sqlite-vec accelerated vector storage for MATHIR.
Provides O(log N) approximate nearest neighbor search instead of O(N) brute-force.
"""

import os
import sqlite3
import struct
import stat
import threading
import numpy as np
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


def _tighten_perms(p: Path) -> None:
    """SECURITY: tighten DB file permissions to owner-only (0o600). Best-effort — Windows may not support fully."""
    try:
        os.chmod(str(p), stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except (OSError, AttributeError):
        pass


def _snippet(text: Any, n: int = 100) -> str:
    """Compact a string to a single-line snippet of at most ``n`` chars.

    Used by ``consolidate_all(dry_run=True)`` so each candidate-pair preview
    stays small (the prior dry_run returned 228K+ chars on ~700 memories
    because every pair emitted full content). Collapses internal whitespace
    and appends '...' when truncated.
    """
    if text is None:
        return ""
    s = " ".join(str(text).split())
    if len(s) <= n:
        return s
    return s[:n] + "..."


log = logging.getLogger("mathir-vec")

# Try to import sqlite-vec
try:
    import sqlite_vec
    HAS_VEC = True
    log.info("sqlite-vec available")
except ImportError:
    HAS_VEC = False
    log.warning("sqlite-vec not installed — using brute-force fallback")


def _quantize_int8(vec: np.ndarray) -> np.ndarray:
    """Scalar-quantize a float32 vector to int8 ([-128, 127])."""
    vmax = np.abs(vec).max()
    if vmax == 0:
        return np.zeros(len(vec), dtype=np.int8)
    scale = 127.0 / vmax
    return np.clip(np.round(vec * scale), -128, 127).astype(np.int8)


def _serialize_embedding(vec: np.ndarray) -> str:
    """Quantize float32 → int8 and return hex string for vec_int8(X'...')."""
    q = _quantize_int8(vec)
    return q.tobytes().hex()


def _serialize_embedding_sql(vec: np.ndarray) -> str:
    """Return the SQL expression ``vec_int8(X'<hex>')`` for INSERT."""
    return f"vec_int8(X'{_serialize_embedding(vec)}')"


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    """Deserialize int8 blob back to float32 (unit-range approximation)."""
    return np.frombuffer(blob, dtype=np.int8).astype(np.float32).copy() / 127.0


def _decode_brute_blob(blob: bytes) -> np.ndarray:
    """Decode the length-prefixed float32 blob used by the brute-force fallback.

    Format written by :func:`store` when ``sqlite-vec`` is unavailable::

        <i:count><floats...>     (4 bytes count + count*4 bytes payload)

    Used by ``search``, ``find_duplicates``, ``find_related``, and
    ``build_links_all`` — previously duplicated as a 2-line struct.unpack
    pattern in each one.
    """
    n = struct.unpack("<i", blob[:4])[0]
    return np.frombuffer(blob[4:4 + 4 * n], dtype=np.float32).copy()


class VecMemory:
    """
    SQLite-backed vector memory with sqlite-vec acceleration.
    
    Uses vec0 virtual table for O(log N) approximate nearest neighbor search.
    Falls back to brute-force cosine similarity if sqlite-vec is not available.
    """
    
    # Whitelist of valid embedding dimensions (prevents SQL injection via embedding_dim)
    VALID_DIMS = {64, 128, 256, 384, 512, 768, 1024, 2048, 4096}

    # Tier promotion order — must match mathir_lib.TIERS_STORAGE exactly.
    # This is the 4 user-facing storage tiers (working -> episodic -> semantic -> procedural).
    # 'immunological' (5th) and 'guardrail' (6th) are TERMINAL — they do NOT
    # participate in this promotion chain.
    TIER_ORDER = ("working_memory", "episodic", "semantic", "procedural")
    TERMINAL_TIERS = ("immunological", "guardrail")

    def _schema_kind(self) -> str:
        """Detect which memories-table schema this DB uses.

        Returns ``"new"`` (has ``content`` column) or ``"legacy"`` (has
        ``modality``/``embedding`` BLOB columns). The two branches differ in
        where ``recall_count``/``label``/``priority`` live (column vs metadata
        JSON) — callers must handle both.
        """
        with self._db_lock:
            conn = self._get_conn()
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = {col[1] for col in cursor.fetchall()}
            return "new" if "content" in columns else "legacy"

    def _field_sql(self, field: str, kind: str) -> str:
        """Return a SQL expression that reads ``field`` regardless of schema.

        Centralizes the new-vs-legacy divergence so promote/touch_recall/
        auto_promote_all don't repeat the same PRAGMA dance.
        """
        if kind == "new":
            table = {
                "tier": "tier",
                "recall_count": "COALESCE(json_extract(metadata, '$.recall_count'), 0)",
                "last_recalled_at": "COALESCE(last_recalled_at, 0)",
                "created_at_raw": "created_at",
                "created_at_iso": "created_at",
                # Was hardcoded to the literal SQL "NULL" -- verified live,
                # 2026-07-21: get_decay_candidates() falls back to this field
                # via COALESCE(last_recalled_at, created_at_unix, 0) for
                # memories that were saved but never explicitly recalled
                # (last_recalled_at=0). With this as NULL, that COALESCE
                # collapsed straight to 0, so effective_recall_ts=0 and the
                # "> 0" eligibility filter permanently excluded every
                # never-recalled memory from decay, no matter how old --
                # the exact common case Ebbinghaus decay exists for. SQLite's
                # strftime('%s', ...) accepts both created_at formats found
                # in this table ("YYYY-MM-DD HH:MM:SS" and ISO-with-'T'-and-
                # microseconds), confirmed live against real rows of each.
                "created_at_unix": "CAST(strftime('%s', created_at) AS REAL)",
                "label": "label",
                "priority": "COALESCE(priority, 5)",
            }
        else:
            table = {
                "tier": "tier",
                "recall_count": "COALESCE(recall_count, 0)",
                "last_recalled_at": "COALESCE(last_recalled_at, 0)",
                "created_at_raw": "timestamp",
                "created_at_iso": "datetime(timestamp, 'unixepoch')",
                "created_at_unix": "timestamp",
                "label": "json_extract(metadata, '$.label')",
                "priority": "COALESCE(json_extract(metadata, '$.priority'), 5)",
            }
        if field not in table:
            raise KeyError(f"Unknown field {field!r}")
        return table[field]

    def __init__(self, db_path, embedding_dim: int = 384):
        if not isinstance(embedding_dim, int) or embedding_dim not in self.VALID_DIMS:
            raise ValueError(
                f"embedding_dim must be one of {sorted(self.VALID_DIMS)}, got {embedding_dim!r}"
            )
        # Accept str or Path; coerce to Path so _get_conn can call .parent.mkdir
        self.db_path = Path(db_path) if not isinstance(db_path, Path) else db_path
        self.embedding_dim = embedding_dim
        self._conn = None
        # H1: serialize concurrent DB ops; RLock allows re-entry from nested helpers.
        self._db_lock = threading.RLock()
        self._anomaly_detector = None  # lazy-loaded MahalanobisDetector, see _get_anomaly_detector()
        self._ensure_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get or create SQLite connection with sqlite-vec loaded."""
        if self._conn is None:
            # Ensure parent directory exists before sqlite3.connect (which
            # would otherwise silently fail on Windows or write to a wrong path).
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            # SECURITY: tighten DB file permissions (0o600) on first connect
            _tighten_perms(self.db_path)
            if HAS_VEC:
                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)
                self._conn.enable_load_extension(False)
                log.info(f"sqlite-vec loaded for {self.db_path.name}")
        return self._conn
    
    def _execute_with_retry(self, sql, params=None, retries=3):
        """Execute SQL with retry on database locked."""
        with self._db_lock:
            conn = self._get_conn()
            for attempt in range(retries):
                try:
                    if params:
                        return conn.execute(sql, params)
                    else:
                        return conn.execute(sql)
                except sqlite3.OperationalError as e:
                    if "locked" in str(e) and attempt < retries - 1:
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    raise

    def _commit_with_retry(self, retries=3):
        """Commit with retry on database locked."""
        with self._db_lock:
            conn = self._get_conn()
            for attempt in range(retries):
                try:
                    conn.commit()
                    return
                except sqlite3.OperationalError as e:
                    if "locked" in str(e) and attempt < retries - 1:
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    raise
    
    def _ensure_db(self):
        """Create tables if they don't exist."""
        with self._db_lock:
            conn = self._get_conn()

            # Check if memories table exists and get its schema
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = {col[1]: col[2] for col in cursor.fetchall()}

            if not columns:
                # Create new table with our schema (includes last_recalled_at from the start)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        memory_id TEXT PRIMARY KEY,
                        content TEXT,
                        agent TEXT,
                        block_type TEXT,
                        label TEXT,
                        priority INTEGER DEFAULT 5,
                        tier TEXT DEFAULT 'episodic',
                        project TEXT,
                        created_at TEXT,
                        last_recalled_at REAL DEFAULT 0,
                        stability REAL DEFAULT 1.0,
                        metadata JSON
                    )
                """)
                log.info("Created new memories table (with stability column v8.5.1)")
            else:
                # Use existing schema - we'll adapt our queries.
                # Idempotent migration: add last_recalled_at column if missing (v8.3+).
                if "last_recalled_at" not in columns:
                    try:
                        conn.execute("ALTER TABLE memories ADD COLUMN last_recalled_at REAL DEFAULT 0")
                        log.info("Migrated memories table: added last_recalled_at column")
                    except sqlite3.OperationalError as exc:
                        # Column already exists (race / re-run) — safe to ignore.
                        if "duplicate column name" not in str(exc):
                            raise
                # Idempotent migration v8.5.1: add stability column if missing.
                # Ebbinghaus decay/boost requires a REAL column to survive restarts.
                if "stability" not in columns:
                    try:
                        conn.execute("ALTER TABLE memories ADD COLUMN stability REAL DEFAULT 1.0")
                        log.info("Migrated memories table: added stability column (v8.5.1)")
                    except sqlite3.OperationalError as exc:
                        if "duplicate column name" not in str(exc):
                            raise
                log.info(f"Using existing memories table with columns: {list(columns.keys())}")

            # Vec table for vector search (if sqlite-vec available)
            if HAS_VEC:
                # Auto-detect dimension and dtype from existing vec_memories table
                import re as _re
                _needs_int8_migration = False
                try:
                    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_memories'")
                    row = cursor.fetchone()
                    if row:
                        m_int8 = _re.search(r'INT8\[(\d+)\]', row[0], _re.IGNORECASE)
                        m_float = _re.search(r'FLOAT\[(\d+)\]', row[0])
                        if m_int8:
                            existing_dim = int(m_int8.group(1))
                            if existing_dim != self.embedding_dim:
                                log.warning(f"Auto-detecting dimension: DB has {existing_dim}, requested {self.embedding_dim}. Using DB dimension.")
                                self.embedding_dim = existing_dim
                        elif m_float:
                            existing_dim = int(m_float.group(1))
                            if existing_dim != self.embedding_dim:
                                log.warning(f"Auto-detecting dimension: DB has {existing_dim}, requested {self.embedding_dim}. Using DB dimension.")
                                self.embedding_dim = existing_dim
                            _needs_int8_migration = True
                except (_re.error, sqlite3.OperationalError):
                    pass

                if _needs_int8_migration:
                    log.info("Migrating vec_memories FLOAT → INT8 (4x compression)…")
                    old_rows = conn.execute("SELECT memory_id, embedding FROM vec_memories").fetchall()
                    conn.execute("DROP TABLE vec_memories")
                    conn.execute(f"""
                        CREATE VIRTUAL TABLE vec_memories USING vec0(
                            memory_id TEXT PRIMARY KEY,
                            embedding INT8[{self.embedding_dim}] distance_metric=cosine
                        )
                    """)
                    migrated = 0
                    for r in old_rows:
                        mid = r[0] if isinstance(r, (list, tuple)) else r["memory_id"]
                        blob = r[1] if isinstance(r, (list, tuple)) else r["embedding"]
                        fvec = np.frombuffer(blob, dtype=np.float32).copy()
                        hex_str = _serialize_embedding(fvec)
                        conn.execute(f"INSERT INTO vec_memories(memory_id, embedding) VALUES (?, vec_int8(X'{hex_str}'))", [mid])
                        migrated += 1
                    conn.commit()
                    log.info(f"Migrated {migrated} embeddings to INT8")
                else:
                    conn.execute(f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                            memory_id TEXT PRIMARY KEY,
                            embedding INT8[{self.embedding_dim}] distance_metric=cosine
                        )
                    """)
                log.info(f"vec0 table ready: dim={self.embedding_dim}, dtype=INT8")
            else:
                # Brute-force fallback table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings_brute (
                        memory_id TEXT PRIMARY KEY,
                        embedding BLOB
                    )
                """)

            # Link graph for spreading activation (Phase 3 of MATHIR Brain).
            # Built via cosine > threshold, then traversed BFS-style for recall.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_links (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    created_at REAL DEFAULT (julianday('now')),
                    PRIMARY KEY (source_id, target_id)
                )
            """)
            # Idempotent indexes — speed up BFS in both directions.
            conn.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id)")

            # Persisted state for the immunological-tier anomaly detector
            # (one row per project DB, id is always 1). See mathir_anomaly.py.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state_json TEXT NOT NULL
                )
            """)

            # Per-DB embedding model pin (one row, id=1). Records WHICH
            # embedding model this DB's vectors were created with, so that
            # changing MATHIR's configured default model never silently
            # mixes two incompatible embedding spaces in the same DB --
            # existing DBs keep using their original model forever; only a
            # brand-new DB (no row yet) picks up the current default. See
            # ensure_embedding_model() / get_stored_embedding_model().
            conn.execute("""
                CREATE TABLE IF NOT EXISTS db_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    embedding_model TEXT
                )
            """)

            # Audit log of real operations (save/delete/promote/decay/
            # consolidate/link). FIX (2026-07-02): the /api/memory/audit
            # route and memory_audit MCP tool have existed since early in
            # MATHIR's history, gracefully handling a missing table by
            # returning empty results + a note -- but NOTHING ever created
            # this table or wrote to it, found via a systematic 23-tool
            # smoke test. The tool silently returned empty results for
            # every user, forever. See _log_audit()/get_audit_log().
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    agent TEXT,
                    operation TEXT NOT NULL,
                    memory_id TEXT,
                    details TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_agent ON memory_audit(agent)")

            conn.commit()

    def _log_audit(self, operation: str, memory_id: str = None,
                   agent: str = None, details: Dict[str, Any] = None) -> None:
        """Best-effort audit log write -- never blocks or fails the calling
        operation (a logging failure must not break a real save/delete/
        promote/decay/consolidate/link call)."""
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO memory_audit(timestamp, agent, operation, memory_id, details) "
                "VALUES (?, ?, ?, ?, ?)",
                [time.time(), agent, operation, memory_id,
                 json.dumps(details) if details else None],
            )
            conn.commit()
        except Exception:
            log.debug("audit log write failed (non-fatal)", exc_info=True)

    def get_audit_log(self, agent: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent audit entries, newest first."""
        with self._db_lock:
            conn = self._get_conn()
            if agent:
                rows = conn.execute(
                    "SELECT * FROM memory_audit WHERE agent = ? ORDER BY id DESC LIMIT ?",
                    [agent, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memory_audit ORDER BY id DESC LIMIT ?", [limit],
                ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                if d.get("details"):
                    try:
                        d["details"] = json.loads(d["details"])
                    except (TypeError, ValueError):
                        pass
                out.append(d)
            return out
    
    def store(self, memory_id: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        """Store a memory with its embedding vector."""
        with self._db_lock:
            conn = self._get_conn()

            # Ensure embedding is the right shape and type
            vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if len(vec) != self.embedding_dim:
                raise ValueError(f"Embedding dim {len(vec)} != expected {self.embedding_dim}")

            # Use the canonical schema detection (avoid duplicating PRAGMA table_info)
            if self._schema_kind() == "new":
                # New schema
                conn.execute("""
                    INSERT OR REPLACE INTO memories
                    (memory_id, content, agent, block_type, label, priority, tier, project, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_id,
                    metadata.get("content", ""),
                    metadata.get("agent", ""),
                    metadata.get("block_type", ""),
                    metadata.get("label", ""),
                    metadata.get("priority", 5),
                    metadata.get("tier", "episodic"),
                    metadata.get("project", ""),
                    datetime.now().isoformat(),
                    json.dumps(metadata)
                ))
            else:
                # Existing MATHIR schema — pull recall_count/stability from metadata
                # if the caller supplied them; otherwise fall back to the legacy
                # defaults (1.0 / 0) for backward compatibility.
                try:
                    _stability = float(metadata.get("stability", 1.0) or 1.0)
                except (TypeError, ValueError):
                    _stability = 1.0
                try:
                    _recall = int(metadata.get("recall_count", 0) or 0)
                except (TypeError, ValueError):
                    _recall = 0
                _existing_ts = conn.execute(
                    "SELECT timestamp FROM memories WHERE memory_id=?",
                    (memory_id,),
                ).fetchone()
                _ts = _existing_ts[0] if _existing_ts else datetime.now().timestamp()
                conn.execute("""
                    INSERT OR REPLACE INTO memories
                    (memory_id, modality, embedding, embedding_dim, metadata, modality_text, timestamp, tier, stability, recall_count, provider, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_id,
                    "text",
                    vec.tobytes(),
                    self.embedding_dim,
                    json.dumps(metadata),
                    metadata.get("content", ""),
                    _ts,
                    metadata.get("tier", "episodic"),
                    _stability,
                    _recall,
                    "mathir-vec",
                    "vec"
                ))

            # Store vector (int8 quantized — 4x smaller than float32)
            if HAS_VEC:
                conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", [memory_id])
                hex_str = _serialize_embedding(vec)
                conn.execute(
                    f"INSERT INTO vec_memories(memory_id, embedding) VALUES (?, vec_int8(X'{hex_str}'))",
                    [memory_id]
                )
            else:
                # Brute-force: store as BLOB with length header
                blob = struct.pack("<i", vec.size) + vec.tobytes()
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings_brute(memory_id, embedding) VALUES (?, ?)",
                    [memory_id, blob]
                )

            conn.commit()
            self._log_audit("save", memory_id=memory_id, agent=metadata.get("agent"),
                            details={"block_type": metadata.get("block_type"), "priority": metadata.get("priority")})
            return memory_id

    def _get_anomaly_detector(self, threshold: float, warmup_count: int = 30):
        """Lazily load (or create) this DB's MahalanobisDetector, caching it
        on the instance so the O(dim^3) inverse isn't recomputed from scratch
        on every call within a single daemon process lifetime."""
        try:
            from .mathir_anomaly import MahalanobisDetector
        except ImportError:
            from mathir_anomaly import MahalanobisDetector  # type: ignore[no-redef]

        if self._anomaly_detector is not None:
            return self._anomaly_detector

        with self._db_lock:
            conn = self._get_conn()
            row = conn.execute("SELECT state_json FROM anomaly_state WHERE id = 1").fetchone()

        if row is not None:
            state = json.loads(row[0])
            self._anomaly_detector = MahalanobisDetector.from_dict(state)
        else:
            self._anomaly_detector = MahalanobisDetector(
                dim=self.embedding_dim, threshold=threshold, warmup_count=warmup_count,
            )
        return self._anomaly_detector

    def reset_anomaly_state(self) -> None:
        """Clear this project's anomaly baseline -- both the persisted DB
        row AND the in-memory cached detector on this instance.

        REAL GOTCHA this fixes (found live, 2026-07-02): _get_anomaly_
        detector() caches the MahalanobisDetector on self._anomaly_detector
        to avoid recomputing an O(dim^3) matrix inverse per request. The
        daemon keeps one long-lived VecMemory instance per db_path across
        many requests -- so deleting the anomaly_state DB row alone (e.g.
        via raw SQL) does NOT reset detection for an already-running
        daemon: the stale, already-drifted in-memory detector object is
        still used until the whole daemon process restarts. Call this
        method instead to reset cleanly without a restart.
        """
        with self._db_lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM anomaly_state WHERE id = 1")
            self._commit_with_retry()
        self._anomaly_detector = None

    def _save_anomaly_state(self) -> None:
        if self._anomaly_detector is None:
            return
        state_json = json.dumps(self._anomaly_detector.to_dict())
        with self._db_lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO anomaly_state (id, state_json) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET state_json = excluded.state_json",
                (state_json,),
            )
            self._commit_with_retry()

    def check_and_update_anomaly(
        self, embedding: np.ndarray, threshold: float, warmup_count: int = 30,
    ) -> Dict[str, Any]:
        """Score ``embedding`` against this project's anomaly baseline.

        During warmup (fewer than ``warmup_count`` prior calls), every
        embedding is folded into the baseline and treated as non-anomalous
        — there isn't enough data yet to trust a covariance estimate.

        After warmup: embeddings scoring above ``threshold`` are flagged as
        anomalous and are NOT folded into the baseline (anomalies must not
        pollute what "normal" means). Non-anomalous embeddings update the
        baseline as usual.

        Returns ``{"is_anomaly": bool, "score": float | None, "warmed_up": bool}``.
        """
        detector = self._get_anomaly_detector(threshold=threshold, warmup_count=warmup_count)

        if not detector.is_warmed_up():
            detector.update(embedding)
            self._save_anomaly_state()
            return {"is_anomaly": False, "score": None, "warmed_up": False}

        score = detector.score(embedding)
        is_anomaly = score > threshold
        if not is_anomaly:
            detector.update(embedding)
        self._save_anomaly_state()
        return {"is_anomaly": is_anomaly, "score": score, "warmed_up": True}

    def list_immunological(self, project: Optional[str] = None, k: int = 20) -> List[Dict[str, Any]]:
        """List memories currently flagged in the immunological tier, most recent first."""
        with self._db_lock:
            conn = self._get_conn()
            if project:
                cursor = conn.execute(
                    "SELECT memory_id, content, agent, label, created_at, metadata "
                    "FROM memories WHERE tier = 'immunological' AND project = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (project, k),
                )
            else:
                cursor = conn.execute(
                    "SELECT memory_id, content, agent, label, created_at, metadata "
                    "FROM memories WHERE tier = 'immunological' "
                    "ORDER BY created_at DESC LIMIT ?",
                    (k,),
                )
            rows = cursor.fetchall()

        results = []
        for row in rows:
            meta = {}
            try:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            results.append({
                "memory_id": row["memory_id"],
                "content": (row["content"] or "")[:500],
                "agent": row["agent"],
                "label": row["label"],
                "created_at": row["created_at"],
                "anomaly_score": meta.get("anomaly_score"),
            })
        return results

    def list_guardrails(self, project: Optional[str] = None, k: int = 50) -> List[Dict[str, Any]]:
        """List all guardrail memories for a project, highest priority first."""
        with self._db_lock:
            conn = self._get_conn()
            if project:
                cursor = conn.execute(
                    "SELECT memory_id, content, agent, label, priority, created_at "
                    "FROM memories WHERE tier = 'guardrail' AND project = ? "
                    "ORDER BY priority DESC, created_at DESC LIMIT ?",
                    (project, k),
                )
            else:
                cursor = conn.execute(
                    "SELECT memory_id, content, agent, label, priority, created_at "
                    "FROM memories WHERE tier = 'guardrail' "
                    "ORDER BY priority DESC, created_at DESC LIMIT ?",
                    (k,),
                )
            rows = cursor.fetchall()

        return [
            {
                "memory_id": row["memory_id"],
                "content": (row["content"] or "")[:500],
                "agent": row["agent"],
                "label": row["label"],
                "priority": row["priority"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def count_guardrails(self, project: Optional[str] = None) -> int:
        """Count guardrail memories for limit enforcement."""
        with self._db_lock:
            conn = self._get_conn()
            if project:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM memories WHERE tier = 'guardrail' AND project = ?",
                    (project,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM memories WHERE tier = 'guardrail'"
                ).fetchone()
            return int(row["c"]) if row else 0

    def search(self, query_embedding: np.ndarray, k: int = 5,
               agent_filter: str = None, block_type_filter: str = None,
               include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """Search for similar memories using vector similarity.

        include_embeddings: when True, each result dict also carries an
        ``"embedding"`` key with the memory's raw stored vector (a list of
        floats, length == embedding_dim). Default False to keep the response
        payload small for normal recall. This exists so downstream retrieval
        algorithms (e.g. spectral/drift re-ranking, anomaly conditioning)
        can operate on the actual per-memory vectors instead of collapsing
        everything to the 1-dim similarity score -- a real gap flagged by
        the multi-agent investigation (the /api/memory/* routes previously
        never exposed raw embeddings, forcing those algorithms onto a
        degraded score proxy)."""
        with self._db_lock:
            conn = self._get_conn()

            query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
            if len(query_vec) != self.embedding_dim:
                raise ValueError(f"Query dim {len(query_vec)} != expected {self.embedding_dim}")

            # Use the canonical schema detection (avoid duplicating PRAGMA table_info)
            new_schema = self._schema_kind() == "new"

            # Build WHERE clause for metadata filters
            where_clauses = []
            params = []
            if agent_filter:
                if new_schema:
                    where_clauses.append("m.agent = ?")
                else:
                    where_clauses.append("json_extract(m.metadata, '$.agent') = ?")
                params.append(agent_filter)
            if block_type_filter:
                if new_schema:
                    where_clauses.append("m.block_type = ?")
                else:
                    where_clauses.append("json_extract(m.metadata, '$.block_type') = ?")
                params.append(block_type_filter)

            where_sql = " AND ".join(where_clauses)
            if where_sql:
                where_sql = "AND " + where_sql

            if HAS_VEC:
                # Use sqlite-vec for fast search (int8 quantized query)
                query_hex = _serialize_embedding(query_vec)
                if new_schema:
                    sql = f"""
                        SELECT m.memory_id, m.content, m.agent, m.block_type, m.label,
                               m.priority, m.tier, m.project, m.created_at, v.distance
                        FROM vec_memories v
                        JOIN memories m ON v.memory_id = m.memory_id
                        WHERE v.embedding MATCH vec_int8(X'{query_hex}')
                          AND k = ?
                          {where_sql}
                    """
                else:
                    # Existing MATHIR schema
                    sql = f"""
                        SELECT m.memory_id,
                               json_extract(m.metadata, '$.content') as content,
                               json_extract(m.metadata, '$.agent') as agent,
                               json_extract(m.metadata, '$.block_type') as block_type,
                               json_extract(m.metadata, '$.label') as label,
                               json_extract(m.metadata, '$.priority') as priority,
                               m.tier,
                               json_extract(m.metadata, '$.project') as project,
                               m.timestamp as created_at,
                               v.distance
                        FROM vec_memories v
                        JOIN memories m ON v.memory_id = m.memory_id
                        WHERE v.embedding MATCH vec_int8(X'{query_hex}')
                          AND k = ?
                          {where_sql}
                    """
                # H2: over-fetch k*4 so post-filter still yields ~k results.
                params = [k * 4] + params
            else:
                # Brute-force fallback
                rows = conn.execute("SELECT memory_id, embedding FROM embeddings_brute").fetchall()
                if not rows:
                    return []

                # Decode all embeddings
                embs = []
                ids = []
                for row in rows:
                    blob = row["embedding"]
                    embs.append(_decode_brute_blob(blob))
                    ids.append(row["memory_id"])

                embs = np.stack(embs)
                query_unit = query_vec / np.linalg.norm(query_vec)
                norms = np.linalg.norm(embs, axis=1)
                norms = np.where(norms == 0, 1.0, norms)
                embs_unit = embs / norms[:, None]
                sims = embs_unit @ query_unit

                # H2: over-fetch when filtering so the post-filter pass still
                # has enough candidates to return k matches.
                fetch_k = k * 4 if (agent_filter or block_type_filter) else k
                if fetch_k >= len(sims):
                    top_idx = np.argsort(-sims)
                else:
                    top_idx = np.argpartition(-sims, fetch_k)[:fetch_k]
                    top_idx = top_idx[np.argsort(-sims[top_idx])]

                # Build results with metadata
                results = []
                for idx in top_idx:
                    mid = ids[idx]
                    meta = conn.execute(
                        "SELECT * FROM memories WHERE memory_id = ?", [mid]
                    ).fetchone()
                    if meta:
                        if agent_filter and meta["agent"] != agent_filter:
                            continue
                        if block_type_filter and meta["block_type"] != block_type_filter:
                            continue
                        result = {
                            "memory_id": mid,
                            "content": meta["content"],
                            "agent": meta["agent"],
                            "block_type": meta["block_type"],
                            "label": meta["label"],
                            "priority": meta["priority"],
                            "score": float(sims[idx]),
                            "created_at": meta["created_at"],
                            "project": meta["project"],
                        }
                        if include_embeddings:
                            result["embedding"] = embs[idx].astype(float).tolist()
                        results.append(result)
                        if len(results) >= k:
                            break
                return results

            # Execute sqlite-vec query
            rows = conn.execute(sql, params).fetchall()

            results = []
            for row in rows:
                result = {
                    "memory_id": row["memory_id"],
                    "content": row["content"],
                    "agent": row["agent"],
                    "block_type": row["block_type"],
                    "label": row["label"],
                    "priority": row["priority"],
                    "score": 1.0 - row["distance"],  # Convert distance to similarity
                    "created_at": row["created_at"],
                    "project": row["project"],
                }
                if include_embeddings:
                    try:
                        emb_row = conn.execute(
                            "SELECT embedding FROM vec_memories WHERE memory_id = ?",
                            [row["memory_id"]],
                        ).fetchone()
                        if emb_row and emb_row["embedding"] is not None:
                            result["embedding"] = _deserialize_embedding(emb_row["embedding"]).astype(float).tolist()
                    except Exception:
                        # Best-effort: never fail a search just because the
                        # raw embedding couldn't be re-read for one result.
                        pass
                results.append(result)
                if len(results) >= k:
                    break

            return results
    
    # ────────────────────────────────────────────────────────────────────
    # LINK GRAPH — Spreading Activation (Phase 3 of MATHIR Brain)
    # ────────────────────────────────────────────────────────────────────
    # When you recall "Tauri", the link graph automatically activates related
    # memories: "Rust", "IPC", "desktop app", "Cargo", "axum" — even if they
    # don't have the highest cosine similarity to the query.
    # Inspired by Collins & Loftus (1975) spreading activation theory.
    # Schema: memory_links(source_id, target_id, weight, created_at).
    # Built via cosine similarity > threshold (default 0.7).
    # Traversal: 1-2 hops with per-hop decay (default 0.5).

    def add_link(self, source_id: str, target_id: str, weight: float = 1.0) -> Dict[str, Any]:
        """Create or update a link between two memories.

        Returns:
            {source_id, target_id, weight, created: bool}
            `created` is True if a new row was inserted, False if an existing
            row was updated (INSERT OR REPLACE semantics).
        """
        if not source_id or not target_id:
            raise ValueError("source_id and target_id must be non-empty")
        if source_id == target_id:
            raise ValueError("source_id and target_id must differ (no self-links)")

        with self._db_lock:
            # Detect insert vs update so we can report `created` accurately.
            existing = self._execute_with_retry(
                "SELECT 1 FROM memory_links WHERE source_id = ? AND target_id = ?",
                [source_id, target_id],
            ).fetchone()
            created = existing is None

            self._execute_with_retry(
                "INSERT OR REPLACE INTO memory_links(source_id, target_id, weight, created_at) VALUES (?, ?, ?, ?)",
                [source_id, target_id, float(weight), time.time()],
            )
            self._commit_with_retry()

            return {
                "source_id": source_id,
                "target_id": target_id,
                "weight": float(weight),
                "created": created,
            }

    def get_links(
        self,
        memory_id: str,
        depth: int = 1,
        decay: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """BFS through the link graph from `memory_id`.

        For each hop, the link's weight is multiplied by `decay`. The result is
        the `cumulative_weight` arriving at the target node. Direction is
        symmetric — outgoing AND incoming links are followed (the link graph
        is undirected for traversal purposes, since similarity is symmetric).

        Returns:
            list of {memory_id, distance, cumulative_weight}, ordered by
            cumulative_weight DESC then distance ASC.
        """
        if depth < 1:
            return []
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")

        with self._db_lock:
            conn = self._get_conn()
            visited: Dict[str, Dict[str, Any]] = {}

            # Seed: outgoing AND incoming 1-hop neighbors. Treat the seed itself
            # as distance 0 (not returned), so we never echo back the query node.
            # cum_weight at distance 1 = edge_weight * decay (implicit parent weight = 1.0).
            seed_rows = conn.execute(
                """
                SELECT target_id AS mid, weight FROM memory_links WHERE source_id = ?
                UNION ALL
                SELECT source_id AS mid, weight FROM memory_links WHERE target_id = ?
                """,
                [memory_id, memory_id],
            ).fetchall()

            for row in seed_rows:
                mid = row["mid"]
                if mid == memory_id:
                    continue
                if mid not in visited:
                    visited[mid] = {
                        "memory_id": mid,
                        "distance": 1,
                        "cumulative_weight": float(row["weight"]) * decay,
                    }

            # BFS for deeper hops. frontier tracks nodes discovered at the
            # previous distance so we don't expand the whole graph in one query.
            # H4: carry the running product from the parent rather than only
            # the final edge weight — cum_weight[neighbor] =
            # cum_weight[parent] * edge_weight * decay.
            frontier = list(visited.keys())
            for d in range(2, depth + 1):
                if not frontier:
                    break
                frontier_set = set(frontier)
                placeholders = ",".join("?" for _ in frontier)
                next_rows = conn.execute(
                    f"""
                    SELECT source_id AS a, target_id AS b, weight FROM memory_links
                    WHERE source_id IN ({placeholders})
                       OR target_id IN ({placeholders})
                    """,
                    frontier + frontier,
                ).fetchall()
                next_frontier: List[str] = []
                for row in next_rows:
                    a, b, w = row["a"], row["b"], float(row["weight"])
                    # Identify which endpoint is the parent (in frontier) vs.
                    # the neighbor to discover.
                    if a in frontier_set:
                        parent, neighbor = a, b
                    elif b in frontier_set:
                        parent, neighbor = b, a
                    else:
                        continue
                    if neighbor == memory_id or neighbor in visited:
                        continue
                    # Parent is always in visited (frontier nodes are). The seed
                    # itself acts as cum_weight=1.0 for distance-1 hops.
                    parent_cum = (
                        visited[parent]["cumulative_weight"]
                        if parent in visited
                        else 1.0
                    )
                    visited[neighbor] = {
                        "memory_id": neighbor,
                        "distance": d,
                        "cumulative_weight": parent_cum * w * decay,
                    }
                    next_frontier.append(neighbor)
                frontier = next_frontier

            # Stable ordering: highest weight first, then shortest path, then id.
            results = list(visited.values())
            results.sort(key=lambda r: (-r["cumulative_weight"], r["distance"], r["memory_id"]))
            return results

    def build_links_all(self, threshold: float = 0.88, limit: int = 1000,
                        top_k: int = 8) -> Dict[str, int]:
        """Build the link graph for all stored memories.

        For every pair (A, B) with cosine > threshold, two directed links are
        created (A→B and B→A) so BFS traversal works in both directions.

        FIXES (2026-08-18):
        - Archived memories and god:reg registration rows are EXCLUDED. Before
          this, the graph was built from every stored row including 210
          archived memories and hundreds of near-duplicate god:reg entries
          (cos~1.0), which flooded the graph with meaningless edges.
        - Per-memory top-K cap. This model's cosine scores run hot (a nonsense
          query scored 0.839 against an unrelated memory), so even at 0.88 the
          raw graph measured ~87% of all possible pairs on the live DB
          (264,056 links / 781 memories) -- an almost-complete graph that
          carries no "these are actually related" signal. Keeping only the
          top-K neighbours per memory bounds the graph at ~N*top_k directed
          edges while preserving the strongest relations.

        Args:
            threshold: minimum cosine similarity to create a link. 0.88 sits
                just below the near-duplicate threshold (0.95 in
                consolidate_all) so links capture "related" without capturing
                "basically the same content".
            limit: maximum number of memories to scan (default 1000). Use this
                to keep one-shot runs bounded; memory grows O(N) so 1000 is a
                reasonable cap.
            top_k: max neighbours kept per memory (default 8).

        Returns:
            {links_created: N, memories_scanned: M}
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        with self._db_lock:
            conn = self._get_conn()

            # Pull up to `limit` memories with their embedding vectors.
            # Skip archived rows and god:reg registration noise (see class
            # docstring note) so the graph only reflects live knowledge.
            if HAS_VEC:
                rows = conn.execute(
                    """
                    SELECT m.memory_id, v.embedding
                    FROM vec_memories v
                    JOIN memories m ON v.memory_id = m.memory_id
                    WHERE m.tier != 'archived' AND m.label NOT LIKE 'god:reg:%'
                    LIMIT ?
                    """,
                    [limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT m.memory_id, e.embedding
                    FROM embeddings_brute e
                    JOIN memories m ON e.memory_id = m.memory_id
                    WHERE m.tier != 'archived' AND m.label NOT LIKE 'god:reg:%'
                    LIMIT ?
                    """,
                    [limit],
                ).fetchall()

            if not rows:
                return {"links_created": 0, "memories_scanned": 0}

            ids: List[str] = []
            vecs: List[np.ndarray] = []
            for row in rows:
                mid = row["memory_id"]
                blob = row["embedding"]
                if HAS_VEC:
                    vec = _deserialize_embedding(blob)
                else:
                    vec = _decode_brute_blob(blob)
                ids.append(mid)
                vecs.append(vec)

            if not vecs:
                return {"links_created": 0, "memories_scanned": 0}

            # Matrix cosine: M[i, j] = cosine(vecs[i], vecs[j])
            # Normalise rows, then dot product = cosine similarity.
            stack = np.stack(vecs).astype(np.float32)
            norms = np.linalg.norm(stack, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            unit = stack / norms[:, None]
            sim = unit @ unit.T  # shape (N, N)

            # Build directed edges: for each memory keep only its top-`top_k`
            # neighbours above `threshold`. Without the per-memory cap this
            # model's hot cosine scores produce an almost-complete graph
            # (~87% of all pairs at 0.88 on the live DB, 2026-08-18) that
            # carries no "actually related" signal. Top-K per row bounds the
            # graph at N * top_k candidate pairs.
            np.fill_diagonal(sim, -1.0)
            edges: List[tuple] = []
            for i in range(sim.shape[0]):
                row = sim[i]
                cand = np.where(row >= threshold)[0]
                if len(cand) > top_k:
                    cand = cand[np.argsort(-row[cand])[:top_k]]
                for j in cand:
                    edges.append((ids[i], ids[j], float(row[j])))

            # Dedup just in case (cosine matrix is symmetric so each pair appears
            # twice with (i,j) and (j,i)). We write both directions deliberately,
            # so leave them — but Dedupe within a batch to avoid hitting the same
            # PK twice when N is small and we re-build.
            seen_pairs: set = set()
            writes: List[tuple] = []
            for src, tgt, w in edges:
                key = (min(src, tgt), max(src, tgt))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                writes.append((src, tgt, w))
                writes.append((tgt, src, w))

            if writes:
                conn = self._get_conn()
                for attempt in range(3):
                    try:
                        conn.executemany(
                            "INSERT OR REPLACE INTO memory_links(source_id, target_id, weight, created_at) VALUES (?, ?, ?, ?)",
                            [(s, t, w, time.time()) for s, t, w in writes],
                        )
                        conn.commit()
                        break
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e) and attempt < 2:
                            time.sleep(0.1 * (attempt + 1))
                            continue
                        raise

            return {
                "links_created": len(writes),
                "memories_scanned": len(ids),
            }

    def build_entity_links_all(self, limit: int = 1000,
                               entity_edge_weight: float = 0.9) -> Dict[str, Any]:
        """Build an ENTITY-linked graph: connect memories that mention the
        same named entity, regardless of embedding similarity.

        This complements build_links_all() (the cosine-similarity graph).
        The similarity graph links memories on the SAME topic; this entity
        graph links memories that share a specific entity even when they're
        about different topics -- the exact "bridge" relation multi-hop
        retrieval needs and that cosine graphs structurally miss (proven on
        HotpotQA, see benchmarks/10_multihop/). Edges go into the same
        memory_links table, so get_links / PPR-LTE / spreading-activation
        traverse them transparently.

        entity_edge_weight defaults high (0.9) because a shared named entity
        is a strong, precise relation signal -- stronger than a marginal
        cosine-above-threshold link. Weight scales up with the number of
        shared entities between a pair (more shared entities = stronger
        relation), capped at 1.0.

        Returns {links_created, memories_scanned, entities_indexed}.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")

        try:
            from .mathir_entity_graph import extract_entities
        except ImportError:
            from mathir_entity_graph import extract_entities  # type: ignore

        with self._db_lock:
            conn = self._get_conn()
            kind = self._schema_kind()
            if kind == "new":
                rows = conn.execute(
                    "SELECT memory_id, content FROM memories LIMIT ?", [limit]
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT memory_id, json_extract(metadata, '$.content') AS content "
                    "FROM memories LIMIT ?", [limit]
                ).fetchall()

            if not rows:
                return {"links_created": 0, "memories_scanned": 0, "entities_indexed": 0}

            # Inverted index: entity -> set of memory_ids that mention it.
            entity_to_mems: Dict[str, set] = {}
            scanned = 0
            for row in rows:
                mid = row["memory_id"]
                content = row["content"] or ""
                scanned += 1
                for ent in extract_entities(content):
                    entity_to_mems.setdefault(ent, set()).add(mid)

            # Count shared entities per unordered memory pair.
            pair_shared: Dict[tuple, int] = {}
            for ent, mems in entity_to_mems.items():
                if len(mems) < 2:
                    continue  # entity mentioned by only one memory -> no edge
                mem_list = sorted(mems)
                for i in range(len(mem_list)):
                    for j in range(i + 1, len(mem_list)):
                        key = (mem_list[i], mem_list[j])
                        pair_shared[key] = pair_shared.get(key, 0) + 1

            # Build bidirectional edges; weight scales with #shared entities.
            writes: List[tuple] = []
            for (a, b), n_shared in pair_shared.items():
                w = min(1.0, entity_edge_weight + 0.05 * (n_shared - 1))
                writes.append((a, b, w))
                writes.append((b, a, w))

            if writes:
                for attempt in range(3):
                    try:
                        conn.executemany(
                            "INSERT OR REPLACE INTO memory_links(source_id, target_id, weight, created_at) VALUES (?, ?, ?, ?)",
                            [(s, t, w, time.time()) for s, t, w in writes],
                        )
                        conn.commit()
                        break
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e) and attempt < 2:
                            time.sleep(0.1 * (attempt + 1))
                            continue
                        raise

            return {
                "links_created": len(writes),
                "memories_scanned": scanned,
                "entities_indexed": len(entity_to_mems),
            }

    def find_related(
        self,
        memory_id: str,
        max_hops: int = 2,
        min_weight: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Combine vector similarity with the link graph.

        Strategy:
          1. Vector search (k=10) anchored on the query memory's own embedding.
          2. BFS through the link graph up to `max_hops`.
          3. Merge the two result sets. The `source` field tags each row as
             "vector", "link", or "both" so downstream code can decide which
             channel a hit came from. Combined hits float to the top.

        Returns:
            Ranked list of {memory_id, score, distance, source} sorted by
            score DESC then distance ASC.
        """
        if max_hops < 0:
            raise ValueError("max_hops must be >= 0")

        with self._db_lock:
            conn = self._get_conn()

            # Pull the seed embedding. If the memory has no embedding, we can
            # still traverse the graph — vector hits will just be empty.
            seed_vec: Optional[np.ndarray] = None
            if HAS_VEC:
                row = conn.execute(
                    "SELECT embedding FROM vec_memories WHERE memory_id = ?",
                    [memory_id],
                ).fetchone()
                if row:
                    seed_vec = _deserialize_embedding(row["embedding"])
            else:
                row = conn.execute(
                    "SELECT embedding FROM embeddings_brute WHERE memory_id = ?",
                    [memory_id],
                ).fetchone()
                if row:
                    seed_vec = _decode_brute_blob(row["embedding"])

            # 1. Vector channel — k=10 neighbours of the seed embedding.
            vector_hits: Dict[str, Dict[str, Any]] = {}
            if seed_vec is not None:
                try:
                    vec_results = self.search(seed_vec, k=10)
                except Exception:
                    vec_results = []
                for r in vec_results:
                    mid = r["memory_id"]
                    if mid == memory_id:
                        continue
                    vector_hits[mid] = {
                        "memory_id": mid,
                        "score": float(r.get("score", 0.0)),
                        "distance": 0,
                        "source": "vector",
                    }

            # 2. Graph channel — BFS up to max_hops with decay=0.5 (spec default).
            link_results: List[Dict[str, Any]] = []
            if max_hops >= 1:
                link_results = self.get_links(memory_id, depth=max_hops, decay=0.5)

            # 3. Merge: graph weights are already decayed per-hop; treat the
            # cumulative weight as the score for the graph channel.
            merged: Dict[str, Dict[str, Any]] = dict(vector_hits)
            for r in link_results:
                mid = r["memory_id"]
                score = r["cumulative_weight"]
                if score < min_weight:
                    continue
                if mid in merged:
                    merged[mid]["source"] = "both"
                    # Prefer the higher score; keep the smaller distance.
                    if score > merged[mid]["score"]:
                        merged[mid]["score"] = score
                    merged[mid]["distance"] = min(merged[mid]["distance"], r["distance"])
                else:
                    merged[mid] = {
                        "memory_id": mid,
                        "score": score,
                        "distance": r["distance"],
                        "source": "link",
                    }

            results = list(merged.values())
            results.sort(key=lambda r: (-r["score"], r["distance"], r["memory_id"]))
            return results

    def count(self) -> int:
        """Get total number of memories."""
        with self._db_lock:
            conn = self._get_conn()
            row = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
            return row["cnt"]

    def delete(self, memory_id: str) -> bool:
        """Soft-delete a memory: sets tier='archived' and removes it from
        the vector index (so it no longer surfaces in search), but keeps
        the row for audit/recovery -- consistent with the rest of MATHIR's
        lifecycle system (decay_all/consolidate_pair both archive rather
        than destroy).

        FIX (2026-07-02): this previously performed a hard, irreversible
        DELETE FROM memories, contradicting the memory_delete MCP tool's
        own documented contract ("Soft-delete a memory (sets tier to
        archived)") -- found via a systematic 23-tool smoke test. Any
        caller trusting the tool description believed deletions were
        recoverable when they were not.
        """
        with self._db_lock:
            conn = self._get_conn()
            row = conn.execute("SELECT 1 FROM memories WHERE memory_id = ?", [memory_id]).fetchone()
            if not row:
                return False
            conn.execute("UPDATE memories SET tier = 'archived' WHERE memory_id = ?", [memory_id])
            if HAS_VEC:
                conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", [memory_id])
            else:
                conn.execute("DELETE FROM embeddings_brute WHERE memory_id = ?", [memory_id])
            conn.commit()
            self._log_audit("delete", memory_id=memory_id)
            return True
    
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics for this project."""
        with self._db_lock:
            conn = self._get_conn()

            # Total memories
            total_row = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
            total = total_row["cnt"] if total_row else 0

            # Count by block_type (extracted from JSON metadata)
            by_block_type = {}
            try:
                rows = conn.execute(
                    "SELECT json_extract(metadata, '$.block_type') as bt, COUNT(*) as cnt "
                    "FROM memories GROUP BY bt"
                ).fetchall()
                for row in rows:
                    key = row["bt"] if row["bt"] else "(none)"
                    by_block_type[key] = row["cnt"]
            except (sqlite3.OperationalError, KeyError, ValueError):
                pass

            # Count by agent (extracted from JSON metadata)
            by_agent = {}
            try:
                rows = conn.execute(
                    "SELECT json_extract(metadata, '$.agent') as ag, COUNT(*) as cnt "
                    "FROM memories GROUP BY ag ORDER BY cnt DESC"
                ).fetchall()
                for row in rows:
                    key = row["ag"] if row["ag"] else "(none)"
                    by_agent[key] = row["cnt"]
            except (sqlite3.OperationalError, KeyError, ValueError):
                pass

            # Count by project (extracted from JSON metadata)
            by_project = {}
            try:
                rows = conn.execute(
                    "SELECT json_extract(metadata, '$.project') as pj, COUNT(*) as cnt "
                    "FROM memories GROUP BY pj ORDER BY cnt DESC"
                ).fetchall()
                for row in rows:
                    key = row["pj"] if row["pj"] else "(none)"
                    by_project[key] = row["cnt"]
            except (sqlite3.OperationalError, KeyError, ValueError):
                pass

            # DB file size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "total": total,
                "by_block_type": by_block_type,
                "by_agent": by_agent,
                "by_project": by_project,
                "db_size_bytes": db_size,
                "db_path": str(self.db_path),
                "embedding_dim": self.embedding_dim,
                "has_vec": HAS_VEC,
            }
    
    def touch_recall(self, memory_id: str) -> Dict[str, Any]:
        """Increment recall_count, stamp last_recalled_at, AND boost stability.

        Ebbinghaus spaced-repetition (BRAIN_ARCHITECTURE.md lines 63-65):
        - recall_count += 1
        - last_recalled_at = now()
        - stability += 0.1 (capped at 1.0)

        Works on both schemas -- the ``stability`` column exists on both
        (added to the new schema by the idempotent migration in
        _ensure_db), so both branches boost it:
        - legacy: writes to the real ``recall_count``, ``last_recalled_at``,
          and ``stability`` columns (stability boost inlined for atomicity)
        - new:    writes ``recall_count`` into the metadata JSON; stability
          is boosted via the top-level ``stability`` column, same as legacy

        FIX (2026-07-02): previously the "new" schema branch never touched
        ``stability`` at all ("no stability column in this schema" -- which
        was true before v8.5.1's migration, but the column has existed
        since then). This meant decay_all() (which DOES operate on
        ``stability`` for both schemas) could only ever decrease it: a
        memory recalled 20 times decayed identically to one never recalled
        -- the entire reinforce-on-use half of the lifecycle design was
        dead on the "new" schema, which every real MATHIR project uses
        (confirmed live on locomo_conv_2 and MATHIR's own project DB).
        Caught and proven with a direct before/after stability comparison,
        not just code inspection.

        Returns ``{memory_id, found, recall_count, last_recalled_at,
        old_stability, new_stability, schema}``. If the memory doesn't exist,
        returns ``{memory_id, found: False}`` and does not raise.
        """
        with self._db_lock:
            conn = self._get_conn()
            kind = self._schema_kind()
            now = datetime.now().timestamp()

            # Existence check first — don't silently bump a phantom row.
            row = conn.execute(
                "SELECT memory_id, stability FROM memories WHERE memory_id = ?", [memory_id]
            ).fetchone()
            if not row:
                log.debug("touch_recall: memory_id %s not found", memory_id)
                return {"memory_id": memory_id, "found": False}

            old_stability: Optional[float] = None
            new_stability: Optional[float] = None

            if kind == "new":
                # H21: atomic json_set avoids the read-modify-write race on the
                # metadata JSON. The COALESCE handles rows where recall_count
                # was never set. last_recalled_at is also stamped in the same
                # UPDATE so the whole bump is one transaction. stability is
                # boosted the same way as legacy (see FIX note above).
                old_stability = float(row["stability"]) if row["stability"] is not None else 1.0
                new_stability = min(1.0, old_stability + 0.1)
                conn.execute(
                    "UPDATE memories SET "
                    "metadata = json_set(metadata, '$.recall_count', "
                    "COALESCE(json_extract(metadata, '$.recall_count'), 0) + 1), "
                    "last_recalled_at = ?, stability = ? WHERE memory_id = ?",
                    [now, new_stability, memory_id],
                )
            else:
                # Legacy schema: read stability first so we can return the
                # before/after pair. Boost = stability + 0.1, capped at 1.0.
                srow = conn.execute(
                    "SELECT stability FROM memories WHERE memory_id = ?", [memory_id]
                ).fetchone()
                old_stability = (
                    float(srow["stability"]) if (srow and srow["stability"] is not None) else 1.0
                )
                new_stability = min(1.0, old_stability + 0.1)

                conn.execute(
                    """
                    UPDATE memories
                    SET recall_count = COALESCE(recall_count, 0) + 1,
                        last_recalled_at = ?,
                        stability = ?
                    WHERE memory_id = ?
                    """,
                    [now, new_stability, memory_id],
                )

            conn.commit()
            new_count = conn.execute(
                "SELECT " + self._field_sql("recall_count", kind) + " AS c FROM memories WHERE memory_id = ?",
                [memory_id],
            ).fetchone()
            return {
                "memory_id": memory_id,
                "found": True,
                "recall_count": int(new_count["c"]) if new_count else 0,
                "last_recalled_at": now,
                "schema": kind,
                "old_stability": old_stability,
                "new_stability": new_stability,
            }

    # ────────────────────────────────────────────────────────────────────
    # DECAY / FORGETTING (Phase 4 of MATHIR Brain — Consolidation)
    # ────────────────────────────────────────────────────────────────────
    # Ebbinghaus-inspired spaced-repetition forgetting curve. Per
    # BRAIN_ARCHITECTURE.md lines 63-65:
    #   - Decay: 5%/month if no access
    #   - Boost: stability += 0.1 per recall
    #   - Archive: stability < 0.05
    # These three methods (boost_on_recall, get_decay_candidates, decay_all)
    # compose into a nightly consolidation pass: scan for cold memories,
    # apply linear decay, archive dead ones. They only operate on the
    # legacy schema (which has the stability REAL column); on the new
    # schema they are no-ops with `skipped=True` returned to the caller.

    def boost_on_recall(self, memory_id: str) -> Dict[str, Any]:
        """Boost a memory's stability on recall (Ebbinghaus spaced-repetition).

        Per BRAIN_ARCHITECTURE.md line 64: "Boost frequently-accessed memories
        (stability += 0.1 per access)".

        For legacy schema:
          - stability = min(1.0, stability + 0.1)
          - last_recalled_at = now()

        For new schema (no stability column): no-op, returns
        ``{skipped: True, reason: "no_stability_column"}``.

        Returns:
            {memory_id, found, schema, old_stability, new_stability, last_recalled_at, skipped?}
            If the memory doesn't exist, returns ``{memory_id, found: False}``.
        """
        with self._db_lock:
            conn = self._get_conn()
            kind = self._schema_kind()

            # Existence check first — don't silently bump a phantom row.
            row = conn.execute(
                "SELECT memory_id FROM memories WHERE memory_id = ?", [memory_id]
            ).fetchone()
            if not row:
                log.debug("boost_on_recall: memory_id %s not found", memory_id)
                return {"memory_id": memory_id, "found": False, "schema": kind}

            if kind != "legacy":
                log.debug("boost_on_recall: skipped (no stability column in new schema)")
                return {
                    "memory_id": memory_id,
                    "found": True,
                    "schema": kind,
                    "skipped": True,
                    "reason": "no_stability_column",
                    "old_stability": None,
                    "new_stability": None,
                }

            # Legacy: read current stability, compute new value, UPDATE.
            srow = conn.execute(
                "SELECT stability FROM memories WHERE memory_id = ?", [memory_id]
            ).fetchone()
            old_stability = (
                float(srow["stability"]) if (srow and srow["stability"] is not None) else 1.0
            )
            new_stability = min(1.0, old_stability + 0.1)
            now = datetime.now().timestamp()

            conn.execute(
                "UPDATE memories SET stability = ?, last_recalled_at = ? WHERE memory_id = ?",
                [new_stability, now, memory_id],
            )
            conn.commit()
            log.debug(
                "boost_on_recall: %s stability %.3f -> %.3f",
                memory_id, old_stability, new_stability,
            )
            return {
                "memory_id": memory_id,
                "found": True,
                "schema": kind,
                "old_stability": old_stability,
                "new_stability": new_stability,
                "last_recalled_at": now,
            }

    def get_decay_candidates(self, threshold_days: int = 30) -> List[Dict[str, Any]]:
        """Return memories eligible for Ebbinghaus decay, oldest first.

        A memory is eligible when its *effective* last-recall timestamp is
        older than ``threshold_days`` ago. The effective recall timestamp is
        defined as::

            COALESCE(NULLIF(last_recalled_at, 0), timestamp, 0)

        Rationale: the migration that added ``last_recalled_at`` used
        ``DEFAULT 0``, so legacy memories created before the migration have
        ``last_recalled_at = 0`` (the sentinel meaning "never explicitly
        recalled"). We treat 0 as "fall back to creation time". Memories
        already at ``tier='archived'`` are excluded (already forgotten).

        Args:
            threshold_days: minimum age in days since last recall. Default 30
                (matches BRAIN_ARCHITECTURE.md "5%/month if no access").

        Returns:
            List of dicts ordered by OLDEST effective recall first (the most
            decayed memories float to the top so consolidation can process
            them first). Each dict has:
            ``{memory_id, stability, recall_count, tier, timestamp,
            effective_recall_ts, days_since_recall}``.
            Empty list when the schema has no stability column (no-op).
        """
        kind = self._schema_kind()
        # Both schemas have stability column since v8.5.1 migration.
        # Use _field_sql for recall_count to handle both DB layouts.

        if threshold_days < 0:
            raise ValueError("threshold_days must be >= 0")

        with self._db_lock:
            conn = self._get_conn()
            now = datetime.now().timestamp()
            cutoff = now - (threshold_days * 86400.0)

            # Use created_at_raw which maps to "timestamp" (legacy) or "created_at" (new)
            ts_col = self._field_sql("created_at_unix", kind)
            rows = conn.execute(
                f"""
                SELECT memory_id,
                       stability,
                       {self._field_sql("recall_count", kind)} AS recall_count,
                       tier,
                       {ts_col} AS ts,
                       COALESCE(NULLIF(last_recalled_at, 0), {ts_col}, 0)
                           AS effective_recall_ts
                FROM memories
                WHERE tier NOT IN ('archived', 'guardrail')
                  AND COALESCE(NULLIF(last_recalled_at, 0), {ts_col}, 0) > 0
                  AND COALESCE(NULLIF(last_recalled_at, 0), {ts_col}, 0) < ?
                ORDER BY effective_recall_ts ASC
                """,
                [cutoff],
            ).fetchall()

            results: List[Dict[str, Any]] = []
            for r in rows:
                eff = float(r["effective_recall_ts"])
                results.append(
                    {
                        "memory_id": r["memory_id"],
                        "stability": (
                            float(r["stability"]) if r["stability"] is not None else 1.0
                        ),
                        "recall_count": (
                            int(r["recall_count"]) if r["recall_count"] is not None else 0
                        ),
                        "tier": r["tier"],
                        "timestamp": r["ts"] if "ts" in r.keys() else None,
                        "effective_recall_ts": eff,
                        "days_since_recall": (now - eff) / 86400.0,
                    }
                )
            return results

    def decay_all(
        self,
        threshold_days: int = 30,
        archive_floor: float = 0.05,
    ) -> Dict[str, Any]:
        """Apply Ebbinghaus decay to all eligible memories.

        Per BRAIN_ARCHITECTURE.md lines 63-65:
          - Decay: 5%/month if no access   (5% per 30 days, linear)
          - Archive: stability < 0.05      (move to tier='archived', soft delete)

        Decay formula (per memory)::

            days_since_recall = (now - effective_recall_ts) / 86400
            decay_amount      = days_since_recall * 0.05 / 30
            new_stability     = max(0.0, stability - decay_amount)
            if new_stability < archive_floor:
                tier = 'archived'

        We use ``tier='archived'`` instead of DELETE so the audit trail is
        preserved — operators can resurrect or inspect forgotten memories
        later. The nightly consolidation tool (``mathir_consolidate.py``)
        should call this method.

        Args:
            threshold_days: only decay memories whose last recall is older
                than this many days. Default 30.
            archive_floor: stability below this is moved to tier='archived'.
                Default 0.05.

        Returns:
            ``{decayed: N, archived: M, by_tier: {tier_name: count, ...}}``
            where ``by_tier`` is the post-decay tier distribution (useful for
            dashboards). For non-legacy schemas, returns a no-op result with
            ``skipped=True``.
        """
        kind = self._schema_kind()
        # Both 'new' (v8.5.1+) and 'legacy' schemas have the stability column
        # thanks to the idempotent migration in _ensure_db. The early-return
        # for "no stability column" was removed — fall through to the SQL.

        if threshold_days < 0:
            raise ValueError("threshold_days must be >= 0")
        if archive_floor < 0 or archive_floor > 1.0:
            raise ValueError("archive_floor must be in [0, 1]")

        with self._db_lock:
            candidates = self.get_decay_candidates(threshold_days)
            conn = self._get_conn()

            decayed = 0
            archived = 0
            # Pre-build updates to use executemany for fewer round-trips.
            decay_updates: List[tuple] = []      # (new_stability, memory_id)
            archive_updates: List[tuple] = []    # (new_stability, memory_id)

            for mem in candidates:
                days_since = mem["days_since_recall"]
                # 5% per 30 days, linear. Negative ages (clock skew) are clamped
                # to 0 to avoid boosting stability via decay.
                days_since = max(0.0, days_since)
                decay_amount = days_since * 0.05 / 30.0
                old_stability = mem["stability"]
                new_stability = max(0.0, old_stability - decay_amount)

                if new_stability < archive_floor:
                    archive_updates.append((new_stability, mem["memory_id"]))
                    archived += 1
                else:
                    decay_updates.append((new_stability, mem["memory_id"]))
                    # M1: only count items actually decayed (not archived ones),
                    # so decayed + archived == total candidates.
                    if old_stability != new_stability:
                        decayed += 1

            if decay_updates:
                conn.executemany(
                    "UPDATE memories SET stability = ? WHERE memory_id = ?",
                    decay_updates,
                )
            if archive_updates:
                conn.executemany(
                    "UPDATE memories SET stability = ?, tier = 'archived' "
                    "WHERE memory_id = ?",
                    archive_updates,
                )
            conn.commit()

            # Recount by_tier after the decay pass (for dashboards / assertions).
            tier_rows = conn.execute(
                "SELECT tier, COUNT(*) AS cnt FROM memories GROUP BY tier"
            ).fetchall()
            by_tier = {row["tier"]: row["cnt"] for row in tier_rows}

            log.info(
                "decay_all: decayed=%d, archived=%d, by_tier=%s",
                decayed, archived, by_tier,
            )
            self._log_audit("decay", details={"decayed": decayed, "archived": archived, "by_tier": by_tier})
            return {
                "decayed": decayed,
                "archived": archived,
                "by_tier": by_tier,
                "schema": kind,
            }

    def promote(self, memory_id: str, force: bool = False) -> Dict[str, Any]:
        """Promote a memory one tier up if it meets the tier-transition rules.

        Rules (skipped when ``force=True``):
        - ``working_memory → episodic``:   recall_count >= 3  AND age >= 1 day
        - ``episodic       → semantic``:   recall_count >= 10 AND age >= 7 days
        - ``semantic       → procedural``: priority >= 8 AND recall_count >= 5
                                          AND label starts with ``how-to:`` or ``recipe:``

        Returns ``{memory_id, found, promoted, old_tier, new_tier, reason}``.
        - ``promoted=False, new_tier=old_tier`` means the memory was inspected
          but did not qualify (reason explains why).
        - ``promoted=False, found=False`` means the memory doesn't exist.
        - Memories already at ``procedural`` are a no-op (already at top tier).
        """
        with self._db_lock:
            conn = self._get_conn()
            kind = self._schema_kind()

            sql = (
                "SELECT memory_id, "
                + self._field_sql("tier", kind) + " AS tier, "
                + self._field_sql("label", kind) + " AS label, "
                + self._field_sql("priority", kind) + " AS priority, "
                + self._field_sql("recall_count", kind) + " AS recall_count, "
                + self._field_sql("created_at_raw", kind) + " AS created_at, "
                + self._field_sql("last_recalled_at", kind) + " AS last_recalled_at "
                + "FROM memories WHERE memory_id = ?"
            )
            row = conn.execute(sql, [memory_id]).fetchone()
            if not row:
                return {
                    "memory_id": memory_id,
                    "found": False,
                    "promoted": False,
                    "old_tier": None,
                    "new_tier": None,
                    "reason": "memory not found",
                }

            old_tier = row["tier"] or "episodic"
            if old_tier in self.TERMINAL_TIERS:
                return {
                    "memory_id": memory_id,
                    "found": True,
                    "promoted": False,
                    "old_tier": old_tier,
                    "new_tier": old_tier,
                    "reason": f"tier {old_tier!r} is terminal (not promotable)",
                }
            if old_tier not in self.TIER_ORDER:
                return {
                    "memory_id": memory_id,
                    "found": True,
                    "promoted": False,
                    "old_tier": old_tier,
                    "new_tier": old_tier,
                    "reason": f"unknown tier {old_tier!r}",
                }

            idx = self.TIER_ORDER.index(old_tier)
            if idx >= len(self.TIER_ORDER) - 1:
                return {
                    "memory_id": memory_id,
                    "found": True,
                    "promoted": False,
                    "old_tier": old_tier,
                    "new_tier": old_tier,
                    "reason": "already at top tier (procedural)",
                }
            new_tier = self.TIER_ORDER[idx + 1]

            recall_count = int(row["recall_count"] or 0)
            priority = int(row["priority"] or 5)
            age_days = self._age_days(row["created_at"], kind)

            # Rule evaluation — gate on force=False only.
            reason = ""
            eligible = True
            if not force:
                if old_tier == "working_memory" and new_tier == "episodic":
                    if recall_count < 3:
                        eligible = False
                        reason = f"recall_count={recall_count} < 3"
                    elif age_days < 1.0:
                        eligible = False
                        reason = f"age={age_days:.3f}d < 1 day"
                    else:
                        reason = f"recall_count={recall_count}>=3 AND age={age_days:.3f}d>=1d"
                elif old_tier == "episodic" and new_tier == "semantic":
                    if recall_count < 10:
                        eligible = False
                        reason = f"recall_count={recall_count} < 10"
                    elif age_days < 7.0:
                        eligible = False
                        reason = f"age={age_days:.3f}d < 7 days"
                    else:
                        reason = f"recall_count={recall_count}>=10 AND age={age_days:.3f}d>=7d"
                elif old_tier == "semantic" and new_tier == "procedural":
                    # FIX (2026-07-02): previously also required label to
                    # start with "how-to:"/"recipe:" -- an undocumented
                    # convention nothing in MATHIR ever generates or tells
                    # callers to use. Scanned 24,235 real memories across
                    # every real project this session touched: ZERO had
                    # such a label, meaning this promotion path had NEVER
                    # fired once in MATHIR's real history. priority>=8 AND
                    # recall_count>=5 alone are a real, reachable signal of
                    # durable/reusable content -- the label gate is removed.
                    if priority < 8:
                        eligible = False
                        reason = f"priority={priority} < 8"
                    elif recall_count < 5:
                        eligible = False
                        reason = f"recall_count={recall_count} < 5"
                    else:
                        reason = (
                            f"priority={priority}>=8, recall_count={recall_count}>=5"
                        )
                else:
                    # (working_memory→semantic would skip episodic — handled above
                    # by the index step. Any other transition is unsupported.)
                    eligible = False
                    reason = f"no rule for {old_tier} → {new_tier}"
            else:
                reason = "force=True (skipped rule check)"

            if not eligible:
                return {
                    "memory_id": memory_id,
                    "found": True,
                    "promoted": False,
                    "old_tier": old_tier,
                    "new_tier": old_tier,
                    "reason": reason,
                }

            conn.execute(
                "UPDATE memories SET tier = ? WHERE memory_id = ?",
                [new_tier, memory_id],
            )
            conn.commit()
            log.info(
                "promote: %s %s -> %s (%s)", memory_id, old_tier, new_tier, reason
            )
            self._log_audit("promote", memory_id=memory_id,
                            details={"old_tier": old_tier, "new_tier": new_tier, "reason": reason})
            return {
                "memory_id": memory_id,
                "found": True,
                "promoted": True,
                "old_tier": old_tier,
                "new_tier": new_tier,
                "reason": reason,
            }

    def auto_promote_all(self) -> List[Dict[str, Any]]:
        """Scan every memory and promote those that currently meet the rules.

        Returns a list of ``{memory_id, old_tier, new_tier, reason}`` for each
        memory that was actually promoted. Memories that were inspected but
        didn't qualify are not included (use ``promote(id)`` to inspect
        individually).
        """
        with self._db_lock:
            conn = self._get_conn()
            rows = conn.execute("SELECT memory_id FROM memories").fetchall()
            promoted: List[Dict[str, Any]] = []
            for row in rows:
                mid = row["memory_id"]
                result = self.promote(mid, force=False)
                if result.get("promoted"):
                    promoted.append(
                        {
                            "memory_id": mid,
                            "old_tier": result["old_tier"],
                            "new_tier": result["new_tier"],
                            "reason": result.get("reason", ""),
                        }
                    )
            log.info("auto_promote_all: scanned %d, promoted %d", len(rows), len(promoted))
            return promoted

    def _age_days(self, created_at: Any, kind: str) -> float:
        """Return the age of a memory in days, given its raw created_at value.

        - legacy schema: ``created_at`` is a Unix timestamp (REAL)
        - new schema:    ``created_at`` is an ISO 8601 string

        Returns ``float('inf')`` on parse failure (M4) so corrupt timestamps
        *block* age-gated promotion rather than auto-passing it. A sentinel
        of 0.0 would have made every rule that checks ``age >= N days`` pass
        for bad rows — the opposite of safe.
        """
        if created_at is None or created_at == "":
            return float("inf")
        try:
            if kind == "legacy" and isinstance(created_at, (int, float)):
                dt = datetime.fromtimestamp(float(created_at))
            elif isinstance(created_at, str):
                # ISO 8601 (new schema) — fromisoformat handles most forms.
                dt = datetime.fromisoformat(created_at)
            elif isinstance(created_at, (int, float)):
                dt = datetime.fromtimestamp(float(created_at))
            else:
                return float("inf")
            return max(0.0, (datetime.now() - dt).total_seconds() / 86400.0)
        except (ValueError, TypeError, OSError, OverflowError):
            return float("inf")

    # ------------------------------------------------------------------
    # Memory Consolidation (Phase 4 of BRAIN_ARCHITECTURE.md)
    # ------------------------------------------------------------------
    #
    # Merges near-duplicate memories (cosine > 0.95) into a single
    # canonical record. Mirrors what the brain does during slow-wave
    # sleep — duplicate engrams are folded together so they surface as
    # one consolidated memory.
    #
    # Design contract:
    #   * find_duplicates()      — pure read, returns candidate pairs
    #   * consolidate_pair()     — single merge, transactional
    #   * consolidate_all()      — orchestrator (dry_run supported)
    #   * Schema-agnostic        — handles BOTH schema branches
    #     (A) new schema with `content` column (recall_count/stability live in metadata JSON)
    #     (B) legacy MATHIR schema (recall_count/stability are top-level columns)
    #
    # When merging:
    #   - canonical_id = stronger memory (highest recall_count, then stability, then content length)
    #   - weaker.memory_id appended to canonical.metadata.merged_from[] (audit trail)
    #   - canonical.recall_count += weaker.recall_count
    #   - canonical.stability = max(canonical.stability, weaker.stability)
    #   - weaker is removed from vec_memories (so search ignores it)
    #   - weaker is NOT hard-deleted in `memories` — its `tier` is set to 'archived'
    #     so the audit trail survives
    # ------------------------------------------------------------------

    def _schema_info(self) -> Dict[str, Any]:
        """Detect schema variant in use and report available columns.

        Returns
        -------
        dict with:
            new_schema   : bool   — True if the `content` column exists (schema A)
            has_metadata : bool   — True if `metadata` is a top-level JSON column
            has_recall   : bool   — True if `recall_count` is a top-level column
            has_stab     : bool   — True if `stability` is a top-level column
            columns      : set    — all column names
        """
        with self._db_lock:
            conn = self._get_conn()
            cursor = conn.execute("PRAGMA table_info(memories)")
            cols = {col[1] for col in cursor.fetchall()}
            return {
                "new_schema": "content" in cols,
                "has_metadata": "metadata" in cols,
                "has_recall": "recall_count" in cols,
                "has_stab": "stability" in cols,
                "columns": cols,
            }

    def _load_meta(self, memory_id: str) -> Dict[str, Any]:
        """Load a memory row as a unified dict regardless of schema variant.

        The returned dict always has these normalized keys:
            memory_id, content, tier, recall_count, stability, metadata (dict)

        For schema A (new): recall_count/stability come from metadata JSON.
        For schema B (legacy): they come from top-level columns; metadata
        is still parsed if present.
        """
        with self._db_lock:
            conn = self._get_conn()
            info = self._schema_info()
            row = conn.execute("SELECT * FROM memories WHERE memory_id = ?", [memory_id]).fetchone()
            if row is None:
                return {}

            keys = row.keys()
            result: Dict[str, Any] = {"memory_id": row["memory_id"]}

            # content + tier + metadata (with fallbacks for both schemas)
            if "content" in keys:
                result["content"] = row["content"] or ""
            elif "modality_text" in keys:
                result["content"] = row["modality_text"] or ""
            else:
                result["content"] = ""

            if "tier" in keys:
                result["tier"] = row["tier"] or "episodic"
            else:
                result["tier"] = "episodic"

            # metadata: prefer top-level column, fall back to per-row content
            meta_raw = None
            if "metadata" in keys:
                meta_raw = row["metadata"]
            if meta_raw:
                try:
                    result["metadata"] = json.loads(meta_raw)
                except (TypeError, ValueError):
                    result["metadata"] = {}
            else:
                result["metadata"] = {}

            # recall_count / stability — schema B uses top-level columns; schema A stores them in JSON
            if info["has_recall"]:
                try:
                    result["recall_count"] = int(row["recall_count"] or 0)
                except (TypeError, ValueError):
                    result["recall_count"] = 0
            else:
                result["recall_count"] = int(result["metadata"].get("recall_count", 0) or 0)

            if info["has_stab"]:
                try:
                    result["stability"] = float(row["stability"] or 1.0)
                except (TypeError, ValueError):
                    result["stability"] = 1.0
            else:
                result["stability"] = float(result["metadata"].get("stability", 1.0) or 1.0)

            return result

    def _save_meta(self, record: Dict[str, Any]) -> None:
        """Persist a normalized record back to the DB, schema-aware.

        Updates content, tier, recall_count, stability, and metadata
        using the column layout that exists in the current schema.
        """
        with self._db_lock:
            conn = self._get_conn()
            info = self._schema_info()
            mid = record["memory_id"]

            # Always rebuild metadata JSON to capture any merged_from / consolidation audit fields
            meta = dict(record.get("metadata") or {})
            meta["recall_count"] = int(record.get("recall_count", 0) or 0)
            meta["stability"] = float(record.get("stability", 1.0) or 1.0)
            if record.get("content") is not None:
                meta["content"] = record["content"]
            meta["tier"] = record.get("tier", meta.get("tier", "episodic"))
            meta_json = json.dumps(meta)

            if info["new_schema"]:
                # Schema A — content/agent/block_type/etc. are top-level columns
                updates = {
                    "content": record.get("content", meta.get("content", "")),
                    "tier": record.get("tier", meta.get("tier", "episodic")),
                    "metadata": meta_json,
                }
                set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
                conn.execute(
                    f"UPDATE memories SET {set_clause} WHERE memory_id = ?",
                    list(updates.values()) + [mid],
                )
            elif info["has_recall"] or info["has_stab"]:
                # Schema B — recall_count/stability are top-level; metadata holds the rest
                updates: Dict[str, Any] = {
                    "modality_text": record.get("content", meta.get("content", "")),
                    "metadata": meta_json,
                    "tier": record.get("tier", meta.get("tier", "episodic")),
                }
                if info["has_recall"]:
                    updates["recall_count"] = int(record.get("recall_count", 0) or 0)
                if info["has_stab"]:
                    updates["stability"] = float(record.get("stability", 1.0) or 1.0)
                set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
                conn.execute(
                    f"UPDATE memories SET {set_clause} WHERE memory_id = ?",
                    list(updates.values()) + [mid],
                )
            else:
                # Pure JSON-only fallback — only metadata is updatable
                conn.execute(
                    "UPDATE memories SET metadata = ? WHERE memory_id = ?",
                    [meta_json, mid],
                )

    def find_duplicates(self, threshold: float = 0.95, limit: int = 100) -> List[Dict[str, Any]]:
        """Find near-duplicate memory pairs with cosine similarity above `threshold`.

        Uses sqlite-vec KNN when available, otherwise brute-force cosine over
        the embeddings_brute table. Pairs are deduplicated (a,b) == (b,a) and
        the weaker side is reported as `memory_id_b`.

        Parameters
        ----------
        threshold : float
            Cosine similarity threshold in [0, 1]. Default 0.95 (per BRAIN_ARCHITECTURE.md).
        limit : int
            Maximum number of pairs to return.

        Returns
        -------
        list of dicts: [{memory_id_a, memory_id_b, similarity}, ...]
            sorted by similarity DESC, capped at `limit`.
        """
        with self._db_lock:
            conn = self._get_conn()

            # Candidate IDs come from the canonical memories table.
            # We deliberately exclude 'archived' so already-merged memories don't
            # churn the result set.
            id_rows = conn.execute(
                "SELECT memory_id FROM memories WHERE tier != 'archived' OR tier IS NULL"
            ).fetchall()
            all_ids = [r["memory_id"] for r in id_rows]
            if len(all_ids) < 2:
                return []

            pairs: Dict[tuple, float] = {}

            if HAS_VEC:
                # Per-memory KNN — sqlite-vec returns cosine *distance*, so we
                # convert to similarity (1 - distance) and filter.
                # k=8 is a pragmatic cap; memories only merge with their closest
                # neighbors in practice (near-duplicate clusters are small).
                for mid in all_ids:
                    row = conn.execute(
                        "SELECT embedding FROM vec_memories WHERE memory_id = ?",
                        [mid],
                    ).fetchone()
                    if row is None:
                        continue
                    emb_hex = row["embedding"].hex()
                    neighbors = conn.execute(
                        f"SELECT memory_id, distance FROM vec_memories "
                        f"WHERE embedding MATCH vec_int8(X'{emb_hex}') AND k = 8 "
                        f"AND memory_id != ?",
                        [mid],
                    ).fetchall()
                    for nb in neighbors:
                        sim = 1.0 - float(nb["distance"])
                        if sim < threshold:
                            continue
                        a, b = (mid, nb["memory_id"]) if mid < nb["memory_id"] else (nb["memory_id"], mid)
                        prev = pairs.get((a, b))
                        if prev is None or sim > prev:
                            pairs[(a, b)] = sim
            else:
                # Brute-force fallback — load all vectors, normalize, pairwise cosine.
                rows = conn.execute("SELECT memory_id, embedding FROM embeddings_brute").fetchall()
                if not rows:
                    return []
                embs: List[np.ndarray] = []
                ids: List[str] = []
                for row in rows:
                    embs.append(_decode_brute_blob(row["embedding"]))
                    ids.append(row["memory_id"])
                mat = np.stack(embs)
                norms = np.linalg.norm(mat, axis=1)
                norms = np.where(norms == 0, 1.0, norms)
                mat_unit = mat / norms[:, None]
                sims = mat_unit @ mat_unit.T
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        sim = float(sims[i, j])
                        if sim < threshold:
                            continue
                        a, b = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                        pairs[(a, b)] = sim

            ordered = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)
            return [
                {"memory_id_a": a, "memory_id_b": b, "similarity": s}
                for (a, b), s in ordered[:limit]
            ]

    def consolidate_pair(self, id_strong: str, id_weak: str) -> Dict[str, Any]:
        """Merge `id_weak` into `id_strong` (the canonical survivor).

        The weaker memory is NOT deleted — its `tier` is set to 'archived'
        and it is removed from the vector index so future searches skip it.
        Its `memory_id` is appended to the stronger's `metadata.merged_from[]`
        list, providing a complete audit trail.

        Parameters
        ----------
        id_strong : str
            Canonical (surviving) memory_id.
        id_weak : str
            Memory to fold into the canonical one.

        Returns
        -------
        dict:
            {
              "canonical_id"    : str,
              "merged_from"     : str,
              "new_recall_count": int,
              "new_stability"   : float,
              "events"          : list[str]  # human-readable audit trail
            }

        Raises
        ------
        ValueError if either id is missing from the memories table.
        """
        if id_strong == id_weak:
            raise ValueError("id_strong and id_weak must differ")

        with self._db_lock:
            conn = self._get_conn()
            strong = self._load_meta(id_strong)
            weak = self._load_meta(id_weak)
            if not strong:
                raise ValueError(f"id_strong not found: {id_strong!r}")
            if not weak:
                raise ValueError(f"id_weak not found: {id_weak!r}")

            events: List[str] = []

            # 1. recall_count: sum
            new_recall = int(strong.get("recall_count", 0)) + int(weak.get("recall_count", 0))
            if new_recall != int(strong.get("recall_count", 0)):
                events.append(
                    f"recall_count {strong.get('recall_count', 0)} -> {new_recall}"
                )
            strong["recall_count"] = new_recall

            # 2. stability: max (more stable = more durable = keep the higher)
            new_stab = max(float(strong.get("stability", 1.0)), float(weak.get("stability", 1.0)))
            if new_stab != float(strong.get("stability", 1.0)):
                events.append(
                    f"stability {strong.get('stability', 1.0)} -> {new_stab}"
                )
            strong["stability"] = new_stab

            # 3. metadata.merged_from[] audit trail
            meta = dict(strong.get("metadata") or {})
            merged_from = list(meta.get("merged_from") or [])
            if id_weak not in merged_from:
                merged_from.append(id_weak)
            meta["merged_from"] = merged_from

            # Also capture weak's metadata fields that aren't already set —
            # useful audit context (don't overwrite stronger's values, only fill gaps)
            weak_meta = dict(weak.get("metadata") or {})
            weak_snapshot_keys = ("agent", "block_type", "label", "project", "created_at")
            if not any(meta.get(k) for k in weak_snapshot_keys):
                for k in weak_snapshot_keys:
                    if weak_meta.get(k):
                        meta.setdefault(k, weak_meta[k])
            meta["last_consolidation"] = datetime.now().isoformat()
            meta["last_consolidated_with"] = id_weak
            strong["metadata"] = meta

            # 4. Persist the canonical
            self._save_meta(strong)
            events.append(f"merged metadata from {id_weak}")

            # 5. Archive the weaker: tier='archived' + record metadata.
            #    Memory row stays for audit; vector index removes it so it
            #    is invisible to recall.
            weak_meta_out = dict(weak.get("metadata") or {})
            weak_meta_out["archived_at"] = datetime.now().isoformat()
            weak_meta_out["archived_into"] = id_strong
            weak_meta_out["archive_reason"] = "consolidation"
            weak_record = {
                "memory_id": id_weak,
                "tier": "archived",
                "metadata": weak_meta_out,
                "recall_count": weak.get("recall_count", 0),
                "stability": weak.get("stability", 1.0),
                "content": weak.get("content", ""),
            }
            # H22: the memories archive (UPDATE) and vec-index delete (DELETE)
            # must commit or roll back together — a failure between them would
            # leave an archived row still searchable, or a deleted vector with
            # no archive trail. They share `conn` with no intervening commit,
            # so they're in one implicit transaction; the try/except below
            # rolls that transaction back atomically on any failure.
            try:
                self._save_meta(weak_record)
                if HAS_VEC:
                    conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", [id_weak])
                else:
                    conn.execute("DELETE FROM embeddings_brute WHERE memory_id = ?", [id_weak])
                events.append(f"archived {id_weak}")
                conn.commit()
            except Exception:
                conn.rollback()
                raise

            return {
                "canonical_id": id_strong,
                "merged_from": id_weak,
                "new_recall_count": new_recall,
                "new_stability": new_stab,
                "events": events,
            }

    def _pick_stronger(self, a: Dict[str, Any], b: Dict[str, Any]) -> tuple:
        """Return (stronger, weaker) using: recall_count > stability > content length.

        Ties are broken by lexical memory_id order for determinism (a < b means
        a is "stronger"), so consolidations are reproducible.
        """
        if int(a["recall_count"]) != int(b["recall_count"]):
            return (a, b) if a["recall_count"] > b["recall_count"] else (b, a)
        if float(a["stability"]) != float(b["stability"]):
            return (a, b) if a["stability"] > b["stability"] else (b, a)
        if len(a["content"]) != len(b["content"]):
            return (a, b) if len(a["content"]) > len(b["content"]) else (b, a)
        # Deterministic tie-breaker
        return (a, b) if a["memory_id"] <= b["memory_id"] else (b, a)

    def consolidate_all(
        self,
        threshold: float = 0.95,
        dry_run: bool = True,
        limit: int = 100,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """Find and merge all near-duplicate pairs above `threshold`.

        Parameters
        ----------
        threshold : float
            Cosine similarity threshold (default 0.95).
        dry_run : bool
            If True, return what WOULD be merged without modifying any rows.
        limit : int
            Maximum number of candidate pairs to consider (caps the scan).
        max_results : int
            For dry_run only: cap on the number of compact preview events
            returned in ``events``. Candidates beyond this cap are counted
            in ``candidates`` / ``returned`` / ``truncated`` but not emitted,
            keeping the response small (the previous shape returned 228K+
            chars on ~700 memories). Ignored when ``dry_run=False``.

        Returns
        -------
        dict:
            {
              "dry_run"     : bool,
              "threshold"   : float,
              "merged"      : int,        # count of pairs actually merged (0 in dry_run)
              "candidates"  : int,        # total candidate pairs found (post-``limit``)
              "returned"    : int,        # events actually included in the response
              "truncated"   : bool,       # True iff candidates > returned (dry_run only)
              "by_tier"     : dict,       # tier distribution after the operation
              "events"      : list[dict]  # dry_run: compact pair previews; live: rich per-merge events
            }

        Dry_run event shape (compact, NEW in v8.7.1):
            {memory_id_a, memory_id_b, snippet_a, snippet_b, similarity}
        """
        with self._db_lock:
            candidates = self.find_duplicates(threshold=threshold, limit=limit)
            by_tier: Dict[str, int] = {}

            # Snapshot tier distribution BEFORE the operation so the report
            # reflects what actually happened (or would happen) on the DB.
            conn = self._get_conn()
            tier_rows = conn.execute(
                "SELECT tier, COUNT(*) AS cnt FROM memories GROUP BY tier"
            ).fetchall()
            for r in tier_rows:
                by_tier[r["tier"] or "(none)"] = int(r["cnt"])

            events: List[Dict[str, Any]] = []
            merged_count = 0

            for pair in candidates:
                id_a = pair["memory_id_a"]
                id_b = pair["memory_id_b"]
                sim = pair["similarity"]

                a = self._load_meta(id_a)
                b = self._load_meta(id_b)
                if not a or not b:
                    # One side disappeared between find and merge (e.g. archived
                    # by an earlier iteration) — skip silently.
                    continue
                if a.get("tier") == "archived" or b.get("tier") == "archived":
                    continue

                strong, weak = self._pick_stronger(a, b)
                strong_id = strong["memory_id"]
                weak_id = weak["memory_id"]

                if dry_run:
                    # Compact preview: ids + ~100-char snippets + similarity only.
                    # Full content / would_set details are intentionally omitted
                    # to keep the response bounded (see max_results docstring).
                    events.append({
                        "memory_id_a": id_a,
                        "memory_id_b": id_b,
                        "snippet_a": _snippet(a.get("content", ""), 100),
                        "snippet_b": _snippet(b.get("content", ""), 100),
                        "similarity": sim,
                    })
                else:
                    result = self.consolidate_pair(strong_id, weak_id)
                    result["similarity"] = sim
                    result["applied"] = True
                    events.append(result)
                    merged_count += 1

            # Final tier snapshot
            if not dry_run:
                tier_rows = conn.execute(
                    "SELECT tier, COUNT(*) AS cnt FROM memories GROUP BY tier"
                ).fetchall()
                by_tier = {r["tier"] or "(none)": int(r["cnt"]) for r in tier_rows}

            total_candidates = len(candidates)
            if dry_run:
                # Cap the preview so the response stays small regardless of
                # how large `limit` was. Caller still sees `candidates` (total)
                # and `truncated` to know whether to narrow `threshold`.
                returned = min(max_results, len(events))
                truncated = len(events) > max_results
                events_out = events[:max_results]
            else:
                returned = len(events)
                truncated = False
                events_out = events

            return {
                "dry_run": dry_run,
                "threshold": threshold,
                "merged": merged_count,
                "candidates": total_candidates,
                "returned": returned,
                "truncated": truncated,
                "by_tier": by_tier,
                "events": events_out,
            }

    def run_maintenance(
        self,
        decay_rate: float = 0.05,
        do_decay: bool = True,
        do_promote: bool = True,
        do_dedupe: bool = True,
        do_links: bool = True,
    ) -> Dict[str, Any]:
        """Single canonical lifecycle entry point — nightly maintenance.

        Composes the four lifecycle passes (decay, promote, dedupe, link
        build) into one call so cron / the MATHIR brain have exactly ONE
        place to invoke. Replaces the prior situation where callers had to
        know to invoke ``decay_all`` / ``apply_decay`` (consolidate.py) /
        ``bump_recall`` separately and they would diverge over time.

        Each step is wrapped in its own try/except so a single failing step
        doesn't abort the others — failures are recorded in ``errors`` and
        the run continues. Sub-calls each acquire ``_db_lock`` (RLock), so
        this method does NOT take the lock itself — that would risk
        dead-locking against concurrent recalls.

        Args:
            decay_rate:   Per-month decay fraction (default 0.05 = 5% / 30d,
                          per BRAIN_ARCHITECTURE.md). Currently informational
                          — ``decay_all`` hardcodes the 0.05 rate internally
                          and is not refactored to accept this param (do not
                          refactor per change request). Accepted for API
                          symmetry and future plumbing.
            do_decay:     If True, run ``decay_all()`` (Ebbinghaus forgetting).
            do_promote:   If True, run ``auto_promote_all()`` (tier upgrades).
            do_dedupe:    If True, run ``consolidate_all(dry_run=False)`` to
                          actually merge near-duplicate pairs.
            do_links:     If True, run ``build_links_all()`` (cosine graph).

        Returns:
            {
              'decay':    {'decayed': N, 'archived': M},  # {} if step skipped/failed
              'promoted': K,                              # count of tier upgrades
              'deduped':  P,                              # count of merged pairs
              'links':    L,                              # count of links created
              'errors':   [{'step': str, 'error': str}, ...]
            }
        """
        summary: Dict[str, Any] = {
            'decay': {},
            'promoted': 0,
            'deduped': 0,
            'links': 0,
            'errors': [],
        }

        # 1. Decay — Ebbinghaus forgetting curve on cold memories.
        if do_decay:
            try:
                # decay_all currently hardcodes its 0.05 rate (see line ~1251);
                # decay_rate is accepted for API symmetry but not yet plumbed
                # through (do NOT refactor per change request).
                result = self.decay_all()
                summary['decay'] = {
                    'decayed': int(result.get('decayed', 0)),
                    'archived': int(result.get('archived', 0)),
                }
            except Exception as exc:
                summary['errors'].append({'step': 'decay', 'error': str(exc)})

        # 2. Promote — bump tiers for memories meeting recall/age rules.
        if do_promote:
            try:
                promoted = self.auto_promote_all()
                summary['promoted'] = len(promoted)
            except Exception as exc:
                summary['errors'].append({'step': 'promote', 'error': str(exc)})

        # 3. Dedupe — fold near-duplicates (cosine > 0.95) into one record.
        if do_dedupe:
            try:
                # dry_run=False so maintenance actually merges pairs.
                result = self.consolidate_all(dry_run=False)
                summary['deduped'] = int(result.get('merged', 0))
            except Exception as exc:
                summary['errors'].append({'step': 'dedupe', 'error': str(exc)})

        # 4. Link graph — rebuild cosine-similarity edges for spreading activation.
        if do_links:
            try:
                result = self.build_links_all()
                summary['links'] = int(result.get('links_created', 0))
            except Exception as exc:
                summary['errors'].append({'step': 'links', 'error': str(exc)})

        log.info(
            "run_maintenance: decayed=%d archived=%d promoted=%d deduped=%d links=%d errors=%d",
            summary['decay'].get('decayed', 0),
            summary['decay'].get('archived', 0),
            summary['promoted'],
            summary['deduped'],
            summary['links'],
            len(summary['errors']),
        )
        return summary

    def get_stored_embedding_model(self) -> Optional[str]:
        """Return the embedding model this DB was created with, or None if
        no memory has ever been embedded here yet (brand-new DB)."""
        with self._db_lock:
            conn = self._get_conn()
            row = conn.execute("SELECT embedding_model FROM db_meta WHERE id = 1").fetchone()
            return row["embedding_model"] if row else None

    # Every project DB created before this per-DB model-pinning feature
    # existed was embedded with this single model -- there was no other
    # option in MATHIR's history until now. Used to backfill pre-existing
    # DBs correctly (see ensure_embedding_model) instead of letting them
    # silently inherit whatever the CURRENTLY configured default is.
    LEGACY_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def ensure_embedding_model(self, model_name: str) -> str:
        """Return the model this DB should use, pinning it on first call.

        Three cases:
        1. db_meta already has a recorded model -> return it, ignore
           `model_name`. This is what prevents a MATHIR-wide default-model
           change from silently mixing two incompatible embedding spaces
           in an already-populated project DB.
        2. No db_meta row AND no existing memories (genuinely brand-new
           DB) -> record and return `model_name` (the current configured
           default). This is how new projects pick up a new default.
        3. No db_meta row BUT existing memories already present (a DB
           created before this per-DB tracking feature existed) -> record
           and return LEGACY_DEFAULT_MODEL, NOT `model_name`. This is the
           critical case: without it, upgrading MATHIR and changing the
           default embedder would silently re-tag every pre-existing
           project onto the new model on first access after upgrade,
           corrupting retrieval against its real, already-stored vectors
           (caught live: locomo_conv_2, 1326 real memories, would have
           been mis-pinned to a newly-configured default without this).
        """
        with self._db_lock:
            conn = self._get_conn()
            row = conn.execute("SELECT embedding_model FROM db_meta WHERE id = 1").fetchone()
            if row and row["embedding_model"]:
                return row["embedding_model"]

            has_existing_memories = conn.execute(
                "SELECT 1 FROM memories LIMIT 1"
            ).fetchone() is not None
            resolved = self.LEGACY_DEFAULT_MODEL if has_existing_memories else model_name

            conn.execute(
                "INSERT OR REPLACE INTO db_meta (id, embedding_model) VALUES (1, ?)",
                [resolved],
            )
            conn.commit()
            return resolved

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def rebuild_vec_index(self, embedder=None, batch_size: int = 64) -> Dict[str, Any]:
        """Rebuild vec_memories (or embeddings_brute) from memories table.

        Called after direct SQL migration where vec_memories was not populated.
        If embedder is None, skips embedding generation and returns a count of
        memories needing embedding.

        Args:
            embedder: A sentence-transformers model with .encode() method.
                      If None, returns {skipped: True, pending: N}.
            batch_size: Number of memories to encode per batch.

        Returns:
            {rebuilt: N, skipped: bool, pending: N, errors: M}
        """
        with self._db_lock:
            conn = self._get_conn()
            # Count memories without embeddings
            if HAS_VEC:
                rows = conn.execute(
                    "SELECT m.memory_id FROM memories m "
                    "LEFT JOIN vec_memories v ON m.memory_id = v.memory_id "
                    "WHERE v.memory_id IS NULL"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT m.memory_id FROM memories m "
                    "LEFT JOIN embeddings_brute e ON m.memory_id = e.memory_id "
                    "WHERE e.memory_id IS NULL"
                ).fetchall()

            pending = len(rows)
            if pending == 0:
                return {"rebuilt": 0, "skipped": False, "pending": 0, "errors": 0}

            if embedder is None:
                return {"rebuilt": 0, "skipped": True, "pending": pending, "errors": 0}

            # Import numpy lazily
            import numpy as np

            rebuilt = 0
            errors = 0
            ids_to_embed = [r["memory_id"] for r in rows]

            for i in range(0, len(ids_to_embed), batch_size):
                batch_ids = ids_to_embed[i:i + batch_size]
                # Fetch content for encoding
                placeholders = ",".join("?" for _ in batch_ids)
                content_rows = conn.execute(
                    f"SELECT memory_id, content FROM memories WHERE memory_id IN ({placeholders})",
                    batch_ids
                ).fetchall()
                if not content_rows:
                    continue

                texts = [r["content"] or "" for r in content_rows]
                ids = [r["memory_id"] for r in content_rows]

                try:
                    embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=batch_size)
                    for mid, emb in zip(ids, embeddings):
                        try:
                            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
                            if len(vec) != self.embedding_dim:
                                errors += 1
                                continue
                            if HAS_VEC:
                                conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", [mid])
                                hex_str = _serialize_embedding(vec)
                                conn.execute(
                                    f"INSERT INTO vec_memories(memory_id, embedding) VALUES (?, vec_int8(X'{hex_str}'))",
                                    [mid]
                                )
                            else:
                                blob = struct.pack("<i", vec.size) + vec.tobytes()
                                conn.execute(
                                    "INSERT OR REPLACE INTO embeddings_brute(memory_id, embedding) VALUES (?, ?)",
                                    [mid, blob]
                                )
                            rebuilt += 1
                        except Exception:
                            errors += 1
                except Exception as e:
                    log.warning(f"rebuild_vec_index batch {i}: {e}")
                    errors += len(batch_ids)

            conn.commit()
            log.info(f"rebuild_vec_index: rebuilt={rebuilt}, pending={pending}, errors={errors}")
            return {"rebuilt": rebuilt, "skipped": False, "pending": pending, "errors": errors}


# Global cache
_vec_memory_cache: Dict[str, VecMemory] = {}


def get_vec_memory(project_name: str = None, embedding_dim: int = 384) -> VecMemory:
    """Get or create a VecMemory instance for a project."""
    from mathir_mcp_server import get_project_db_path
    
    if project_name is None:
        from mathir_mcp_server import get_project_name
        project_name = get_project_name()
    
    cache_key = f"{project_name}:{embedding_dim}"
    if cache_key not in _vec_memory_cache:
        db_path = get_project_db_path(project_name)
        _vec_memory_cache[cache_key] = VecMemory(db_path, embedding_dim)
        log.info(f"VecMemory created for project '{project_name}': {db_path}")
    
    return _vec_memory_cache[cache_key]

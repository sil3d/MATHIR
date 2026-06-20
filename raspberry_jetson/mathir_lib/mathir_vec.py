"""
MATHIR Vec — sqlite-vec accelerated vector storage for MATHIR.
Provides O(log N) approximate nearest neighbor search instead of O(N) brute-force.
"""

import sqlite3
import struct
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

log = logging.getLogger("mathir-vec")

# Try to import sqlite-vec
try:
    import sqlite_vec
    HAS_VEC = True
    log.info("sqlite-vec available")
except ImportError:
    HAS_VEC = False
    log.warning("sqlite-vec not installed — using brute-force fallback")


def _serialize_embedding(vec: np.ndarray) -> bytes:
    """Serialize float32 numpy array to bytes for sqlite-vec."""
    return sqlite_vec.serialize_float32(vec.tolist())


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    """Deserialize bytes back to float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32).copy()


class VecMemory:
    """
    SQLite-backed vector memory with sqlite-vec acceleration.
    
    Uses vec0 virtual table for O(log N) approximate nearest neighbor search.
    Falls back to brute-force cosine similarity if sqlite-vec is not available.
    """
    
    # Whitelist of valid embedding dimensions (prevents SQL injection via embedding_dim)
    VALID_DIMS = {64, 128, 256, 384, 512, 768, 1024, 2048, 4096}

    def __init__(self, db_path: Path, embedding_dim: int = 384):
        if not isinstance(embedding_dim, int) or embedding_dim not in self.VALID_DIMS:
            raise ValueError(
                f"embedding_dim must be one of {sorted(self.VALID_DIMS)}, got {embedding_dim!r}"
            )
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._conn = None
        self._ensure_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get or create SQLite connection with sqlite-vec loaded."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            if HAS_VEC:
                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)
                self._conn.enable_load_extension(False)
                log.info(f"sqlite-vec loaded for {self.db_path.name}")
        return self._conn
    
    def _ensure_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        
        # Check if memories table exists and get its schema
        cursor = conn.execute("PRAGMA table_info(memories)")
        columns = {col[1]: col[2] for col in cursor.fetchall()}
        
        if not columns:
            # Create new table with our schema
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
                    metadata JSON
                )
            """)
            log.info("Created new memories table")
        else:
            # Use existing schema - we'll adapt our queries
            log.info(f"Using existing memories table with columns: {list(columns.keys())}")
        
        # Vec table for vector search (if sqlite-vec available)
        if HAS_VEC:
            # Auto-detect dimension from existing vec_memories table
            import re
            try:
                cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_memories'")
                row = cursor.fetchone()
                if row:
                    m = re.search(r'FLOAT\[(\d+)\]', row[0])
                    if m:
                        existing_dim = int(m.group(1))
                        if existing_dim != self.embedding_dim:
                            log.info(f"Auto-detecting dimension: DB has {existing_dim}, requested {self.embedding_dim}. Using DB dimension.")
                            self.embedding_dim = existing_dim
            except Exception:
                pass

            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                    memory_id TEXT PRIMARY KEY,
                    embedding FLOAT[{self.embedding_dim}] distance_metric=cosine
                )
            """)
            log.info(f"vec0 table ready: dim={self.embedding_dim}")
        else:
            # Brute-force fallback table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_brute (
                    memory_id TEXT PRIMARY KEY,
                    embedding BLOB
                )
            """)
        
        conn.commit()
    
    def store(self, memory_id: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        """Store a memory with its embedding vector."""
        conn = self._get_conn()
        
        # Ensure embedding is the right shape and type
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if len(vec) != self.embedding_dim:
            raise ValueError(f"Embedding dim {len(vec)} != expected {self.embedding_dim}")
        
        # Check if we're using the existing MATHIR schema or our new schema
        cursor = conn.execute("PRAGMA table_info(memories)")
        columns = {col[1] for col in cursor.fetchall()}
        
        if "content" in columns:
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
            # Existing MATHIR schema
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
                datetime.now().timestamp(),
                metadata.get("tier", "episodic"),
                1.0,
                0,
                "mathir-vec",
                "vec"
            ))
        
        # Store vector
        if HAS_VEC:
            # Delete existing if present (sqlite-vec doesn't support INSERT OR REPLACE)
            conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", [memory_id])
            conn.execute(
                "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
                [memory_id, _serialize_embedding(vec)]
            )
        else:
            # Brute-force: store as BLOB with length header
            blob = struct.pack("<i", vec.size) + vec.tobytes()
            conn.execute(
                "INSERT OR REPLACE INTO embeddings_brute(memory_id, embedding) VALUES (?, ?)",
                [memory_id, blob]
            )
        
        conn.commit()
        return memory_id
    
    def search(self, query_embedding: np.ndarray, k: int = 5, 
               agent_filter: str = None, block_type_filter: str = None) -> List[Dict[str, Any]]:
        """Search for similar memories using vector similarity."""
        conn = self._get_conn()
        
        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if len(query_vec) != self.embedding_dim:
            raise ValueError(f"Query dim {len(query_vec)} != expected {self.embedding_dim}")
        
        # Check which schema we're using
        cursor = conn.execute("PRAGMA table_info(memories)")
        columns = {col[1] for col in cursor.fetchall()}
        new_schema = "content" in columns
        
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
            # Use sqlite-vec for fast search
            query_blob = _serialize_embedding(query_vec)
            if new_schema:
                sql = f"""
                    SELECT m.memory_id, m.content, m.agent, m.block_type, m.label, 
                           m.priority, m.tier, m.project, m.created_at, v.distance
                    FROM vec_memories v
                    JOIN memories m ON v.memory_id = m.memory_id
                    WHERE v.embedding MATCH ?
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
                    WHERE v.embedding MATCH ?
                      AND k = ?
                      {where_sql}
                """
            params = [query_blob, k * 2] + params  # Extra for filtering
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
                n = struct.unpack("<i", blob[:4])[0]
                vec = np.frombuffer(blob[4:4 + 4 * n], dtype=np.float32)
                embs.append(vec)
                ids.append(row["memory_id"])
            
            embs = np.stack(embs)
            query_unit = query_vec / np.linalg.norm(query_vec)
            norms = np.linalg.norm(embs, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            embs_unit = embs / norms[:, None]
            sims = embs_unit @ query_unit
            
            # Top-k
            if k >= len(sims):
                top_idx = np.argsort(-sims)
            else:
                top_idx = np.argpartition(-sims, k)[:k]
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
                    results.append({
                        "memory_id": mid,
                        "content": meta["content"],
                        "agent": meta["agent"],
                        "block_type": meta["block_type"],
                        "label": meta["label"],
                        "priority": meta["priority"],
                        "score": float(sims[idx]),
                        "created_at": meta["created_at"],
                        "project": meta["project"],
                    })
                    if len(results) >= k:
                        break
            return results
        
        # Execute sqlite-vec query
        rows = conn.execute(sql, params).fetchall()
        
        results = []
        for row in rows:
            results.append({
                "memory_id": row["memory_id"],
                "content": row["content"],
                "agent": row["agent"],
                "block_type": row["block_type"],
                "label": row["label"],
                "priority": row["priority"],
                "score": 1.0 - row["distance"],  # Convert distance to similarity
                "created_at": row["created_at"],
                "project": row["project"],
            })
            if len(results) >= k:
                break
        
        return results
    
    def count(self) -> int:
        """Get total number of memories."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
        return row["cnt"]
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        conn = self._get_conn()
        conn.execute("DELETE FROM memories WHERE memory_id = ?", [memory_id])
        if HAS_VEC:
            conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", [memory_id])
        else:
            conn.execute("DELETE FROM embeddings_brute WHERE memory_id = ?", [memory_id])
        conn.commit()
        return True
    
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics for this project."""
        conn = self._get_conn()
        
        # Total memories
        total_row = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
        total = total_row["cnt"] if total_row else 0
        
        # Count by block_type
        by_block_type = {}
        try:
            rows = conn.execute(
                "SELECT block_type, COUNT(*) as cnt FROM memories GROUP BY block_type"
            ).fetchall()
            for row in rows:
                by_block_type[row["block_type"]] = row["cnt"]
        except Exception:
            pass
        
        # Count by agent
        by_agent = {}
        try:
            rows = conn.execute(
                "SELECT agent, COUNT(*) as cnt FROM memories GROUP BY agent ORDER BY cnt DESC"
            ).fetchall()
            for row in rows:
                by_agent[row["agent"]] = row["cnt"]
        except Exception:
            pass
        
        # Count by project
        by_project = {}
        try:
            rows = conn.execute(
                "SELECT project, COUNT(*) as cnt FROM memories GROUP BY project ORDER BY cnt DESC"
            ).fetchall()
            for row in rows:
                by_project[row["project"]] = row["cnt"]
        except Exception:
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
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


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

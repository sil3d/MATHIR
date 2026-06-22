#!/usr/bin/env python3
"""
MATHIR Database Schema Migration - Standardize all DBs to Mycerise V2 schema.

Idempotent: safe to run multiple times.

Target schema (from Mycerise V2):
  - memories: memory_id, modality, embedding, embedding_dim, metadata, modality_text, timestamp, tier, stability, recall_count, provider, model
  - memories_fts: FTS5 on (memory_id UNINDEXED, modality_text)
  - memory_embeddings: id, memory_id, provider, model, embedding, embedding_dim, created_at
  - memory_links: source_id, target_id, weight, created_at
  - vec_memories: vec0(memory_id TEXT PRIMARY KEY, embedding FLOAT[384] distance_metric=cosine)
  - Indexes on memories(modality, tier, timestamp), memory_embeddings(memory_id, provider), memory_links(source_id, target_id)
"""
import sqlite3
import os
import sys
import struct
import json
import shutil
from pathlib import Path
from datetime import datetime

# --- Configuration ---
TARGET_VEC_DIM = 384

DB_PATHS = [
    ("Mycerise V2",    r"C:\Users\So-i-learn-3D\Desktop\SECRET_CODE\Mycerise_V2_Taur\.mathir\mathir.db"),
    ("OpenCode Home",  r"C:\Users\So-i-learn-3D\.config\opencode\data\mathir.db"),
    ("OpenCode Project", r"C:\Users\So-i-learn-3D\.config\opencode\data\projects\mathir_08ee1e64\mathir.db"),
    ("OpenCode Config", r"C:\Users\So-i-learn-3D\.config\opencode\.mathir\mathir.db"),
]


def log(msg, level="INFO"):
    print(f"  [{level}] {msg}")


def backup_db(db_path):
    """Create a .bak backup before migration."""
    bak_path = db_path + ".schema_bak"
    if not os.path.exists(bak_path):
        shutil.copy2(db_path, bak_path)
        log(f"Backup created: {bak_path}")
    else:
        log(f"Backup already exists: {bak_path}")


def load_vec_extension(conn):
    """Try to load sqlite-vec extension. Returns True if successful."""
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        log("sqlite-vec extension loaded successfully")
        return True
    except Exception as e:
        log(f"sqlite-vec extension not available: {e}", "WARN")
        return False


def ensure_memories_table(conn):
    """Ensure memories table exists with the correct schema."""
    cursor = conn.execute("PRAGMA table_info(memories)")
    columns = {col[1]: col[2] for col in cursor.fetchall()}
    
    if not columns:
        # Create from scratch
        conn.execute("""
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
            )
        """)
        log("Created memories table from scratch")
        return True
    
    # Table exists - check for missing columns
    required_cols = {
        'memory_id', 'modality', 'embedding', 'embedding_dim', 'metadata',
        'modality_text', 'timestamp', 'tier', 'stability', 'recall_count',
        'provider', 'model'
    }
    missing = required_cols - set(columns.keys())
    
    # Handle old 'content' column -> rename to 'modality_text'
    if 'content' in columns and 'modality_text' not in columns:
        try:
            conn.execute("ALTER TABLE memories RENAME COLUMN content TO modality_text")
            log("Renamed 'content' column to 'modality_text'")
            return True
        except Exception as e:
            log(f"Could not rename content->modality_text: {e}", "WARN")
            missing.discard('modality_text')
    
    # Add missing columns
    for col in missing:
        type_map = {
            'modality': 'TEXT NOT NULL DEFAULT "text"',
            'embedding': 'BLOB',
            'embedding_dim': 'INTEGER',
            'metadata': 'TEXT',
            'modality_text': 'TEXT',
            'timestamp': 'REAL',
            'tier': 'TEXT',
            'stability': 'REAL DEFAULT 1.0',
            'recall_count': 'INTEGER DEFAULT 0',
            'provider': "TEXT DEFAULT 'unknown'",
            'model': "TEXT DEFAULT 'unknown'",
        }
        col_def = type_map.get(col, 'TEXT')
        try:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {col_def}")
            log(f"Added missing column: {col} ({col_def})")
        except Exception as e:
            log(f"Column {col} already exists or error: {e}", "WARN")
    
    return True


def ensure_fts(conn):
    """Ensure FTS5 virtual table exists."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'")
    if cursor.fetchone():
        log("memories_fts already exists")
        return True
    
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id UNINDEXED,
                modality_text,
                tokenize = 'porter unicode61'
            )
        """)
        log("Created memories_fts FTS5 table")
    except Exception as e:
        log(f"Could not create FTS table: {e}", "WARN")
    return True


def ensure_memory_embeddings(conn):
    """Ensure memory_embeddings table exists."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'")
    if cursor.fetchone():
        log("memory_embeddings already exists")
        return True
    
    conn.execute("""
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
        )
    """)
    log("Created memory_embeddings table")
    return True


def ensure_memory_links(conn):
    """Ensure memory_links table exists."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_links'")
    if cursor.fetchone():
        log("memory_links already exists")
        return True
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_links (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            weight REAL NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (source_id, target_id)
        )
    """)
    log("Created memory_links table")
    return True


def ensure_vec_memories(conn, has_vec):
    """Ensure vec_memories vec0 table exists with correct dimension."""
    if not has_vec:
        log("sqlite-vec not available, skipping vec_memories", "WARN")
        return True
    
    # Check if vec_memories exists and get its dimension
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_memories'")
    row = cursor.fetchone()
    
    if row:
        sql = row[0]
        # Check dimension
        if f'FLOAT[{TARGET_VEC_DIM}]' in sql:
            log(f"vec_memories already correct (FLOAT[{TARGET_VEC_DIM}])")
            return True
        else:
            # Dimension mismatch - need to recreate
            log(f"vec_memories has wrong dimension, recreating...")
            
            # Extract existing data to re-insert later
            try:
                cursor.execute("SELECT memory_id, embedding FROM vec_memories")
                existing_data = cursor.fetchall()
                log(f"Found {len(existing_data)} existing vectors to migrate")
            except Exception:
                existing_data = []
            
            # Drop and recreate
            conn.execute("DROP TABLE IF EXISTS vec_memories_chunks")
            conn.execute("DROP TABLE IF EXISTS vec_memories_info")
            conn.execute("DROP TABLE IF EXISTS vec_memories_rowids")
            conn.execute("DROP TABLE IF EXISTS vec_memories_vector_chunks00")
            conn.execute("DROP TABLE IF EXISTS vec_memories")
            
            conn.execute(f"""
                CREATE VIRTUAL TABLE vec_memories USING vec0(
                    memory_id TEXT PRIMARY KEY,
                    embedding FLOAT[{TARGET_VEC_DIM}] distance_metric=cosine
                )
            """)
            log(f"Recreated vec_memories with FLOAT[{TARGET_VEC_DIM}]")
            
            # Re-insert existing vectors (only if they match the target dimension)
            reinserted = 0
            for mid, emb_blob in existing_data:
                if emb_blob and len(emb_blob) == TARGET_VEC_DIM * 4:
                    try:
                        conn.execute("INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)", [mid, emb_blob])
                        reinserted += 1
                    except Exception:
                        pass
            log(f"Re-inserted {reinserted}/{len(existing_data)} vectors")
            return True
    else:
        # Create from scratch
        conn.execute(f"""
            CREATE VIRTUAL TABLE vec_memories USING vec0(
                memory_id TEXT PRIMARY KEY,
                embedding FLOAT[{TARGET_VEC_DIM}] distance_metric=cosine
            )
        """)
        log(f"Created vec_memories with FLOAT[{TARGET_VEC_DIM}]")
        return True


def ensure_indexes(conn):
    """Ensure all standard indexes exist."""
    indexes = [
        ("idx_memories_modality",   "CREATE INDEX IF NOT EXISTS idx_memories_modality  ON memories(modality)"),
        ("idx_memories_tier",       "CREATE INDEX IF NOT EXISTS idx_memories_tier      ON memories(tier)"),
        ("idx_memories_timestamp",  "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)"),
        ("idx_embeddings_memory_id","CREATE INDEX IF NOT EXISTS idx_embeddings_memory_id ON memory_embeddings(memory_id)"),
        ("idx_embeddings_provider", "CREATE INDEX IF NOT EXISTS idx_embeddings_provider ON memory_embeddings(provider)"),
        ("idx_links_source",        "CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id)"),
        ("idx_links_target",        "CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id)"),
    ]
    for name, sql in indexes:
        try:
            conn.execute(sql)
        except Exception as e:
            log(f"Index {name}: {e}", "WARN")
    log("All indexes ensured")


def sync_fts_from_memories(conn):
    """Rebuild FTS index from memories table if FTS is empty but memories has data."""
    try:
        cur = conn.execute("SELECT COUNT(*) FROM memories_fts")
        fts_count = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM memories WHERE modality_text IS NOT NULL AND modality_text != ''")
        mem_count = cur.fetchone()[0]
        
        if fts_count == 0 and mem_count > 0:
            log(f"FTS empty but memories has {mem_count} rows with text, rebuilding FTS...")
            conn.execute("DELETE FROM memories_fts")
            conn.execute("""
                INSERT INTO memories_fts(memory_id, modality_text)
                SELECT memory_id, modality_text FROM memories
                WHERE modality_text IS NOT NULL AND modality_text != ''
            """)
            conn.commit()
            log(f"FTS rebuilt with {mem_count} entries")
        else:
            log(f"FTS sync not needed (fts={fts_count}, memories_with_text={mem_count})")
    except Exception as e:
        log(f"FTS sync error: {e}", "WARN")


def get_schema_summary(conn):
    """Return a summary of the schema state."""
    summary = {}
    
    # Tables
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    summary['tables'] = [r[0] for r in cur.fetchall()]
    
    # memories columns
    try:
        cur = conn.execute("PRAGMA table_info(memories)")
        summary['memories_columns'] = [col[1] for col in cur.fetchall()]
    except:
        summary['memories_columns'] = []
    
    # Row counts
    summary['row_counts'] = {}
    for table in summary['tables']:
        try:
            cur = conn.execute(f"SELECT COUNT(*) FROM [{table}]")
            summary['row_counts'][table] = cur.fetchone()[0]
        except:
            summary['row_counts'][table] = 'ERROR'
    
    # vec_memories dimension
    try:
        cur = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_memories'")
        row = cur.fetchone()
        if row:
            import re
            match = re.search(r'FLOAT\[(\d+)\]', row[0])
            summary['vec_dim'] = int(match.group(1)) if match else 'unknown'
        else:
            summary['vec_dim'] = 'MISSING'
    except:
        summary['vec_dim'] = 'error'
    
    return summary


def migrate_db(label, db_path):
    """Run full migration on a single database."""
    print(f"\n{'='*70}")
    print(f"  MIGRATING: {label}")
    print(f"  Path: {db_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(db_path):
        log(f"Database not found: {db_path}", "ERROR")
        return False
    
    # Show pre-migration state
    conn = sqlite3.connect(db_path)
    pre_summary = get_schema_summary(conn)
    log(f"Pre-migration: {len(pre_summary['tables'])} tables, vec_dim={pre_summary['vec_dim']}")
    log(f"Row counts: {pre_summary['row_counts']}")
    
    # Backup
    backup_db(db_path)
    
    # Load vec extension
    has_vec = load_vec_extension(conn)
    
    # Run all migrations
    ensure_memories_table(conn)
    ensure_fts(conn)
    ensure_memory_embeddings(conn)
    ensure_memory_links(conn)
    ensure_vec_memories(conn, has_vec)
    ensure_indexes(conn)
    sync_fts_from_memories(conn)
    
    conn.commit()
    
    # Show post-migration state
    post_summary = get_schema_summary(conn)
    log(f"Post-migration: {len(post_summary['tables'])} tables, vec_dim={post_summary['vec_dim']}")
    log(f"Row counts: {post_summary['row_counts']}")
    
    # Check for issues
    issues = []
    if 'modality_text' not in post_summary.get('memories_columns', []):
        issues.append("memories table missing modality_text column")
    if post_summary['vec_dim'] != TARGET_VEC_DIM and post_summary['vec_dim'] != 'MISSING':
        issues.append(f"vec_memories dimension is {post_summary['vec_dim']}, expected {TARGET_VEC_DIM}")
    
    required_tables = {'memories', 'memory_embeddings', 'memory_links'}
    missing_tables = required_tables - set(post_summary['tables'])
    if missing_tables:
        issues.append(f"Missing required tables: {missing_tables}")
    
    if issues:
        log(f"ISSUES: {issues}", "WARN")
    else:
        log("Migration complete - all checks passed")
    
    conn.close()
    return len(issues) == 0


def verify_all_databases():
    """Verify all databases have identical schemas."""
    print(f"\n{'='*70}")
    print(f"  VERIFICATION: Comparing all database schemas")
    print(f"{'='*70}")
    
    summaries = {}
    for label, db_path in DB_PATHS:
        if not os.path.exists(db_path):
            continue
        conn = sqlite3.connect(db_path)
        load_vec_extension(conn)
        summaries[label] = get_schema_summary(conn)
        conn.close()
    
    # Compare key attributes
    all_ok = True
    for label, s in summaries.items():
        issues = []
        if 'modality_text' not in s.get('memories_columns', []):
            issues.append("missing modality_text")
        if s['vec_dim'] != TARGET_VEC_DIM:
            issues.append(f"vec_dim={s['vec_dim']}")
        required = {'memories', 'memory_embeddings', 'memory_links'}
        missing = required - set(s['tables'])
        if missing:
            issues.append(f"missing tables: {missing}")
        
        status = "OK" if not issues else f"ISSUES: {issues}"
        print(f"  {label}: {status}")
        if issues:
            all_ok = False
    
    return all_ok


if __name__ == "__main__":
    print("MATHIR Database Schema Migration")
    print(f"Target: standardize all DBs to match Mycerise V2 schema")
    print(f"Target vec dimension: {TARGET_VEC_DIM}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Migrate each database
    results = {}
    for label, db_path in DB_PATHS:
        results[label] = migrate_db(label, db_path)
    
    # Verify
    verify_ok = verify_all_databases()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  MIGRATION SUMMARY")
    print(f"{'='*70}")
    for label, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {label}: {status}")
    print(f"  Verification: {'PASS' if verify_ok else 'FAIL'}")
    print(f"{'='*70}")

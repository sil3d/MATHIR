"""
MATHIR Spreading Activation — Phase 3
=====================================
When a memory is recalled, automatically retrieve related memories via the link graph.

Schema: memory_links table
- source_id TEXT (memory_id)
- target_id TEXT (memory_id)
- weight REAL (similarity 0-1)
- created_at REAL

Concept (Collins & Loftus 1975):
- Recall "Tauri" → activates "Tauri" with strength 1.0
- Spreads to linked memories "Rust" (0.7), "IPC" (0.5), "Desktop" (0.6)
- Each linked memory contributes its score × link weight
- After 2-3 hops, you have a context-rich recall

This is what makes MATHIR brain-like: recalling one thing brings related things.
"""
import sys
import time
import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))
from mathir_vec import VecMemory
from mathir_mcp_server import get_project_db_path, get_project_name


# --- C8/H7 schema detection -------------------------------------------------
# These spreading-activation helpers are best-effort features over the legacy
# `memories` schema (which has direct `embedding` / `modality_text` / `block_type`
# columns). Vec-era ("new" schema) DBs store that data in `vec_memories` + a
# JSON `metadata` blob instead, so the legacy SQL below would raise
# sqlite3.OperationalError. We detect the schema once per DB path and degrade
# gracefully (empty result / 0) on new-schema DBs.
_SCHEMA_WARNED: set = set()  # db paths we've already printed the one-time warning for


def _schema_kind(db_path: Path) -> str:
    """Return ``"legacy"`` or ``"new"`` based on the memories-table columns.

    Mirrors ``VecMemory._schema_kind``: "new" means a ``content`` column exists,
    "legacy" means the old ``embedding`` / ``modality_text`` / ``block_type``
    BLOB/column layout is present.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        cols = {col[1] for col in conn.execute("PRAGMA table_info(memories)").fetchall()}
        conn.close()
    except sqlite3.OperationalError:
        return "legacy"  # table missing — let the caller's own SQL surface any error
    return "new" if "content" in cols else "legacy"


def _warn_new_schema(db_path: Path) -> None:
    """Print a one-time warning that this module is a no-op on new-schema DBs."""
    key = str(db_path)
    if key not in _SCHEMA_WARNED:
        _SCHEMA_WARNED.add(key)
        print(f"[mathir_spread] schema is 'new' for {db_path.name}; "
              "spreading-activation is legacy-only, degrading to no-op.")
# ---------------------------------------------------------------------------


def ensure_links_table(db_path: Path):
    """Create memory_links table if not exists."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_links (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            weight REAL NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (source_id, target_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id)
    """)
    conn.commit()
    conn.close()


def build_links_for_memory(db_path: Path, memory_id: str, threshold: float = 0.65, max_links: int = 10):
    """
    Find related memories via vector similarity and create links.
    Threshold: minimum cosine similarity to consider related.
    """
    # C8/H7: legacy-only feature — no-op on vec-era DBs.
    if _schema_kind(db_path) == "new":
        _warn_new_schema(db_path)
        return 0
    ensure_links_table(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    # Get the source memory
    cur = conn.execute("SELECT embedding FROM memories WHERE memory_id = ?", (memory_id,))
    row = cur.fetchone()
    if not row or not row['embedding']:
        conn.close()
        return 0
    
    source_emb = np.frombuffer(row['embedding'], dtype=np.float32)
    
    # Find all other memories with embeddings
    cur = conn.execute("""
        SELECT memory_id, embedding FROM memories 
        WHERE memory_id != ? AND embedding IS NOT NULL
    """, (memory_id,))
    
    candidates = []
    for r in cur.fetchall():
        try:
            target_emb = np.frombuffer(r['embedding'], dtype=np.float32)
            if len(target_emb) != len(source_emb):
                continue
            sim = float(np.dot(source_emb, target_emb) / (np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8))
            if sim >= threshold:
                candidates.append((r['memory_id'], sim))
        except Exception:
            continue
    
    # Keep top-K
    candidates.sort(key=lambda x: -x[1])
    candidates = candidates[:max_links]
    
    # Insert links (bidirectional).
    # NOTE (H8): These weights are intentionally ASYMMETRIC — the forward edge
    # source->target keeps `weight`, the reverse edge target->source is damped
    # to `weight * 0.9`. This diverges from mathir_vec.build_links_all, which
    # uses symmetric weights. Do not "fix" the asymmetry here without also
    # reconciling the two modules — preserving current behavior on purpose.
    now = time.time()
    for target_id, weight in candidates:
        conn.execute("""
            INSERT OR REPLACE INTO memory_links (source_id, target_id, weight, created_at)
            VALUES (?, ?, ?, ?)
        """, (memory_id, target_id, weight, now))
        # Reverse link with 0.9x weight (asymmetric: source -> target with full, target -> source with damped)
        conn.execute("""
            INSERT OR REPLACE INTO memory_links (source_id, target_id, weight, created_at)
            VALUES (?, ?, ?, ?)
        """, (target_id, memory_id, weight * 0.9, now))
    
    conn.commit()
    conn.close()
    return len(candidates)


def spread_recall(db_path: Path, initial_results: List[Dict], 
                  hops: int = 2, 
                  decay: float = 0.5,
                  max_results: int = 10) -> List[Dict]:
    """
    Spreading activation: take initial vector search results, then follow links.
    
    Each hop: for each memory in current activation, find its links, add to activation with weight × decay.
    After `hops` hops, return top `max_results` memories by activation.
    
    Args:
        db_path: SQLite database path
        initial_results: First-pass vector search results (each has 'memory_id' and 'score')
        hops: Number of spreading iterations (1-3 typical)
        decay: Multiplicative factor per hop (0.5 means each hop halves the activation)
        max_results: Final cap on returned memories
    
    Returns:
        Combined list of memories with boosted scores
    """
    # C8/H7: legacy-only feature — no-op on vec-era DBs.
    if _schema_kind(db_path) == "new":
        _warn_new_schema(db_path)
        return []
    ensure_links_table(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    # Activation: memory_id -> score
    activation: Dict[str, float] = {}
    for r in initial_results:
        mid = r.get('memory_id')
        if mid:
            activation[mid] = max(activation.get(mid, 0), r.get('score', 0))
    
    # Spreading
    for hop in range(hops):
        new_activation = dict(activation)
        decay_factor = decay ** (hop + 1)
        for mid, score in list(activation.items()):
            # Get outgoing links
            cur = conn.execute("""
                SELECT target_id, weight FROM memory_links
                WHERE source_id = ? AND weight > 0.3
                ORDER BY weight DESC LIMIT 5
            """, (mid,))
            for link in cur.fetchall():
                target_id = link['target_id']
                link_weight = link['weight']
                contribution = score * link_weight * decay_factor
                if contribution > 0.05:  # Minimum threshold
                    new_activation[target_id] = max(
                        new_activation.get(target_id, 0),
                        contribution
                    )
        activation = new_activation
    
    # Get top results
    sorted_mems = sorted(activation.items(), key=lambda x: -x[1])[:max_results]
    
    # Hydrate with content
    results = []
    for mid, score in sorted_mems:
        cur = conn.execute("""
            SELECT memory_id, modality_text, metadata, agent, block_type, label
            FROM memories WHERE memory_id = ?
        """, (mid,))
        row = cur.fetchone()
        if row:
            results.append({
                'memory_id': row['memory_id'],
                'content': row['modality_text'] or '',
                'score': float(score),
                'agent': row['agent'],
                'block_type': row['block_type'],
                'label': row['label'],
            })
    
    conn.close()
    return results


def build_links_for_all(db_path: Path, threshold: float = 0.7, batch_size: int = 50):
    """
    Background job: build links for all memories in DB.
    Use during 'sleep' consolidation.
    """
    # C8/H7: legacy-only feature — no-op on vec-era DBs.
    if _schema_kind(db_path) == "new":
        _warn_new_schema(db_path)
        return 0
    ensure_links_table(db_path)
    conn = sqlite3.connect(str(db_path))
    
    # Get all memory IDs
    cur = conn.execute("SELECT memory_id FROM memories WHERE modality_text IS NOT NULL")
    memory_ids = [r[0] for r in cur.fetchall()]
    conn.close()
    
    log_total = len(memory_ids)
    log_done = 0
    
    for mid in memory_ids:
        try:
            n = build_links_for_memory(db_path, mid, threshold=threshold)
            log_done += 1
            if log_done % 10 == 0:
                print(f"  Links built: {log_done}/{log_total} (last: {n} links)")
        except Exception as e:
            print(f"  Error building links for {mid}: {e}")
    
    return log_done


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python mathir_spread.py build_all    # Build links for all memories in current project")
        print("  python mathir_spread.py build <id>   # Build links for one memory")
        sys.exit(1)
    
    db_path = get_project_db_path()
    print(f"DB: {db_path}")
    
    if sys.argv[1] == 'build_all':
        print("Building links for all memories...")
        n = build_links_for_all(db_path)
        print(f"Done. {n} memories processed.")
    elif sys.argv[1] == 'build' and len(sys.argv) >= 3:
        n = build_links_for_memory(db_path, sys.argv[2])
        print(f"Built {n} links for {sys.argv[2]}")
    else:
        print("Unknown command")

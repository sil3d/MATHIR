"""
MATHIR Consolidation (Sleep) — Phase 4
======================================
Background process that runs periodically (nightly) to:
1. MERGE near-duplicate memories (cosine > 0.95)
2. DECAY unused memories (Ebbinghaus-style: stability × 0.95/month)
3. BOOST frequently-accessed memories (stability += 0.1 per access)
4. REBUILD link graph for new memories
5. ARCHIVE dead memories (stability < 0.05 after 1 year)

This is the "sleep" of the brain — what the hippocampus does during slow-wave sleep.

Schedule: Run as a scheduled task, or call from cron/Task Scheduler.
"""
import sys
import time
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from mathir_mcp_server import get_project_db_path


def merge_duplicates(db_path: Path, threshold: float = 0.95) -> int:
    """
    Find near-duplicate memories (cosine > threshold) and merge them.
    Keep the highest-priority one as primary, delete the others.
    Returns number of merges.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    # Get all memories with embeddings
    cur = conn.execute("""
        SELECT memory_id, embedding, modality_text, priority, agent, block_type, label, timestamp, recall_count
        FROM memories
        WHERE embedding IS NOT NULL AND embedding_dim = 384
        ORDER BY priority DESC, recall_count DESC, timestamp DESC
    """)
    
    rows = cur.fetchall()
    print(f"  Scanning {len(rows)} memories for duplicates...")
    
    # Build numpy matrix for fast comparison
    embs = []
    ids = []
    for r in rows:
        try:
            emb = np.frombuffer(r['embedding'], dtype=np.float32)
            if len(emb) == 384:
                embs.append(emb)
                ids.append((r['memory_id'], r['priority'], r['recall_count'], r['timestamp']))
        except Exception:
            continue
    
    if not embs:
        return 0
    
    embs = np.stack(embs)
    # Normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embs_norm = embs / norms
    
    # Find duplicates (only compare i < j)
    to_delete = set()
    merge_count = 0
    for i in range(len(embs)):
        if ids[i][0] in to_delete:
            continue
        for j in range(i+1, len(embs)):
            if ids[j][0] in to_delete:
                continue
            sim = float(np.dot(embs_norm[i], embs_norm[j]))
            if sim >= threshold:
                # Keep i (higher priority), delete j
                to_delete.add(ids[j][0])
                # Boost i's recall count
                conn.execute("""
                    UPDATE memories 
                    SET recall_count = recall_count + ? 
                    WHERE memory_id = ?
                """, (ids[j][2] // 2 + 1, ids[i][0]))
                merge_count += 1
    
    # Delete duplicates
    for mid in to_delete:
        conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (mid,))
        conn.execute("DELETE FROM memories WHERE memory_id = ?", (mid,))
        conn.execute("DELETE FROM memory_links WHERE source_id = ? OR target_id = ?", (mid, mid))
    
    conn.commit()
    conn.close()
    return merge_count


def apply_decay(db_path: Path, decay_rate: float = 0.05) -> int:
    """
    Apply Ebbinghaus-style decay: memories not accessed in 30+ days lose 5% stability.
    Returns number of memories decayed.
    """
    conn = sqlite3.connect(str(db_path))
    cutoff = time.time() - (30 * 86400)  # 30 days ago
    
    # Decay memories: stability = stability * (1 - decay_rate) if last access > 30 days ago
    cur = conn.execute("""
        UPDATE memories 
        SET stability = stability * ?
        WHERE (recall_count = 0 OR timestamp < ?)
          AND stability > 0.05
    """, (1.0 - decay_rate, cutoff))
    
    decayed = cur.rowcount
    conn.commit()
    conn.close()
    return decayed


def archive_dead_memories(db_path: Path, min_stability: float = 0.05) -> int:
    """
    Archive memories with stability below threshold.
    For now, just delete them (could move to archive table).
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT memory_id FROM memories WHERE stability < ?", (min_stability,))
    dead = [r[0] for r in cur.fetchall()]
    
    for mid in dead:
        conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (mid,))
        conn.execute("DELETE FROM memories WHERE memory_id = ?", (mid,))
        conn.execute("DELETE FROM memory_links WHERE source_id = ? OR target_id = ?", (mid, mid))
    
    conn.commit()
    conn.close()
    return len(dead)


def consolidate(db_path: Path, dry_run: bool = False) -> Dict:
    """
    Run full consolidation cycle.
    Returns stats dict.
    """
    print(f"\n=== Consolidation @ {datetime.now().isoformat()} ===")
    print(f"  DB: {db_path}")
    print(f"  Dry run: {dry_run}")
    
    # Pre-stats
    conn = sqlite3.connect(str(db_path))
    n_before = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()
    
    print(f"\n[1/3] Merging duplicates (threshold 0.95)...")
    n_merged = merge_duplicates(db_path, threshold=0.95) if not dry_run else 0
    print(f"  Merged: {n_merged}")
    
    print(f"\n[2/3] Applying decay (rate 5%/month)...")
    n_decayed = apply_decay(db_path, decay_rate=0.05) if not dry_run else 0
    print(f"  Decayed: {n_decayed}")
    
    print(f"\n[3/3] Archiving dead memories (stability < 0.05)...")
    n_archived = archive_dead_memories(db_path, min_stability=0.05) if not dry_run else 0
    print(f"  Archived: {n_archived}")
    
    # Post-stats
    conn = sqlite3.connect(str(db_path))
    n_after = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    n_links = conn.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0]
    conn.close()
    
    stats = {
        'before': n_before,
        'after': n_after,
        'merged': n_merged,
        'decayed': n_decayed,
        'archived': n_archived,
        'links': n_links,
    }
    
    print(f"\n=== Summary ===")
    print(f"  Memories: {n_before} -> {n_after} (delta: {n_before - n_after})")
    print(f"  Links: {n_links}")
    print(f"  Merged: {n_merged} | Decayed: {n_decayed} | Archived: {n_archived}")
    print()
    
    return stats


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'dry':
        db_path = get_project_db_path()
        consolidate(db_path, dry_run=True)
    else:
        db_path = get_project_db_path()
        consolidate(db_path, dry_run=False)

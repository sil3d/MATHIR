#!/usr/bin/env python3
"""Migrate existing memory_blocks from memory.db to MATHIR."""

import json
import os
import sqlite3
import sys
from pathlib import Path

MEMORY_DB = Path(os.path.expanduser("~/.config/opencode/data/memory.db"))
MATHIR_DB = Path(os.path.expanduser("~/.config/opencode/data/mathir.db"))
CONFIG_PATH = Path(os.path.expanduser("~/.config/opencode/config/mathir.json"))


def migrate():
    print(f"Reading from: {MEMORY_DB}")
    print(f"Writing to: {MATHIR_DB}")

    if not MEMORY_DB.exists():
        print("No memory.db found — nothing to migrate.")
        return

    # Read all memory blocks
    old_conn = sqlite3.connect(str(MEMORY_DB))
    old_conn.row_factory = sqlite3.Row
    rows = old_conn.execute("SELECT * FROM memory_blocks ORDER BY id").fetchall()
    print(f"Found {len(rows)} memory blocks to migrate")

    if not rows:
        print("Nothing to migrate.")
        old_conn.close()
        return

    # Initialize MATHIR
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from mathir_dropin.memory import MATHIRMemory

    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = json.load(f)

    db_path = str(MATHIR_DB)
    MATHIR_DB.parent.mkdir(parents=True, exist_ok=True)

    embedding_dim = int(os.environ.get("MATHIR_EMBEDDING_DIM", "1024"))
    memory = MATHIRMemory(
        embedding_dim=embedding_dim,
        config=config,
        db_path=db_path,
        provider="migration",
        model=config.get("embedding", {}).get("model", "BAAI/bge-large-en-v1.5"),
    )

    # Load embedder
    from sentence_transformers import SentenceTransformer
    model_name = config.get("embedding", {}).get("model", "BAAI/bge-large-en-v1.5")
    embedder = SentenceTransformer(model_name)

    # Migrate each block
    migrated = 0
    for row in rows:
        row = dict(row)
        content = row["content"]
        if not content or not content.strip():
            continue

        embedding = embedder.encode(content, convert_to_tensor=True).unsqueeze(0).clone()

        block_type = row["block_type"]
        tier_map = {
            "working_memory": "working",
            "episodic": "episodic",
            "semantic": "semantic",
            "procedural": "semantic",
        }
        tier = tier_map.get(block_type, "episodic")

        metadata = {
            "agent": row["agent_name"],
            "block_type": block_type,
            "label": row["label"],
            "priority": row["priority"],
            "content": content,
            "migrated_from": "memory.db",
            "old_id": row["id"],
        }

        memory_id = memory.store(
            embedding=embedding,
            metadata=metadata,
            tier=tier,
            provider="migration",
            model=model_name,
        )

        print(f"  Migrated: [{row['agent_name']}/{block_type}] {row['label']} -> {memory_id}")
        migrated += 1

    old_conn.close()
    print(f"\nMigration complete: {migrated}/{len(rows)} blocks migrated")
    print(f"MATHIR DB: {MATHIR_DB}")
    if MATHIR_DB.exists():
        print(f"DB size: {MATHIR_DB.stat().st_size} bytes")


if __name__ == "__main__":
    migrate()

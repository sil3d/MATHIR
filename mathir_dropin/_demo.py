"""
MATHIR Drop-in — Runnable demo.

Run with::

    python -m mathir_dropin._demo

or::

    cd mathir_dropin && python _demo.py

It will:
    1. Create an in-memory + SQLite-backed MATHIRMemory.
    2. Store 10 random "memories" with text metadata.
    3. Recall similar ones by embedding.
    4. Run a FTS5 text search.
    5. Print stats.
    6. Reopen the DB in a new instance and prove persistence.
    7. Clean up the demo DB file.

No network, no GPU strictly required (works on CPU).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

# Force UTF-8 on stdout/stderr when possible so the emoji markers
# (``✅``, ``❌``, etc.) survive on Windows consoles that default
# to cp1252. Falls back silently to the locale default if the
# reconfigure call is unsupported (e.g. on some embedded Pythons).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

import torch

try:
    from . import MATHIRMemory, configure
except ImportError:
    # Allow running as `python _demo.py` from inside the mathir_dropin dir
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from __init__ import MATHIRMemory, configure  # type: ignore


def _header(s: str) -> None:
    print("\n" + "=" * 60)
    print(s)
    print("=" * 60)


def main() -> int:
    # Use a temp file so the demo is self-cleaning.
    tmpdir = tempfile.mkdtemp(prefix="mathir_demo_")
    db_path = os.path.join(tmpdir, "demo.db")
    print(f"Demo database: {db_path}")

    try:
        # ----- Step 1: configure -------------------------------------
        _header("1. Configuration")
        cfg = configure({
            "memory": {
                "embedding_dim": 384,
                "working_capacity": 32,
                "episodic_capacity": 256,
            },
            "storage": {"db_path": db_path, "auto_save": True},
        })
        print("Config keys:", list(cfg.keys()))
        print("Memory tier capacities:",
              {k: cfg["memory"][k] for k in
               ("working_capacity", "episodic_capacity",
                "semantic_prototypes", "immunological_capacity")})

        # ----- Step 2: create memory ---------------------------------
        _header("2. Create MATHIRMemory (1 line)")
        memory = MATHIRMemory(embedding_dim=384, config=cfg, db_path=db_path)
        print("✅ Memory created")
        print("   embedding_dim =", memory.embedding_dim)
        print("   storage       =", memory.get_stats()["storage"])

        # ----- Step 3: perceive() ------------------------------------
        _header("3. perceive() — 4-tier routing")
        emb = torch.randn(2, 384)
        out = memory.perceive(emb, metadata={"text": "first perception"})
        print("✅ perceive() returned keys:", list(out.keys()))
        print("   enhanced_embedding.shape =", tuple(out["enhanced_embedding"].shape))
        print("   router_weights[0]        =", out["router_weights"][0].tolist())
        print("   anomaly_score[0]         =", float(out["anomaly_score"][0]))
        print("   modality                 =", out["modality"])

        # ----- Step 4: store() ---------------------------------------
        _header("4. store() — 10 memories with text")
        torch.manual_seed(0)  # reproducible
        ids = []
        for i in range(10):
            e = torch.randn(1, 384)
            md = {
                "user": "alice",
                "turn": i,
                "text": (
                    f"Message {i}: the quick brown fox jumps over the lazy dog. "
                    f"Topic is {'transformer' if i % 2 else 'convolution'}."
                ),
            }
            mid = memory.store(e, md)
            ids.append(mid)
            print(f"  stored {mid}  text={md['text'][:50]!r}...")
        print(f"✅ Stored {len(ids)} memories")

        # ----- Step 5: recall() by embedding -------------------------
        _header("5. recall() — semantic similarity")
        q = torch.randn(1, 384)
        hits = memory.recall(q, k=3)
        print(f"✅ Top-{len(hits)} hits for random query:")
        for h in hits:
            text = h["metadata"].get("text", "<no text>")[:50]
            print(f"   {h['memory_id']}  sim={h['similarity']:+.3f}  text={text!r}")

        # ----- Step 6: recall_text() via FTS5 ------------------------
        _header("6. recall_text() — FTS5 BM25 search")
        text_hits = memory.recall_text("transformer", k=3)
        print(f"✅ {len(text_hits)} hits for 'transformer':")
        for h in text_hits:
            text = h.get("modality_text", h.get("metadata", {}).get("text", ""))[:50]
            print(f"   {h['memory_id']}  sim={h['similarity']:+.3f}  text={text!r}")

        # ----- Step 7: get_stats() -----------------------------------
        _header("7. get_stats() — tier usage")
        stats = memory.get_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # ----- Step 8: persistence (reopen) --------------------------
        _header("8. Reopen DB in a fresh MATHIRMemory")
        memory2 = MATHIRMemory(embedding_dim=384, config=cfg, db_path=db_path)
        # Trigger load by rehydrating the in-memory tiers.
        memory2.load()
        print(f"✅ Reopened; episodic usage after load: {memory2.get_stats()['tier_episodic']['usage']}")
        print(f"   SQLite row count:                    {memory2.get_stats()['storage']['row_count']}")

        # ----- Step 9: forget() --------------------------------------
        _header("9. forget() — utility pruning")
        # Threshold high enough to drop something, but the seed is small
        # so we'll just observe a 0-count drop.
        dropped = memory2.forget(threshold=0.99)
        print(f"✅ forget(threshold=0.99) dropped {dropped} memories")

        # ----- Step 10: dimension error ------------------------------
        _header("10. Error handling — DimensionMismatchError")
        try:
            memory2.store(torch.randn(1, 768), {"text": "wrong dim"})
        except Exception as e:
            print(f"✅ Caught {type(e).__name__}: {e}")

        # ----- Step 11: inspect the DB file --------------------------
        _header("11. The database file")
        size = os.path.getsize(db_path)
        print(f"✅ {db_path}  ({size} bytes)")
        print("   You can inspect it with:  sqlite3 " + db_path + " 'SELECT * FROM memories;'")

        # ----- Final --------------------------------------------------
        _header("🎉 Demo complete")
        print("Next steps:")
        print("  • Try memory.recall_text('fox') to see the FTS5 BM25 search.")
        print("  • Open the .db file in DB Browser for SQLite to see the rows.")
        print("  • Read mathir_dropin/README.md for the 5-minute quickstart.")
        return 0

    finally:
        # Clean up the temp file.
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

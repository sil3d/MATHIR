"""
MATHIR SimpleMemory Demo
========================

Zero-dependency memory using SQLite FTS5.
No torch, no sentence_transformers, no external models.

Run:
    python examples/simple_memory_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathir_dropin.simple import SimpleMemory


def main():
    print("=" * 60)
    print("  MATHIR SimpleMemory Demo (FTS5, zero deps)")
    print("=" * 60)

    # 1. Create memory
    print("\n[1] Creating SimpleMemory...")
    mem = SimpleMemory(db_path="demo_memory.db")
    print(f"    OK: {mem}")

    # 2. Store conversations
    print("\n[2] Storing 10 conversation memories...")
    conversations = [
        "User asked about Python closures",
        "Explained that closures capture variables from enclosing scope",
        "User then asked about decorators",
        "Decorators are functions that wrap other functions",
        "User mentioned they work at Google as a software engineer",
        "User prefers Python over JavaScript",
        "User is building an autonomous driving project called MATHIR",
        "User uses PyTorch for the neural network implementation",
        "User has a cat named Schrodinger",
        "User graduated from MIT in 2020",
    ]
    for text in conversations:
        mem.store(text, metadata={"source": "demo"})
    print(f"    Stored {len(conversations)} memories")

    # 3. Recall
    print("\n[3] Recall tests...")
    queries = [
        "What does the user do for work?",
        "What programming language does the user prefer?",
        "What is the user's project called?",
        "What did the user study?",
        "Tell me about the user's pet",
    ]

    for query in queries:
        results = mem.recall(query, k=3)
        print(f"\n  Q: {query}")
        for i, r in enumerate(results, 1):
            print(f"    #{i} [{r['score']:.3f}] {r['text']}")

    # 4. Context injection
    print("\n[4] Context injection (for LLM)...")
    context = mem.search_context("How do decorators work?", k=5, last_n=3)
    print(f"  Context for LLM:\n{context}")

    # 5. Stats
    print("\n[5] Memory stats...")
    stats = mem.get_stats()
    print(f"  Total memories: {stats['total_memories']}")

    # 6. Get last N
    print("\n[6] Last 3 memories...")
    last = mem.get_last(n=3)
    for r in last:
        print(f"  - {r['text']}")

    # Cleanup
    import os
    os.remove("demo_memory.db")
    print("\n" + "=" * 60)
    print("  Demo complete! (demo_memory.db cleaned up)")
    print("=" * 60)


if __name__ == "__main__":
    main()

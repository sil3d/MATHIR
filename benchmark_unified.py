"""
MATHIR Vector Search Benchmark
===============================
Compares all backends against FAISS baseline.
Run: python benchmark_unified.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, r"D:\SECRET_PROJECT\MATHIR")

from mathir_search import VectorSearch


def generate_data(n: int, dim: int):
    """Generate n random embeddings + metadata."""
    np.random.seed(42)
    embeddings = np.random.randn(n, dim).astype("float32")
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    items = []
    agents = ["coder", "debugger", "scout", "qa", "pm"]
    tiers = ["working_memory", "episodic", "semantic", "procedural"]
    for i in range(n):
        items.append({
            "memory_id": f"mem_{i}",
            "embedding": embeddings[i],
            "metadata": {
                "agent": agents[i % len(agents)],
                "tier": tiers[i % len(tiers)],
                "content": f"Test memory {i}",
            },
        })
    return embeddings, items


def bench_insert(search: VectorSearch, items, label: str):
    """Benchmark insert (single + batch)."""
    # Single inserts
    start = time.perf_counter()
    for item in items[:min(100, len(items))]:
        search.store(item["memory_id"], item["embedding"], item["metadata"])
    elapsed = (time.perf_counter() - start) * 1000
    n = min(100, len(items))
    print(f"  {label} single insert ({n}): {elapsed:.1f}ms ({elapsed/n:.2f}ms/item)")

    # Batch insert (remaining)
    remaining = items[n:]
    if remaining:
        search.close()
        if search.backend_name == "sqlite":
            search._backend = __import__("mathir_search", fromlist=["_SQLiteBackend"])._SQLiteBackend(search.dim, search.db_path)
        elif search.backend_name == "gpu":
            search._backend = __import__("mathir_search", fromlist=["_GPUBackend"])._GPUBackend(search.dim)
        else:
            search._backend = __import__("mathir_search", fromlist=["_NumpyBackend"])._NumpyBackend(search.dim)

        start = time.perf_counter()
        search.store_batch(remaining)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  {label} batch insert ({len(remaining)}): {elapsed:.1f}ms ({elapsed/len(remaining):.2f}ms/item)")


def bench_search(search: VectorSearch, queries, k=5, label=""):
    """Benchmark search."""
    # Warmup
    for q in queries[:3]:
        search.search(q, k=k)

    times = []
    for q in queries:
        start = time.perf_counter()
        results = search.search(q, k=k)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"  {label} search k={k}: avg={times.mean():.2f}ms "
          f"p50={np.percentile(times, 50):.2f}ms "
          f"p99={np.percentile(times, 99):.2f}ms "
          f"(n={len(queries)})")
    return times.mean()


def bench_faiss(embeddings, queries, k=5):
    """Benchmark FAISS if available."""
    try:
        import faiss
    except ImportError:
        print("  FAISS not installed, skipping. Install: pip install faiss-cpu")
        return None

    dim = embeddings.shape[1]
    n = embeddings.shape[0]

    # FAISS index (flat = brute force, same as our GPU/numpy backends)
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine since normalized)
    index.add(embeddings)

    # Warmup
    for q in queries[:3]:
        index.search(q.reshape(1, -1), k)

    times = []
    for q in queries:
        start = time.perf_counter()
        index.search(q.reshape(1, -1), k)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"  FAISS IndexFlatIP search k={k}: avg={times.mean():.2f}ms "
          f"p50={np.percentile(times, 50):.2f}ms "
          f"p99={np.percentile(times, 99):.2f}ms "
          f"(n={len(queries)})")
    return times.mean()


def main():
    dim = 1024
    print(f"MATHIR Vector Search Benchmark — dim={dim}")
    print("=" * 60)

    for n in [100, 1000, 5000]:
        print(f"\n--- N={n} ---")
        embeddings, items = generate_data(n, dim)
        queries = np.random.randn(50, dim).astype("float32")
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        # Test each backend
        for backend in ["gpu", "sqlite", "numpy"]:
            try:
                db_path = f"C:\\Users\\So-i-learn-3D\\AppData\\Local\\Temp\\bench_{backend}.db"
                if os.path.exists(db_path):
                    os.remove(db_path)
                if backend == "gpu" and not os.environ.get("CUDA_VISIBLE_DEVICES") and not __import__("torch").cuda.is_available():
                    print(f"  [SKIP] {backend}: no GPU")
                    continue

                search = VectorSearch(dim=dim, backend=backend, db_path=db_path)
                bench_insert(search, items, f"{backend:>6}")
                avg_search = bench_search(search, queries, k=5, label=f"{backend:>6}")
                search.close()

                # Cleanup
                if os.path.exists(db_path):
                    os.remove(db_path)
            except Exception as e:
                print(f"  [ERROR] {backend}: {e}")

        # FAISS baseline
        bench_faiss(embeddings, queries, k=5)

    print("\n" + "=" * 60)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()

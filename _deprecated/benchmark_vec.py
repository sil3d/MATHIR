"""
MATHIR VecMemory — Benchmark: Baseline vs Optimized
Measures throughput for N=1000 operations using sqlite-vec.
"""
import sys
import os
import time
import statistics
import tempfile
import shutil
import numpy as np

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mathir_vec import VecMemory as VecMemoryBaseline
from mathir_vec_optimized import VecMemory as VecMemoryOptimized


def generate_items(n: int, dim: int = 384):
    """Generate N test items with pre-computed embeddings."""
    rng = np.random.RandomState(42)
    items = []
    for i in range(n):
        vec = rng.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        items.append({
            "memory_id": f"mem-{i:06d}",
            "embedding": vec,
            "metadata": {
                "agent": f"agent_{i % 5}",
                "tier": ["semantic", "episodic", "working_memory", "procedural"][i % 4],
                "label": f"label-{i}",
                "priority": i % 10,
                "project": "benchmark",
            },
        })
    return items


def bench_single_inserts(mem, items, label):
    """Benchmark N individual insert operations."""
    times = []
    for item in items:
        t0 = time.perf_counter()
        mem.store(item["memory_id"], item["embedding"], item["metadata"])
        times.append(time.perf_counter() - t0)
    total = sum(times)
    avg = statistics.mean(times)
    med = statistics.median(times)
    print(f"  [{label}] {len(items)} inserts: {total:.3f}s total, "
          f"{avg*1000:.2f}ms avg, {med*1000:.2f}ms median, "
          f"{len(items)/total:.0f} ops/s")
    return total


def bench_batch_inserts(mem, items, label):
    """Benchmark batch insert (one transaction)."""
    t0 = time.perf_counter()
    ids = mem.store_batch(items)
    total = time.perf_counter() - t0
    print(f"  [{label}] batch {len(items)} inserts: {total:.3f}s, "
          f"{len(items)/total:.0f} ops/s")
    return total, len(ids)


def bench_search(mem, queries, k=5, label=""):
    """Benchmark search operations."""
    times = []
    for q in queries:
        t0 = time.perf_counter()
        mem.search(q, k=k)
        times.append(time.perf_counter() - t0)
    total = sum(times)
    avg = statistics.mean(times)
    print(f"  [{label}] {len(queries)} searches: {total:.3f}s, "
          f"{avg*1000:.2f}ms avg, {len(queries)/total:.0f} ops/s")
    return total


def bench_search_cached(mem, queries, k=5, label=""):
    """Benchmark search with cache hits (repeat same queries)."""
    # First run populates cache
    for q in queries:
        mem.search(q, k=k)
    # Second run hits cache
    times = []
    for q in queries:
        t0 = time.perf_counter()
        mem.search(q, k=k)
        times.append(time.perf_counter() - t0)
    total = sum(times)
    avg = statistics.mean(times)
    print(f"  [{label}] {len(queries)} CACHED searches: {total:.4f}s, "
          f"{avg*1000:.3f}ms avg, {len(queries)/total:.0f} ops/s")
    return total


def bench_delete(mem, ids, label=""):
    """Benchmark delete operations."""
    times = []
    for mid in ids:
        t0 = time.perf_counter()
        mem.delete(mid)
        times.append(time.perf_counter() - t0)
    total = sum(times)
    avg = statistics.mean(times)
    print(f"  [{label}] {len(ids)} deletes: {total:.3f}s, "
          f"{avg*1000:.2f}ms avg, {len(ids)/total:.0f} ops/s")
    return total


def run_benchmark(n: int = 1000):
    """Full benchmark suite."""
    dim = 384
    items = generate_items(n, dim)
    # Generate 100 random query vectors
    rng = np.random.RandomState(99)
    queries = []
    for _ in range(100):
        q = rng.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8
        queries.append(q)

    print(f"\n{'='*70}")
    print(f"  MATHIR VecMemory Benchmark — N={n}  (dim={dim})")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    print(f"\n--- BASELINE (mathir_vec.py — sqlite-vec) ---")
    base_dir = tempfile.mkdtemp(prefix="mathir_base_")
    base_db = os.path.join(base_dir, "vec.db")
    base = VecMemoryBaseline(db_path=base_db, dim=dim)

    base_insert = bench_single_inserts(base, items, "baseline")
    base_search = bench_search(base, queries, k=5, label="baseline")
    base_search_cached = bench_search_cached(base, queries, k=5, label="baseline")
    base_search2 = bench_search(base, queries, k=5, label="baseline")

    del base
    shutil.rmtree(base_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Optimized
    # ------------------------------------------------------------------
    print(f"\n--- OPTIMIZED (mathir_vec_optimized.py — WAL + pool + cache) ---")
    opt_dir = tempfile.mkdtemp(prefix="mathir_opt_")
    opt_db = os.path.join(opt_dir, "vec.db")
    opt = VecMemoryOptimized(db_path=opt_db, dim=dim)

    opt_insert = bench_single_inserts(opt, items, "optimized")
    opt_batch_total, opt_batch_count = bench_batch_inserts(opt, items, "optimized")
    opt_search = bench_search(opt, queries, k=5, label="optimized")
    opt_search_cached = bench_search_cached(opt, queries, k=5, label="optimized")
    opt_search2 = bench_search(opt, queries, k=5, label="optimized")

    # Delete benchmark (use first 50 IDs)
    del_ids = [it["memory_id"] for it in items[:50]]
    opt_delete = bench_delete(opt, del_ids, label="optimized")

    opt_stats = opt.stats()

    del opt
    shutil.rmtree(opt_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Operation':<35} {'Baseline':>10} {'Optimized':>10} {'Speedup':>8}")
    print(f"  {'-'*63}")
    print(f"  {'Single inserts (total)':<35} {base_insert:>9.3f}s {opt_insert:>9.3f}s {base_insert/opt_insert:>7.1f}x")
    print(f"  {'Batch insert (N in 1 tx)':<35} {'N/A':>10} {opt_batch_total:>9.3f}s {'inf':>8}")
    print(f"  {'Search (100 queries)':<35} {base_search:>9.3f}s {opt_search:>9.3f}s {base_search/opt_search:>7.1f}x")
    print(f"  {'Search (100, CACHED)':<35} {base_search_cached:>9.4f}s {opt_search_cached:>9.4f}s {base_search_cached/opt_search_cached:>7.1f}x")
    print(f"  {'Second search pass':<35} {base_search2:>9.3f}s {opt_search2:>9.3f}s {base_search2/opt_search2:>7.1f}x")
    print(f"  {'-'*63}")
    print(f"  Total baseline:   {base_insert + base_search + base_search2:.3f}s")
    total_opt = opt_insert + opt_batch_total + opt_search + opt_search2
    print(f"  Total optimized:  {total_opt:.3f}s")
    overall = (base_insert + base_search + base_search2) / total_opt
    print(f"  Overall speedup:  {overall:.1f}x")
    print(f"\n  Optimized stats: {json.dumps(opt_stats, indent=2, default=str)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import json
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run_benchmark(N)

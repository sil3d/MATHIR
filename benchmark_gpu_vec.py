"""
GPU Vector Search Benchmark — MATHIR
=====================================

Benchmarks ``GPUVecMemory`` vs ``VecMemory`` (sqlite-vec) at N=100, 1000, 5000.

Measures:
  • Insert throughput  (single + batch)
  • Search latency    (P50 / P99 / mean)
  • Throughput        (queries/sec)
  • Memory usage      (GPU VRAM or CPU RAM)

Usage::

    python benchmark_gpu_vec.py              # run all sizes
    python benchmark_gpu_vec.py --dim 384    # custom embedding dim
    python benchmark_gpu_vec.py --k 10       # top-k = 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mathir_gpu_vec import GPUVecMemory

# sqlite-vec is optional — skip if not installed
try:
    from mathir_vec import VecMemory
    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_random_embeddings(n: int, dim: int) -> np.ndarray:
    """Generate n random L2-normalized float32 embeddings."""
    embs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs /= np.clip(norms, 1e-8, None)
    return embs


def _percentile(data: List[float], p: float) -> float:
    """Compute percentile from a list."""
    s = sorted(data)
    idx = int(len(s) * p / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


def _gpu_memory_mb() -> float:
    """Return current GPU VRAM usage in MB (0 if CPU)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def _clear_gpu():
    """Free GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Benchmark: Insert
# ---------------------------------------------------------------------------

def bench_insert(
    n: int,
    dim: int,
    use_gpu: bool = True,
) -> Dict:
    """Benchmark single + batch insert."""
    _clear_gpu()

    results: Dict = {}

    # --- GPUVecMemory: batch insert ---
    if use_gpu:
        gpu = GPUVecMemory(dim=dim)
        embs = _generate_random_embeddings(n, dim)
        ids = [f"gpu_{i}" for i in range(n)]
        metas = [{"agent": f"agent_{i % 5}", "tier": "episodic"} for i in range(n)]

        t0 = time.perf_counter()
        gpu.store_batch(ids, embs, metas)
        batch_us = (time.perf_counter() - t0) * 1_000_000
        gpu_stats = gpu.stats()
        results["gpu_batch"] = {
            "count": n,
            "total_us": round(batch_us, 1),
            "per_item_us": round(batch_us / n, 1),
            "throughput": round(n / (batch_us / 1_000_000)) if batch_us > 0 else 0,
            "memory_mb": gpu_stats["memory_mb"],
            "device": str(gpu.device),
        }
        gpu.close()
        del gpu, embs, ids, metas
        _clear_gpu()

    # --- GPUVecMemory: single insert ---
    if use_gpu:
        gpu = GPUVecMemory(dim=dim)
        single_latencies: List[float] = []
        for i in range(min(n, 1000)):  # cap at 1000 for timing
            emb = np.random.randn(dim).astype(np.float32)
            emb /= max(np.linalg.norm(emb), 1e-8)
            t0 = time.perf_counter()
            gpu.store(f"sg_{i}", emb, {"agent": "test"})
            single_latencies.append((time.perf_counter() - t0) * 1_000_000)
        results["gpu_single"] = {
            "count": len(single_latencies),
            "mean_us": round(statistics.mean(single_latencies), 1),
            "p50_us": round(_percentile(single_latencies, 50), 1),
            "p99_us": round(_percentile(single_latencies, 99), 1),
        }
        gpu.close()
        del gpu
        _clear_gpu()

    # --- VecMemory: batch insert ---
    if HAS_SQLITE_VEC:
        db_path = str(PROJECT_ROOT / "_bench_vec.db")
        vm = VecMemory(db_path, dim)
        embs = _generate_random_embeddings(n, dim)
        ids = [f"vec_{i}" for i in range(n)]
        metas = [{"agent": f"agent_{i % 5}", "tier": "episodic"} for i in range(n)]

        t0 = time.perf_counter()
        for i in range(n):
            vm.store(ids[i], embs[i], metas[i])
        vec_total_us = (time.perf_counter() - t0) * 1_000_000
        results["sqlite_vec"] = {
            "count": n,
            "total_us": round(vec_total_us, 1),
            "per_item_us": round(vec_total_us / n, 1),
            "throughput": round(n / (vec_total_us / 1_000_000)) if vec_total_us > 0 else 0,
        }
        vm.close()
        del vm, embs, ids, metas
        # Clean up db file
        for f in ["_bench_vec.db", "_bench_vec.db-wal", "_bench_vec.db-shm"]:
            p = PROJECT_ROOT / f
            if p.exists():
                p.unlink(missing_ok=True)

    return results


# ---------------------------------------------------------------------------
# Benchmark: Search
# ---------------------------------------------------------------------------

def bench_search(
    n: int,
    dim: int,
    k: int,
    n_queries: int = 200,
    use_gpu: bool = True,
) -> Dict:
    """Benchmark search latency."""
    _clear_gpu()
    results: Dict = {}

    query_embs = _generate_random_embeddings(n_queries, dim)

    # --- GPUVecMemory search ---
    if use_gpu:
        gpu = GPUVecMemory(dim=dim)
        embs = _generate_random_embeddings(n, dim)
        ids = [f"gpu_{i}" for i in range(n)]
        metas = [{"agent": f"agent_{i % 5}", "tier": "episodic"} for i in range(n)]
        gpu.store_batch(ids, embs, metas)

        # Warmup (first CUDA kernel launch is slow)
        gpu.search(query_embs[0], k=k)

        # Timed runs
        latencies_us: List[float] = []
        for i in range(n_queries):
            t0 = time.perf_counter()
            gpu.search(query_embs[i], k=k)
            latencies_us.append((time.perf_counter() - t0) * 1_000_000)

        results["gpu"] = {
            "n": n,
            "k": k,
            "n_queries": n_queries,
            "mean_us": round(statistics.mean(latencies_us), 1),
            "p50_us": round(_percentile(latencies_us, 50), 1),
            "p99_us": round(_percentile(latencies_us, 99), 1),
            "throughput_qps": round(n_queries / (sum(latencies_us) / 1_000_000), 1),
            "device": str(gpu.device),
        }

        # Filtered search (10% match rate)
        filter_latencies: List[float] = []
        for i in range(min(n_queries, 50)):
            t0 = time.perf_counter()
            gpu.search(query_embs[i], k=k, agent_filter="agent_0")
            filter_latencies.append((time.perf_counter() - t0) * 1_000_000)
        results["gpu_filtered"] = {
            "mean_us": round(statistics.mean(filter_latencies), 1),
            "p50_us": round(_percentile(filter_latencies, 50), 1),
        }

        gpu.close()
        del gpu, embs, ids, metas
        _clear_gpu()

    # --- VecMemory search ---
    if HAS_SQLITE_VEC:
        db_path = str(PROJECT_ROOT / "_bench_vec.db")
        vm = VecMemory(db_path, dim)
        embs = _generate_random_embeddings(n, dim)
        ids = [f"vec_{i}" for i in range(n)]
        metas = [{"agent": f"agent_{i % 5}", "tier": "episodic"} for i in range(n)]
        for i in range(n):
            vm.store(ids[i], embs[i], metas[i])

        # Warmup
        vm.search(query_embs[0], k=k)

        # Timed runs
        vec_latencies: List[float] = []
        for i in range(n_queries):
            t0 = time.perf_counter()
            vm.search(query_embs[i], k=k)
            vec_latencies.append((time.perf_counter() - t0) * 1_000_000)

        results["sqlite_vec"] = {
            "n": n,
            "k": k,
            "n_queries": n_queries,
            "mean_us": round(statistics.mean(vec_latencies), 1),
            "p50_us": round(_percentile(vec_latencies, 50), 1),
            "p99_us": round(_percentile(vec_latencies, 99), 1),
            "throughput_qps": round(n_queries / (sum(vec_latencies) / 1_000_000), 1),
        }

        vm.close()
        del vm, embs, ids, metas
        for f in ["_bench_vec.db", "_bench_vec.db-wal", "_bench_vec.db-shm"]:
            p = PROJECT_ROOT / f
            if p.exists():
                p.unlink(missing_ok=True)

    # --- Speedup ---
    if "gpu" in results and "sqlite_vec" in results:
        gpu_us = results["gpu"]["mean_us"]
        vec_us = results["sqlite_vec"]["mean_us"]
        results["speedup"] = round(vec_us / gpu_us, 2) if gpu_us > 0 else float("inf")

    return results


# ---------------------------------------------------------------------------
# Benchmark: Delete
# ---------------------------------------------------------------------------

def bench_delete(n: int, dim: int, use_gpu: bool = True) -> Dict:
    """Benchmark single delete."""
    _clear_gpu()
    results: Dict = {}

    if use_gpu:
        gpu = GPUVecMemory(dim=dim)
        embs = _generate_random_embeddings(n, dim)
        ids = [f"del_{i}" for i in range(n)]
        gpu.store_batch(ids, embs)

        del_latencies: List[float] = []
        for i in range(min(n, 500)):
            mid = f"del_{i}"
            t0 = time.perf_counter()
            gpu.delete(mid)
            del_latencies.append((time.perf_counter() - t0) * 1_000_000)

        results["gpu_delete"] = {
            "count": len(del_latencies),
            "mean_us": round(statistics.mean(del_latencies), 1),
            "p50_us": round(_percentile(del_latencies, 50), 1),
            "p99_us": round(_percentile(del_latencies, 99), 1),
        }
        gpu.close()
        del gpu
        _clear_gpu()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MATHIR GPU Vector Search Benchmark")
    parser.add_argument("--dim", type=int, default=1024, help="Embedding dimension (default: 1024)")
    parser.add_argument("--k", type=int, default=5, help="Top-k for search (default: 5)")
    parser.add_argument("--queries", type=int, default=200, help="Number of query iterations (default: 200)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU benchmarks")
    args = parser.parse_args()

    dim = args.dim
    k = args.k
    n_queries = args.queries
    use_gpu = not args.no_gpu

    # Device info
    device = "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    else:
        gpu_name = "N/A"
        gpu_mem = 0

    print("=" * 72)
    print("MATHIR GPU Vector Search Benchmark")
    print("=" * 72)
    print(f"  Device       : {device}")
    print(f"  GPU          : {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"  Embedding dim: {dim}")
    print(f"  Top-k        : {k}")
    print(f"  Queries      : {n_queries}")
    print(f"  sqlite-vec   : {'available' if HAS_SQLITE_VEC else 'NOT installed — skipping CPU baseline'}")
    print("=" * 72)

    sizes = [100, 1000, 5000]
    all_results: Dict[str, Dict] = {}

    for n in sizes:
        print(f"\n{'-' * 72}")
        print(f"  N = {n}")
        print(f"{'-' * 72}")

        # --- Insert ---
        print(f"  [1/3] Insert benchmark...")
        insert_res = bench_insert(n, dim, use_gpu=use_gpu)
        all_results[f"insert_{n}"] = insert_res

        if "gpu_batch" in insert_res:
            r = insert_res["gpu_batch"]
            print(f"    GPU batch  : {r['total_us']:>10.0f} µs  ({r['throughput']:>8} items/s)  [{r['device']}]")
        if "sqlite_vec" in insert_res:
            r = insert_res["sqlite_vec"]
            print(f"    sqlite-vec : {r['total_us']:>10.0f} µs  ({r['throughput']:>8} items/s)")
        if "gpu_single" in insert_res:
            r = insert_res["gpu_single"]
            print(f"    GPU single : mean={r['mean_us']:.0f} µs  p50={r['p50_us']:.0f}  p99={r['p99_us']:.0f}")

        # --- Search ---
        print(f"  [2/3] Search benchmark (k={k}, {n_queries} queries)...")
        search_res = bench_search(n, dim, k, n_queries, use_gpu=use_gpu)
        all_results[f"search_{n}"] = search_res

        if "gpu" in search_res:
            r = search_res["gpu"]
            print(f"    GPU        : mean={r['mean_us']:>7.0f} µs  p50={r['p50_us']:>7.0f}  p99={r['p99_us']:>7.0f}  "
                  f"({r['throughput_qps']:.0f} q/s)  [{r['device']}]")
        if "gpu_filtered" in search_res:
            r = search_res["gpu_filtered"]
            print(f"    GPU filtered: mean={r['mean_us']:>7.0f} µs  p50={r['p50_us']:>7.0f}")
        if "sqlite_vec" in search_res:
            r = search_res["sqlite_vec"]
            print(f"    sqlite-vec : mean={r['mean_us']:>7.0f} µs  p50={r['p50_us']:>7.0f}  p99={r['p99_us']:>7.0f}  "
                  f"({r['throughput_qps']:.0f} q/s)")
        if "speedup" in search_res:
            print(f"    Speedup    : {search_res['speedup']:.1f}x (GPU vs sqlite-vec)")

        # --- Delete ---
        print(f"  [3/3] Delete benchmark...")
        delete_res = bench_delete(n, dim, use_gpu=use_gpu)
        all_results[f"delete_{n}"] = delete_res

        if "gpu_delete" in delete_res:
            r = delete_res["gpu_delete"]
            print(f"    GPU delete : mean={r['mean_us']:>7.0f} µs  p50={r['p50_us']:>7.0f}  p99={r['p99_us']:>7.0f}")

    # --- Summary table ---
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"{'N':>6}  {'GPU search (µs)':>16}  {'sqlite-vec (µs)':>16}  {'Speedup':>8}")
    print("-" * 72)
    for n in sizes:
        sr = all_results.get(f"search_{n}", {})
        gpu_us = sr.get("gpu", {}).get("mean_us", "—")
        vec_us = sr.get("sqlite_vec", {}).get("mean_us", "—")
        spd = sr.get("speedup", "—")
        gpu_str = f"{gpu_us:>10.0f}" if isinstance(gpu_us, (int, float)) else f"{gpu_us:>10}"
        vec_str = f"{vec_us:>10.0f}" if isinstance(vec_us, (int, float)) else f"{vec_us:>10}"
        spd_str = f"{spd:>6.1f}x" if isinstance(spd, (int, float)) else f"{spd:>8}"
        print(f"{n:>6}  {gpu_str}  {vec_str}  {spd_str}")
    print("-" * 72)

    # --- Save results ---
    out_dir = PROJECT_ROOT / "benchmarks" / "results_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "gpu_vec_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()

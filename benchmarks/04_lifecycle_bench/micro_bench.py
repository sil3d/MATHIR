"""
Micro-benchmark for the 4 lifecycle phases — memory-only, no LLM.

Measures throughput and timing for:
  1. touch_recall on N memories
  2. promote + auto_promote_all
  3. boost_on_recall + decay_all
  4. consolidate_all with planted duplicates
  5. build_links_all + get_links BFS
  6. find_related (vector + graph)

Usage:
  python micro_bench.py --count 1000 --seed 42
  python micro_bench.py --count 5000 --out results_micro.json
"""
import os
import sys
import json
import time
import shutil
import tempfile
import argparse
import statistics
from pathlib import Path
from typing import List

import numpy as np

# Bootstrap imports — package is portable
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent / "mathir_mcp"
_LIB = _PKG_ROOT / "mathir_lib"
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_LIB))

from mathir_vec import VecMemory


def _unit_vec(dim=384, seed=None):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _seed_memories(memory: VecMemory, n: int, dim: int, seed: int,
                    duplicate_ratio: float = 0.0) -> List[str]:
    """Store N random memories. Optional duplicates near the first ones."""
    rng = np.random.default_rng(seed)
    ids = []
    t0 = time.perf_counter()
    for i in range(n):
        mid = f"bench_{i:06d}"
        v = rng.standard_normal(dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        memory.store(mid, v, {
            "agent": "bench",
            "block_type": "episodic",
            "label": f"item-{i}",
            "priority": 5,
            "content": f"benchmark item {i}",
        })
        ids.append(mid)
        # Plant duplicates
        if i > 0 and i < int(n * duplicate_ratio):
            dup_id = f"bench_dup_{i:06d}"
            v_dup = v + 0.005 * rng.standard_normal(dim).astype(np.float32)
            v_dup /= (np.linalg.norm(v_dup) + 1e-9)
            memory.store(dup_id, v_dup, {
                "agent": "bench",
                "block_type": "episodic",
                "label": f"item-{i}-copy",
                "priority": 5,
                "content": f"benchmark item {i} copy",
            })
            ids.append(dup_id)
    return ids, time.perf_counter() - t0


def _bench_touch_recall(memory: VecMemory, ids: List[str], iters: int = 3) -> dict:
    """Touch every memory N times; measure p50/p95/p99 + ops/sec."""
    timings = []
    n_ops = 0
    t0 = time.perf_counter()
    for _ in range(iters):
        for mid in ids:
            ts = time.perf_counter()
            memory.touch_recall(mid)
            timings.append((time.perf_counter() - ts) * 1000)
            n_ops += 1
    wall = time.perf_counter() - t0
    return {
        "ops": n_ops,
        "wall_s": round(wall, 3),
        "ops_per_sec": round(n_ops / wall, 1),
        "latency_ms": {
            "p50": round(statistics.median(timings), 3),
            "p95": round(np.percentile(timings, 95), 3),
            "p99": round(np.percentile(timings, 99), 3),
        },
    }


def _bench_promote(memory: VecMemory, ids: List[str]) -> dict:
    t0 = time.perf_counter()
    res = memory.auto_promote_all()
    wall = time.perf_counter() - t0
    return {
        "scanned": len(ids),
        "promoted": len(res),
        "wall_s": round(wall, 3),
        "ms_per_mem": round(wall * 1000 / max(len(ids), 1), 3),
    }


def _bench_decay(memory: VecMemory, ids: List[str], threshold_days: int = 30) -> dict:
    # First boost a few to have non-zero stability
    for mid in ids[: max(1, len(ids) // 4)]:
        try:
            memory.boost_on_recall(mid)
        except Exception:
            pass
    t0 = time.perf_counter()
    res = memory.decay_all(threshold_days=threshold_days, archive_floor=0.05)
    wall = time.perf_counter() - t0
    return {
        "scanned": len(ids),
        "decayed": res.get("decayed", 0),
        "archived": res.get("archived", 0),
        "by_tier": res.get("by_tier", {}),
        "wall_s": round(wall, 3),
        "ms_per_mem": round(wall * 1000 / max(len(ids), 1), 3),
    }


def _bench_consolidate(memory: VecMemory, duplicate_ratio: float) -> dict:
    t0 = time.perf_counter()
    res = memory.consolidate_all(threshold=0.95, dry_run=False, limit=10000)
    wall = time.perf_counter() - t0
    return {
        "merged": res.get("merged", 0),
        "candidates": res.get("candidates", 0) if isinstance(res.get("candidates"), int)
                       else len(res.get("candidates", [])),
        "by_tier": res.get("by_tier", {}),
        "wall_s": round(wall, 3),
    }


def _bench_link_graph(memory: VecMemory, ids: List[str], threshold: float = 0.7) -> dict:
    # Build
    t0 = time.perf_counter()
    build_res = memory.build_links_all(threshold=threshold, limit=10000)
    build_wall = time.perf_counter() - t0

    # BFS — sample 50 random nodes
    sample = ids[:: max(1, len(ids) // 50)][:50]
    bfs_timings = []
    bfs_results = []
    for mid in sample:
        ts = time.perf_counter()
        links = memory.get_links(mid, depth=2, decay=0.5)
        bfs_timings.append((time.perf_counter() - ts) * 1000)
        bfs_results.append(len(links))

    # Vector + graph combined
    t0 = time.perf_counter()
    related_count = 0
    for mid in sample:
        related = memory.find_related(mid, max_hops=2, min_weight=0.1)
        related_count += len(related)
    related_wall = time.perf_counter() - t0

    return {
        "build": {
            "links_created": build_res.get("links_created", 0),
            "memories_scanned": build_res.get("memories_scanned", 0),
            "wall_s": round(build_wall, 3),
            "links_per_sec": round(build_res.get("links_created", 0) / max(build_wall, 0.001), 1),
        },
        "bfs_get_links": {
            "samples": len(sample),
            "avg_links_per_node": round(statistics.mean(bfs_results), 2) if bfs_results else 0,
            "latency_ms": {
                "p50": round(statistics.median(bfs_timings), 3) if bfs_timings else 0,
                "p95": round(np.percentile(bfs_timings, 95), 3) if bfs_timings else 0,
            },
        },
        "find_related": {
            "total_related": related_count,
            "wall_s": round(related_wall, 3),
            "avg_per_query": round(related_count / max(len(sample), 1), 2),
        },
    }


def run(count: int, dim: int, seed: int, duplicate_ratio: float, out: Path):
    print(f"\n=== Micro-benchmark: {count} memories (dim={dim}, seed={seed}) ===\n")
    tmp = Path(tempfile.mkdtemp(prefix="mathir_microbench_"))
    db = tmp / "bench.db"
    memory = VecMemory(db, dim)

    # 1. Seed
    print(f"[1/6] Seeding {count} memories (duplicates: {duplicate_ratio*100:.0f}%)...")
    ids, seed_wall = _seed_memories(memory, count, dim, seed, duplicate_ratio)
    print(f"      -> {len(ids)} memories in {seed_wall:.2f}s "
          f"({len(ids)/seed_wall:.0f} mem/s)")

    # 2. Touch recall
    print("\n[2/6] touch_recall (3 iterations over all memories)...")
    touch = _bench_touch_recall(memory, ids, iters=3)
    print(f"      -> {touch['ops_per_sec']:.0f} ops/s, "
          f"p50={touch['latency_ms']['p50']:.2f}ms p95={touch['latency_ms']['p95']:.2f}ms")

    # 3. Promote
    print("\n[3/6] auto_promote_all...")
    promote = _bench_promote(memory, ids)
    print(f"      -> scanned={promote['scanned']}, promoted={promote['promoted']}, "
          f"{promote['ms_per_mem']:.2f}ms/mem")

    # 4. Decay
    print("\n[4/6] decay_all (threshold=30d)...")
    decay = _bench_decay(memory, ids)
    print(f"      -> decayed={decay['decayed']}, archived={decay['archived']}, "
          f"by_tier={decay['by_tier']}")

    # 5. Consolidate
    print("\n[5/6] consolidate_all (threshold=0.95)...")
    cons = _bench_consolidate(memory, duplicate_ratio)
    print(f"      -> merged={cons['merged']}, candidates={cons['candidates']}, "
          f"by_tier={cons['by_tier']}")

    # 6. Link graph
    print("\n[6/6] build_links_all + BFS + find_related...")
    links = _bench_link_graph(memory, ids, threshold=0.7)
    print(f"      -> {links['build']['links_created']} links in {links['build']['wall_s']:.2f}s")
    print(f"      -> BFS avg {links['bfs_get_links']['avg_links_per_node']} links/node, "
          f"p95={links['bfs_get_links']['latency_ms']['p95']:.2f}ms")
    print(f"      -> find_related: {links['find_related']['avg_per_query']} avg per query")

    # Final DB size
    db_size_mb = db.stat().st_size / (1024 * 1024)

    result = {
        "config": {"count": count, "dim": dim, "seed": seed, "duplicate_ratio": duplicate_ratio},
        "seed_wall_s": round(seed_wall, 3),
        "db_size_mb": round(db_size_mb, 2),
        "touch_recall": touch,
        "promote": promote,
        "decay": decay,
        "consolidate": cons,
        "link_graph": links,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n=== Results saved to {out}")
    print(f"=== DB size: {db_size_mb:.2f} MB")

    # Cleanup
    memory.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duplicates", type=float, default=0.2,
                   help="Fraction of memories to duplicate (for consolidate test)")
    p.add_argument("--out", type=Path, default=Path("results_micro.json"))
    args = p.parse_args()
    run(args.count, args.dim, args.seed, args.duplicates, args.out)


if __name__ == "__main__":
    main()

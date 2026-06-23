"""
STRESS TEST 2: Cache Warm + Repeat Query Patterns
==================================================

Tests the V7.2 LRU cache under realistic chat-like workloads:
  - 50 unique queries
  - 3 reps each (mimics conversation rephrasing)
  - 5 rounds (250 total queries)
"""

import os
import sys
import time
import json
import statistics
import argparse
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import fitz

# Use the same setup as comprehensive_stress_test
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comprehensive_stress_test import (
    extract_pdf_chunks,
    RealEmbedder,
    FLUID_MECHANICS_QUERIES,
    keyword_overlap_score,
    run_mathir_v7_approach_d,
)


def run_cache_warm_test(embeddings: np.ndarray, query_embeddings: np.ndarray,
                        queries: List[str], chunks: List[Dict]) -> Dict:
    """Run cache warm stress test on Hybrid Episodic Memory with cache ON vs OFF."""
    from mathir_lib.memory import HybridEpisodicMemory

    dim = embeddings.shape[1]

    # Test 1: Cache OFF (cold)
    print("\n--- Test 1: Hybrid Episodic with Cache OFF (cold path) ---")
    mem_off = HybridEpisodicMemory(
        feature_dim=dim,
        use_cross_encoder=False,
        use_result_cache=False,
        use_adaptive_rerank=False,
        capacity=len(embeddings) + 100
    )
    t0 = time.perf_counter()
    for i, emb in enumerate(embeddings):
        emb_t = torch.from_numpy(emb).float()
        mem_off.store(emb_t, text=chunks[i]["text"])
    storage_off = (time.perf_counter() - t0) * 1000

    cold_latencies = []
    cold_qualities = []
    for q_emb, q in zip(query_embeddings, queries):
        emb_t = torch.from_numpy(q_emb).float()
        t0 = time.perf_counter()
        try:
            indices, scores = mem_off.search(emb_t, query_text=q, k=5)
        except Exception as e:
            indices = None
        cold_latencies.append((time.perf_counter() - t0) * 1000)
        if indices is not None and len(indices) > 0:
            first = indices[0]
            if hasattr(first, '__len__'):
                top_idx = int(first[0].item() if hasattr(first[0], 'item') else first[0])
            else:
                top_idx = int(first.item() if hasattr(first, 'item') else first)
            if 0 <= top_idx < len(chunks):
                overlap = keyword_overlap_score(q, chunks[top_idx]["text"])
                cold_qualities.append(overlap)

    print(f"  Storage: {storage_off:.2f} ms")
    print(f"  Latency mean: {statistics.mean(cold_latencies):.2f} ms, median: {statistics.median(cold_latencies):.2f} ms")
    print(f"  Quality: {statistics.mean(cold_qualities) if cold_qualities else 0:.2%}")

    # Test 2: Cache ON (cold then warm)
    print("\n--- Test 2: Hybrid Episodic with Cache ON (warm path) ---")
    mem_on = HybridEpisodicMemory(
        feature_dim=dim,
        use_cross_encoder=False,
        use_result_cache=True,
        use_adaptive_rerank=False,
        cache_size=10000,
        capacity=len(embeddings) + 100
    )
    t0 = time.perf_counter()
    for i, emb in enumerate(embeddings):
        emb_t = torch.from_numpy(emb).float()
        mem_on.store(emb_t, text=chunks[i]["text"])
    storage_on = (time.perf_counter() - t0) * 1000

    # Run each query 3 times to test cache hits
    warm_latencies = []
    warm_qualities = []
    cache_hits = 0
    total_queries = 0

    for round_num in range(3):  # 3 repetitions
        round_qualities = []
        for q_emb, q in zip(query_embeddings, queries):
            emb_t = torch.from_numpy(q_emb).float()
            t0 = time.perf_counter()
            try:
                indices, scores = mem_on.search(emb_t, query_text=q, k=5)
            except Exception as e:
                indices = None
            warm_latencies.append((time.perf_counter() - t0) * 1000)
            total_queries += 1

            if indices is not None and len(indices) > 0:
                first = indices[0]
                if hasattr(first, '__len__'):
                    top_idx = int(first[0].item() if hasattr(first[0], 'item') else first[0])
                else:
                    top_idx = int(first.item() if hasattr(first, 'item') else first)
                if 0 <= top_idx < len(chunks):
                    overlap = keyword_overlap_score(q, chunks[top_idx]["text"])
                    round_qualities.append(overlap)
        warm_qualities.append(round_qualities)

    # Get cache info
    cache_info = mem_on.cache_info() if hasattr(mem_on, 'cache_info') else {}

    # First round = cold, subsequent rounds = warm
    first_round_size = len(query_embeddings)
    cold_lat = warm_latencies[:first_round_size]
    warm_lat = warm_latencies[first_round_size:]

    print(f"  Storage: {storage_on:.2f} ms")
    print(f"  Cold round latency: mean={statistics.mean(cold_lat):.2f} ms")
    print(f"  Warm rounds latency: mean={statistics.mean(warm_lat) if warm_lat else 0:.2f} ms")
    print(f"  Quality (all rounds): {statistics.mean([item for sub in warm_qualities for item in sub]) if warm_qualities else 0:.2%}")
    print(f"  Cache info: {cache_info}")

    # Test 3: Cache with adaptive rerank
    print("\n--- Test 3: Hybrid Episodic with Cache + Adaptive Rerank ---")
    mem_ada = HybridEpisodicMemory(
        feature_dim=dim,
        use_cross_encoder=False,
        use_result_cache=True,
        use_adaptive_rerank=True,
        cache_size=10000,
        capacity=len(embeddings) + 100
    )
    t0 = time.perf_counter()
    for i, emb in enumerate(embeddings):
        emb_t = torch.from_numpy(emb).float()
        mem_ada.store(emb_t, text=chunks[i]["text"])
    storage_ada = (time.perf_counter() - t0) * 1000

    ada_latencies = []
    for round_num in range(3):
        for q_emb, q in zip(query_embeddings, queries):
            emb_t = torch.from_numpy(q_emb).float()
            t0 = time.perf_counter()
            try:
                indices, scores = mem_ada.search(emb_t, query_text=q, k=5)
            except Exception as e:
                indices = None
            ada_latencies.append((time.perf_counter() - t0) * 1000)

    cold_lat_ada = ada_latencies[:first_round_size]
    warm_lat_ada = ada_latencies[first_round_size:]

    print(f"  Storage: {storage_ada:.2f} ms")
    print(f"  Cold round latency: mean={statistics.mean(cold_lat_ada):.2f} ms")
    print(f"  Warm rounds latency: mean={statistics.mean(warm_lat_ada) if warm_lat_ada else 0:.2f} ms")

    # Summary
    cold_speedup = statistics.mean(cold_lat) / statistics.mean(warm_lat) if warm_lat else 1
    print(f"\n=== CACHE SPEEDUP ===")
    print(f"  Cold median: {statistics.median(cold_lat):.2f} ms")
    print(f"  Warm median: {statistics.median(warm_lat):.2f} ms")
    print(f"  Speedup: {cold_speedup:.1f}x")

    return {
        "cache_off": {
            "storage_ms": storage_off,
            "latency_mean_ms": statistics.mean(cold_latencies),
            "latency_median_ms": statistics.median(cold_latencies),
            "quality": statistics.mean(cold_qualities) if cold_qualities else 0,
        },
        "cache_on_cold": {
            "latency_mean_ms": statistics.mean(cold_lat),
            "latency_median_ms": statistics.median(cold_lat),
        },
        "cache_on_warm": {
            "latency_mean_ms": statistics.mean(warm_lat) if warm_lat else 0,
            "latency_median_ms": statistics.median(warm_lat) if warm_lat else 0,
        },
        "cache_adaptive_cold": {
            "latency_mean_ms": statistics.mean(cold_lat_ada),
            "latency_median_ms": statistics.median(cold_lat_ada),
        },
        "cache_adaptive_warm": {
            "latency_mean_ms": statistics.mean(warm_lat_ada) if warm_lat_ada else 0,
            "latency_median_ms": statistics.median(warm_lat_ada) if warm_lat_ada else 0,
        },
        "speedup": cold_speedup,
        "cache_info": cache_info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=r"D:\COURS\Fluid Mechanics 2\27846107.pdf")
    parser.add_argument("--chunks", type=int, default=200)
    parser.add_argument("--queries", type=int, default=50)
    args = parser.parse_args()

    print("=" * 80)
    print("CACHE WARM STRESS TEST - V7.2 LRU Cache Performance")
    print("=" * 80)

    print("\n[1/3] Loading PDF and computing embeddings...")
    chunks = extract_pdf_chunks(args.pdf, max_chunks=args.chunks)
    if not chunks:
        print("ERROR: no chunks extracted")
        return

    embedder = RealEmbedder()
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(chunk_texts, show_progress=False)

    queries = FLUID_MECHANICS_QUERIES[:args.queries]
    if len(queries) < args.queries:
        queries = (queries * (args.queries // len(queries) + 1))[:args.queries]

    print(f"\n[2/3] Encoding {len(queries)} queries...")
    query_embeddings = embedder.encode(queries, show_progress=False)

    print(f"\n[3/3] Running cache warm stress test...")
    results = run_cache_warm_test(embeddings, query_embeddings, queries, chunks)

    # Save
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "cache_warm_stress_test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

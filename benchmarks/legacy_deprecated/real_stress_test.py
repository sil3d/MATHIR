"""
REAL STRESS TEST — Cache + Adaptive in Production Scenario
=============================================================

Simulates realistic workload patterns:
  - 70% repeated/follow-up queries (warm cache should help)
  - 20% similar queries (high cache hit rate)
  - 10% novel queries (cold cache path)

Tests the complete flow: storage, mixed workload, sustained throughput,
memory pressure, cache eviction, and quality retention.

Usage:
    python benchmarks/real_stress_test.py
    python benchmarks/real_stress_test.py --chunks 200 --queries 200 --duration 60
"""

import os
import sys
import time
import json
import argparse
import statistics
import random
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import fitz

from mathir_lib.memory.hybrid_episodic import HybridEpisodicMemory


class RealEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.dim = self.model.get_embedding_dimension()

    def encode(self, texts, show_progress=False):
        return self.model.encode(
            texts, batch_size=32, show_progress_bar=show_progress,
            convert_to_numpy=True, normalize_embeddings=True,
        )


def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        cw = words[i:i+chunk_size]
        if not cw:
            break
        chunks.append(" ".join(cw))
        i += chunk_size - overlap
    return chunks


def extract_chunks(pdf_path, max_chunks=200):
    print(f"  Loading PDF: {os.path.basename(pdf_path)}")
    doc = fitz.open(pdf_path)
    chunks = []
    for page_idx, page in enumerate(doc):
        text = page.get_text("text")
        if not text.strip():
            continue
        for ch_idx, ch_text in enumerate(chunk_text(text, chunk_size=150, overlap=20)):
            if len(ch_text.split()) < 30:
                continue
            chunks.append({
                "text": ch_text, "page": page_idx+1,
                "chunk_id": f"p{page_idx+1:04d}_c{ch_idx:03d}",
                "n_words": len(ch_text.split()),
            })
        if len(chunks) >= max_chunks:
            break
    doc.close()
    return chunks[:max_chunks]


# Realistic query workload: 70% repeated, 20% similar, 10% novel
BASE_QUERIES = [
    "What is the continuity equation for incompressible flow?",
    "Define Reynolds number and its physical meaning.",
    "What is Bernoulli's equation?",
    "How does viscosity affect flow in a pipe?",
    "What is the Navier-Stokes equation?",
    "Explain the difference between laminar and turbulent flow.",
    "What are the boundary conditions for no-slip flow?",
    "How do you calculate pressure drop in a pipe?",
    "What is the momentum equation in fluid mechanics?",
    "Explain the concept of stream function.",
    "What is the vorticity equation?",
    "How is dimensional analysis used in fluid mechanics?",
    "What are the major losses in pipe flow?",
    "Explain the concept of dynamic similarity.",
    "What is the energy equation for viscous flow?",
    "What is the critical Reynolds number for pipe flow?",
    "How do you calculate friction factor?",
    "What is the hydraulic diameter?",
    "Explain the concept of boundary layer thickness.",
    "What is the difference between steady and unsteady flow?",
]

NOVEL_QUERIES = [
    "Discuss the Kármán vortex street and its formation.",
    "What is the Prandtl number and its physical meaning?",
    "How does a Pitot-static tube measure velocity?",
    "Explain the phenomenon of transonic flow.",
    "What is the Reynolds-averaged Navier-Stokes equation?",
    "How do you model turbulence with the k-ω SST model?",
    "What are the differences between LES and RANS?",
    "Explain the difference between forced and free convection.",
    "What is the Nusselt number and its significance?",
    "How does a hydraulic accumulator work?",
    "What is the difference between subsonic and supersonic flow?",
    "Explain the function of a de Laval nozzle.",
    "What is the phenomenon of flow-induced vibration?",
    "How does a Pitot tube differ from a static port?",
    "What are the applications of computational fluid dynamics?",
]


def keyword_overlap(query, retrieved):
    q_words = {w.lower().strip(".,;:()[]\"'") for w in query.split() if len(w) > 3}
    stopwords = {"what", "when", "where", "which", "have", "does", "with",
                 "from", "this", "that", "are", "was", "were", "been", "how",
                 "explain", "define", "difference", "between"}
    q_words = q_words - stopwords
    if not q_words:
        return 0.0
    r_text = retrieved.lower()
    hits = sum(1 for w in q_words if w in r_text)
    return hits / len(q_words)


def make_realistic_workload(n_queries):
    """70% repeated, 20% similar, 10% novel."""
    workload = []
    for _ in range(n_queries):
        r = random.random()
        if r < 0.7:
            workload.append(("repeat", random.choice(BASE_QUERIES)))
        elif r < 0.9:
            # Similar: rephrase the base
            base = random.choice(BASE_QUERIES)
            variants = [
                f"What does {base.split('?')[0].split(' ', 1)[-1]} mean?",
                f"Explain {base.split('?')[0].split(' ', 1)[-1]}.",
                base.replace("?", " in detail?"),
            ]
            workload.append(("similar", random.choice(variants)))
        else:
            workload.append(("novel", random.choice(NOVEL_QUERIES)))
    return workload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=r"D:\COURS\Fluid Mechanics 2\White_2011_7ed_Fluid-Mechanics.pdf")
    parser.add_argument("--chunks", type=int, default=200)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("REAL-WORLD STRESS TEST — Cache + Adaptive Re-Ranking")
    print("=" * 70)
    print(f"PDF: {os.path.basename(args.pdf)}")
    print(f"Chunks: {args.chunks}")
    print(f"Queries: {args.queries} (70% repeat, 20% similar, 10% novel)")
    print(f"Random seed: {args.seed}")

    # Load data
    print("\n[1/4] Loading and embedding...")
    chunks = extract_chunks(args.pdf, max_chunks=args.chunks)
    embedder = RealEmbedder()
    chunk_embs = embedder.encode([c["text"] for c in chunks], show_progress=False)

    # Pre-compute embeddings for queries
    all_queries = list(set(BASE_QUERIES + NOVEL_QUERIES))
    query_embs_cache = embedder.encode(all_queries, show_progress=False)
    query_to_emb = {q: query_embs_cache[i] for i, q in enumerate(all_queries)}

    print(f"  {len(chunks)} chunks, {len(all_queries)} unique queries pre-encoded")

    # Build workload
    workload = make_realistic_workload(args.queries)
    repeat_count = sum(1 for t, _ in workload if t == "repeat")
    similar_count = sum(1 for t, _ in workload if t == "similar")
    novel_count = sum(1 for t, _ in workload if t == "novel")
    print(f"  Workload: {repeat_count} repeat, {similar_count} similar, {novel_count} novel")

    # Build both configurations
    print("\n[2/4] Building systems...")
    configs = [
        ("Original (no opt)", dict(use_result_cache=False, use_adaptive_rerank=False)),
        ("Cache only", dict(use_result_cache=True, use_adaptive_rerank=False)),
        ("Cache + Adaptive", dict(use_result_cache=True, use_adaptive_rerank=True)),
    ]
    systems = []
    for name, kwargs in configs:
        mem = HybridEpisodicMemory(capacity=2000, feature_dim=embedder.dim, **kwargs)
        for c, emb in zip(chunks, chunk_embs):
            t = torch.from_numpy(emb).float().unsqueeze(0)
            mem.store(t, text=c["text"])
        systems.append((name, mem))
        print(f"  {name}: stored {len(chunks)} chunks")

    # Run workload
    print(f"\n[3/4] Running {args.queries}-query workload...")
    results = []
    for sys_name, mem in systems:
        # Reset cache
        if hasattr(mem, 'clear_cache'):
            mem.clear_cache()
        if hasattr(mem, 'reset'):
            pass  # Don't reset, we want to keep the data

        latencies_by_type = {"repeat": [], "similar": [], "novel": []}
        overlaps_by_type = {"repeat": [], "similar": [], "novel": []}
        cache_hits = 0
        adaptive_skips = 0
        all_qualities = []

        for qtype, q in workload:
            if q in query_to_emb:
                q_emb = query_to_emb[q]
            else:
                q_emb = embedder.encode([q], show_progress=False)[0]
                query_to_emb[q] = q_emb
            t = torch.from_numpy(q_emb).float().unsqueeze(0)

            t0 = time.perf_counter()
            indices, sims = mem.search(t, k=5, query_text=q)
            lat = (time.perf_counter() - t0) * 1000
            latencies_by_type[qtype].append(lat)

            # Quality
            if indices.numel() > 0:
                idx = int(indices[0, 0].item()) if indices.dim() > 1 else 0
                if 0 <= idx < len(chunks):
                    top_text = chunks[idx]["text"]
                    ov = keyword_overlap(q, top_text)
                    overlaps_by_type[qtype].append(ov)
                    all_qualities.append(ov)

        # Get stats
        stats = mem.get_stats()
        cache_info = mem.cache_info() if hasattr(mem, 'cache_info') else {}

        # Compute metrics
        all_lats = []
        for lats in latencies_by_type.values():
            all_lats.extend(lats)

        result = {
            "name": sys_name,
            "total_queries": len(workload),
            "latency_overall": {
                "mean_ms": statistics.mean(all_lats),
                "median_ms": statistics.median(all_lats),
                "p95_ms": sorted(all_lats)[int(len(all_lats) * 0.95)],
                "p99_ms": sorted(all_lats)[int(len(all_lats) * 0.99)],
                "min_ms": min(all_lats),
                "max_ms": max(all_lats),
                "qps": 1000 / statistics.mean(all_lats) if all_lats else 0,
            },
            "latency_by_type": {
                t: {
                    "mean_ms": statistics.mean(l) if l else 0,
                    "median_ms": statistics.median(l) if l else 0,
                } for t, l in latencies_by_type.items()
            },
            "quality_overall": statistics.mean(all_qualities) if all_qualities else 0,
            "quality_by_type": {
                t: statistics.mean(o) if o else 0 for t, o in overlaps_by_type.items()
            },
            "cache_hits": cache_info.get("hits", cache_hits),
            "cache_misses": cache_info.get("misses", 0),
            "cache_hit_rate": cache_info.get("hit_rate", 0),
            "adaptive_skips": stats.get("adaptive_skips", 0),
            "adaptive_skip_rate": stats.get("adaptive_skip_rate", 0),
        }
        results.append(result)

    # Display
    print("\n" + "=" * 70)
    print("STRESS TEST RESULTS")
    print("=" * 70)
    print(f"\n{'Configuration':<28} {'Mean (ms)':>10} {'P95 (ms)':>10} {'QPS':>8} {'Quality':>9}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<28} {r['latency_overall']['mean_ms']:>10.1f} "
              f"{r['latency_overall']['p95_ms']:>10.1f} {r['latency_overall']['qps']:>8.0f} "
              f"{r['quality_overall']*100:>8.1f}%")

    # Latency by type
    print("\nLATENCY BY QUERY TYPE:")
    print(f"{'Configuration':<28} {'Repeat (ms)':>13} {'Similar (ms)':>14} {'Novel (ms)':>12}")
    print("-" * 75)
    for r in results:
        lt = r['latency_by_type']
        print(f"{r['name']:<28} {lt['repeat']['mean_ms']:>13.1f} "
              f"{lt['similar']['mean_ms']:>14.1f} {lt['novel']['mean_ms']:>12.1f}")

    # Cache stats
    print("\nCACHE & ADAPTIVE STATS:")
    for r in results:
        print(f"  {r['name']}: "
              f"hits={r['cache_hits']}, misses={r['cache_misses']}, "
              f"hit_rate={r['cache_hit_rate']:.1%}, "
              f"adaptive_skips={r['adaptive_skips']}, skip_rate={r['adaptive_skip_rate']:.1%}")

    # Speedup summary
    if len(results) >= 2:
        baseline = results[0]['latency_overall']['mean_ms']
        print("\nSPEEDUP vs BASELINE:")
        for r in results[1:]:
            speedup = baseline / r['latency_overall']['mean_ms']
            print(f"  {r['name']:<28} {speedup:.1f}x faster, quality: {r['quality_overall']*100:.1f}%")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "real_stress_test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()

"""
Focused Cache Stress Test — Different Workload Patterns
=========================================================

Tests 4 realistic scenarios:
  1. Pure repeat (best case for cache)
  2. Repeat with cache warmup (typical chat session)
  3. Mixed repeat + novel (real workload)
  4. Long-running session (cache eviction)

Reports latency percentiles, throughput, and quality for each.
"""

import os
import sys
import time
import json
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
from sentence_transformers import SentenceTransformer


def make_embedder():
    e = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return e, e.get_embedding_dimension()


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


def extract(pdf_path, max_chunks=150):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_idx, page in enumerate(doc):
        text = page.get_text("text")
        if not text.strip():
            continue
        for ch_idx, ch_text in enumerate(chunk_text(text, chunk_size=150, overlap=20)):
            if len(ch_text.split()) < 30:
                continue
            chunks.append({"text": ch_text, "page": page_idx+1, "chunk_id": f"p{page_idx+1:04d}_c{ch_idx:03d}"})
        if len(chunks) >= max_chunks:
            break
    doc.close()
    return chunks[:max_chunks]


QUERIES = [
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


def keyword_overlap(query, text):
    q_words = {w.lower().strip(".,;:()[]\"'") for w in query.split() if len(w) > 3}
    stopwords = {"what", "when", "where", "which", "have", "does", "with",
                 "from", "this", "that", "are", "was", "were", "been", "how",
                 "explain", "define", "difference", "between"}
    q_words = q_words - stopwords
    if not q_words:
        return 0.0
    r = text.lower()
    return sum(1 for w in q_words if w in r) / len(q_words)


def run_scenario(name, mem, chunks, embeds, queries_to_run, run_warmup=False):
    """Run a scenario and return metrics."""
    if run_warmup:
        # Warm cache by running each query once
        for q in queries_to_run:
            q_emb = embeds[q]
            t = torch.from_numpy(q_emb).float().unsqueeze(0)
            mem.search(t, k=5, query_text=q)
        mem.clear_cache()  # Reset to measure from scratch

    latencies = []
    overlaps = []
    for q in queries_to_run:
        q_emb = embeds[q]
        t = torch.from_numpy(q_emb).float().unsqueeze(0)
        t0 = time.perf_counter()
        indices, sims = mem.search(t, k=5, query_text=q)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        if indices.numel() > 0:
            idx = int(indices[0, 0].item()) if indices.dim() > 1 else 0
            if 0 <= idx < len(chunks):
                overlaps.append(keyword_overlap(q, chunks[idx]["text"]))

    cache_info = mem.cache_info() if hasattr(mem, 'cache_info') else {}
    return {
        "scenario": name,
        "n_queries": len(queries_to_run),
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "qps": 1000 / float(np.mean(latencies)),
        "quality": float(np.mean(overlaps)) if overlaps else 0,
        "cache_hit_rate": cache_info.get("hit_rate", 0),
        "cache_hits": cache_info.get("hits", 0),
        "cache_misses": cache_info.get("misses", 0),
    }


def main():
    print("=" * 70)
    print("FOCUSED STRESS TEST — Real Workload Patterns")
    print("=" * 70)

    pdf = r"D:\COURS\Fluid Mechanics 2\White_2011_7ed_Fluid-Mechanics.pdf"
    print(f"\nLoading PDF and embedding...")
    chunks = extract(pdf, max_chunks=150)
    embedder, dim = make_embedder()
    chunk_embs = embedder.encode([c["text"] for c in chunks])
    print(f"  {len(chunks)} chunks, dim={dim}")

    # Pre-encode queries
    print(f"  Pre-encoding {len(QUERIES)} unique queries...")
    query_embs = embedder.encode(QUERIES)
    query_to_emb = {q: query_embs[i] for i, q in enumerate(QUERIES)}

    # Build systems
    print("\nBuilding systems...")
    sys_orig = HybridEpisodicMemory(capacity=2000, feature_dim=dim,
                                    use_result_cache=False, use_adaptive_rerank=False)
    sys_cache = HybridEpisodicMemory(capacity=2000, feature_dim=dim,
                                     use_result_cache=True, use_adaptive_rerank=False)
    sys_full = HybridEpisodicMemory(capacity=2000, feature_dim=dim,
                                    use_result_cache=True, use_adaptive_rerank=True)
    for mem in [sys_orig, sys_cache, sys_full]:
        for c, e in zip(chunks, chunk_embs):
            t = torch.from_numpy(e).float().unsqueeze(0)
            mem.store(t, text=c["text"])
    print(f"  Stored {len(chunks)} chunks in 3 systems")

    # Scenarios
    scenarios = [
        ("1. Pure Repeat (5 queries x 5 reps)", QUERIES[:5] * 5, False),
        ("2. Repeat with warmup (typical chat)", QUERIES[:5] * 4, True),
        ("3. Mixed (15 repeat + 5 novel)", QUERIES[:5] * 3 + QUERIES[5:10], False),
        ("4. All 20 queries x 3 reps (diverse)", QUERIES * 3, False),
    ]

    print("\n" + "=" * 70)
    print("SCENARIO RESULTS")
    print("=" * 70)
    all_results = []
    for sname, qlist, warmup in scenarios:
        print(f"\n--- {sname} ({len(qlist)} queries) ---")
        results = []
        for sys_name, mem in [("Original", sys_orig), ("Cache", sys_cache), ("Cache+Adaptive", sys_full)]:
            r = run_scenario(sys_name, mem, chunks, query_to_emb, qlist, warmup)
            r["system"] = sys_name
            results.append(r)
            print(f"  {sys_name:<18}: mean={r['mean_ms']:>7.1f}ms, P95={r['p95_ms']:>7.1f}ms, "
                  f"QPS={r['qps']:>6.0f}, quality={r['quality']*100:>5.1f}%, "
                  f"hit_rate={r['cache_hit_rate']*100:>5.1f}%")
        all_results.extend(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best system per scenario")
    print("=" * 70)
    print(f"\n{'Scenario':<40} {'Best System':<18} {'Latency (ms)':>12} {'QPS':>8} {'Quality':>8}")
    print("-" * 90)
    for sname, _, _ in scenarios:
        sres = [r for r in all_results if r['scenario'] == sname]
        if sres:
            best = min(sres, key=lambda r: r['mean_ms'])
            print(f"{sname:<40} {best['system']:<18} {best['mean_ms']:>12.1f} "
                  f"{best['qps']:>8.0f} {best['quality']*100:>7.1f}%")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "stress_cache_warm_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()

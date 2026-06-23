"""
Latency Optimization Benchmark for HybridEpisodicMemory
========================================================

Tests the four latency optimizations:
  Opt 1: Cached cross-encoder (LRU cache by query+doc)
  Opt 2: Batched cross-encoder scoring
  Opt 3: Adaptive re-ranking (skip CE when dense+BM25 agree)
  Opt 4: Distilled cross-encoder (TinyBERT)

Compares cold-cache latency, warm-cache latency, and quality vs original.
"""

import os
import sys
import time
import json
import argparse
import statistics
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import fitz

# UTF-8 output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from mathir_lib import MATHIRPluginV7
from mathir_lib.memory.hybrid_episodic import HybridEpisodicMemory


class RealEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
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
    "How is Mach number defined?",
    "What is the compressibility effect in fluid flow?",
    "Explain the concept of potential flow.",
    "What is the Stokes flow regime?",
    "How do you solve the Laplace equation for flow?",
    "What is the drag coefficient?",
    "Explain the lift force on an airfoil.",
    "What is the Prandtl boundary layer approximation?",
    "How is the Reynolds stress tensor defined?",
    "What is turbulent kinetic energy?",
    "Explain the k-epsilon turbulence model.",
    "What is the law of the wall?",
    "How do you calculate the entrance length in a pipe?",
    "What is the Hagen-Poiseuille flow?",
    "Explain the concept of flow separation.",
    "What is the Buckingham Pi theorem?",
    "How do you nondimensionalize the Navier-Stokes equation?",
    "What is the Euler equation for inviscid flow?",
    "Explain the concept of vorticity dynamics.",
    "What is the circulation theorem?",
    "How does surface tension affect fluid flow?",
    "What is the Weber number?",
    "Explain the concept of cavitation.",
    "What is the Froude number?",
    "How is open channel flow characterized?",
    "What is the Manning equation?",
    "Explain the concept of hydraulic jump.",
    "What is the Chezy formula?",
    "How is the flow rate measured with a Venturi meter?",
    "What is the orifice discharge coefficient?",
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


def evaluate_config(name, memory, query_embs, queries, chunks, warm_cache=True):
    """Run queries and return quality + latency stats."""
    print(f"  Evaluating: {name}")
    latencies = []
    overlaps = []
    hits_30 = []

    # Optional warm-up pass for cache testing
    if warm_cache:
        # Run once to warm cache
        for q_emb, q in zip(query_embs[:5], queries[:5]):
            t = torch.from_numpy(q_emb).float().unsqueeze(0)
            memory.search(t, k=5, query_text=q)
        memory.clear_cache()  # Reset to start fresh

    for q_emb, q in zip(query_embs, queries):
        t = torch.from_numpy(q_emb).float().unsqueeze(0)
        t0 = time.perf_counter()
        indices, sims = memory.search(t, k=5, query_text=q)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        if indices.numel() > 0:
            for ci, c in enumerate(chunks):
                if c["chunk_id"] == memory._texts_for_slots(indices.size(1))[indices[0,0].item() if indices.dim() > 1 else 0] if memory.has_bm25() else None:
                    pass
            # Get top text
            top_text = memory._texts_for_slots(indices.size(1))[indices[0,0].item() if indices.dim() > 1 else 0] if memory.has_bm25() else ""
            if not top_text:
                # Fallback: get from chunk
                for ci, c in enumerate(chunks):
                    if c["chunk_id"] == memory._texts_for_slots(memory.count.item())[indices[0,0].item() if indices.dim() > 1 else 0]:
                        top_text = c["text"]
                        break
            if top_text:
                ov = keyword_overlap(q, top_text)
                overlaps.append(ov)
                hits_30.append(1 if ov >= 0.3 else 0)
    return {
        "latencies": latencies, "overlaps": overlaps, "hits_30": hits_30,
    }


def run_with_repeats(memory, query_embs, queries, chunks, n_repeats=2):
    """Run queries N times to test cache warming."""
    all_latencies = []
    for repeat in range(n_repeats):
        for q_emb, q in zip(query_embs, queries):
            t = torch.from_numpy(q_emb).float().unsqueeze(0)
            t0 = time.perf_counter()
            indices, sims = memory.search(t, k=5, query_text=q)
            all_latencies.append((time.perf_counter() - t0) * 1000)
    return all_latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=r"D:\COURS\Fluid Mechanics 2\White_2011_7ed_Fluid-Mechanics.pdf")
    parser.add_argument("--chunks", type=int, default=100)  # Smaller for speed
    parser.add_argument("--queries", type=int, default=30)
    args = parser.parse_args()

    print("=" * 70)
    print("LATENCY OPTIMIZATION BENCHMARK")
    print("=" * 70)
    print(f"PDF: {os.path.basename(args.pdf)}, Chunks: {args.chunks}, Queries: {args.queries}")

    # Load data
    print("\n[1/3] Loading and embedding...")
    chunks = extract_chunks(args.pdf, max_chunks=args.chunks)
    embedder = RealEmbedder()
    chunk_embs = embedder.encode([c["text"] for c in chunks], show_progress=False)
    queries = (QUERIES * (args.queries // len(QUERIES) + 1))[:args.queries]
    query_embs = embedder.encode(queries, show_progress=False)
    print(f"  {len(chunks)} chunks, {len(queries)} queries, dim={embedder.dim}")

    # Test configurations
    configs = [
        ("D: Cache off, Adaptive off", dict(use_result_cache=False, use_adaptive_rerank=False)),
        ("D + Cache only", dict(use_result_cache=True, use_adaptive_rerank=False)),
        ("D + Adaptive only", dict(use_result_cache=False, use_adaptive_rerank=True)),
        ("D + Cache + Adaptive (all)", dict(use_result_cache=True, use_adaptive_rerank=True)),
    ]

    print(f"\n[2/3] Testing {len(configs)} configurations...")
    results = []
    for name, kwargs in configs:
        try:
            mem = HybridEpisodicMemory(capacity=1000, feature_dim=embedder.dim, **kwargs)
            for c, emb in zip(chunks, chunk_embs):
                t = torch.from_numpy(emb).float().unsqueeze(0)
                mem.store(t, text=c["text"])

            # Measure cold cache + repeated queries
            latencies_cold = []
            latencies_warm = []
            overlaps = []
            hits_30 = []

            # Cold pass
            for q_emb, q in zip(query_embs, queries):
                t = torch.from_numpy(q_emb).float().unsqueeze(0)
                t0 = time.perf_counter()
                indices, sims = mem.search(t, k=5, query_text=q)
                latencies_cold.append((time.perf_counter() - t0) * 1000)
                # Quality
                if indices.numel() > 0:
                    idx = int(indices[0,0].item()) if indices.dim() > 1 else 0
                    if 0 <= idx < len(chunks):
                        top_text = chunks[idx]["text"]
                        ov = keyword_overlap(q, top_text)
                        overlaps.append(ov)
                        hits_30.append(1 if ov >= 0.3 else 0)

            # Warm pass (cache should be populated now)
            for q_emb, q in zip(query_embs, queries):
                t = torch.from_numpy(q_emb).float().unsqueeze(0)
                t0 = time.perf_counter()
                indices, sims = mem.search(t, k=5, query_text=q)
                latencies_warm.append((time.perf_counter() - t0) * 1000)

            # Get cache stats
            stats = mem.get_stats()
            cache_info = mem.cache_info() if hasattr(mem, 'cache_info') else {}

            r = {
                "name": name,
                "cold_mean_ms": statistics.mean(latencies_cold),
                "cold_median_ms": statistics.median(latencies_cold),
                "warm_mean_ms": statistics.mean(latencies_warm),
                "warm_median_ms": statistics.median(latencies_warm),
                "warmup_speedup": statistics.mean(latencies_cold) / max(statistics.mean(latencies_warm), 0.001),
                "quality": statistics.mean(overlaps) if overlaps else 0,
                "hits_30": sum(hits_30),
                "n_queries": len(queries),
                "adaptive_skip_rate": stats.get("adaptive_skip_rate", "N/A"),
                "cache_hit_rate": cache_info.get("hit_rate", "N/A"),
            }
            results.append(r)
            print(f"    Cold: {r['cold_mean_ms']:.1f}ms | Warm: {r['warm_mean_ms']:.1f}ms | "
                  f"Speedup: {r['warmup_speedup']:.1f}x | Quality: {r['quality']*100:.1f}%")
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Display
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Configuration':<30} {'Cold (ms)':>10} {'Warm (ms)':>10} {'Speedup':>8} {'Quality':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<30} {r['cold_mean_ms']:>10.1f} {r['warm_mean_ms']:>10.1f} "
              f"{r['warmup_speedup']:>7.1f}x {r['quality']*100:>7.1f}%")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "latency_optimization_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()

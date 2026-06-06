"""
Quick Experiment: Test if swapping the embedder fixes MATHIR V7.1 D
====================================================================

Hypothesis: MATHIR V7.1 D lost because it uses all-MiniLM-L6-v2 (0.6451 nDCG@10)
while BGE-base-en-v1.5 alone gets 0.7376 nDCG@10. The hybrid pipeline
adds only +3.3pp on top of the embedder.

This experiment runs MATHIR V7.1 D with TWO different embedders to verify.
"""

import os
import sys
import time
import json
import statistics
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the existing benchmark code
from real_sota_benchmark import (
    SCIFACT_DIR, download_scifact, load_scifact,
    ndcg_at_k, mrr_at_k, recall_at_k, evaluate_run
)


def run_mathir_with_embedder(model_name: str, corpus: Dict, queries: Dict, qrels: Dict) -> Dict:
    """Run MATHIR V7.1 D with a configurable embedder."""
    from mathir_lib.memory import HybridEpisodicMemory
    from sentence_transformers import SentenceTransformer

    print(f"\n--- MATHIR V7.1 D with {model_name} ---")
    embedder = SentenceTransformer(model_name)
    dim = embedder.get_sentence_embedding_dimension()
    print(f"  Embedder dim: {dim}")

    # Initialize HybridEpisodicMemory
    mem = HybridEpisodicMemory(
        feature_dim=dim,
        use_cross_encoder=False,  # Skip CE for speed
        use_result_cache=False,
        use_adaptive_rerank=False,
        capacity=len(corpus) + 100,
    )

    # Encode and store
    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[did]["title"] + " " + corpus[did]["text"]) for did in doc_ids]

    t0 = time.perf_counter()
    doc_embeddings = embedder.encode(doc_texts, show_progress_bar=False, batch_size=64,
                                       convert_to_numpy=True, normalize_embeddings=True)
    encode_time = (time.perf_counter() - t0) * 1000
    print(f"  Encode: {encode_time:.0f} ms")

    t0 = time.perf_counter()
    for i, did in enumerate(doc_ids):
        emb_t = torch.from_numpy(doc_embeddings[i]).float()
        mem.store(emb_t, text=corpus[did]["title"] + " " + corpus[did]["text"])
    store_time = (time.perf_counter() - t0) * 1000
    print(f"  Store: {store_time:.0f} ms ({len(doc_ids)} memories)")

    # Encode queries
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = embedder.encode(query_texts, show_progress_bar=False, batch_size=32,
                                         convert_to_numpy=True, normalize_embeddings=True)

    # Search
    results = {}
    latencies = []
    for i, qid in enumerate(query_ids):
        emb_t = torch.from_numpy(query_embeddings[i]).float()
        t0 = time.perf_counter()
        try:
            indices, scores = mem.search(emb_t, query_text=query_texts[i], k=100)
        except Exception as e:
            results[qid] = []
            latencies.append((time.perf_counter() - t0) * 1000)
            continue
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved = []
        if indices is not None and len(indices) > 0:
            first = indices[0]
            if hasattr(first, '__len__'):
                idx_list = [int(x.item() if hasattr(x, 'item') else x) for x in first]
            else:
                idx_list = [int(first.item() if hasattr(first, 'item') else first)]
            for j, doc_idx in enumerate(idx_list):
                if 0 <= doc_idx < len(doc_ids):
                    score = float(scores[0][j].item() if hasattr(scores[0][j], 'item') else scores[0][j])
                    retrieved.append((doc_ids[doc_idx], score))
        results[qid] = retrieved

    metrics = evaluate_run(results, qrels, k_values=[10, 100])
    metrics["latency_mean_ms"] = float(np.mean(latencies))
    metrics["latency_median_ms"] = float(np.median(latencies))
    metrics["encode_time_ms"] = encode_time
    metrics["store_time_ms"] = store_time
    metrics["num_queries"] = len(queries)
    metrics["dim"] = dim
    metrics["embedder"] = model_name

    print(f"  nDCG@10: {metrics['nDCG@10']:.4f}")
    print(f"  MRR@10:  {metrics['MRR@10']:.4f}")
    print(f"  Recall@100: {metrics['Recall@100']:.4f}")
    print(f"  Latency: {metrics['latency_mean_ms']:.1f} ms/query")
    return metrics


def main():
    print("=" * 80)
    print("EXPERIMENT: Test if swapping embedder fixes MATHIR V7.1 D")
    print("=" * 80)
    print("Hypothesis: All-MiniLM-L6-v2 is the bottleneck, not the hybrid pipeline")
    print()

    print("[1/2] Loading SciFact dataset...")
    if not download_scifact():
        print("Failed to download")
        return

    corpus, queries, qrels = load_scifact()
    print(f"  Loaded: {len(corpus)} docs, {len(queries)} queries")

    print("\n[2/2] Running MATHIR with different embedders...")
    results = {}

    # Test with each embedder
    embedders = [
        ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM (current default)"),
        ("BAAI/bge-small-en-v1.5", "BGE-small (SOTA small)"),
        ("BAAI/bge-base-en-v1.5", "BGE-base (SOTA base)"),
    ]

    for model_name, label in embedders:
        try:
            results[label] = run_mathir_with_embedder(model_name, corpus, queries, qrels)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS: Does swapping the embedder help?")
    print("=" * 80)
    print(f"\n{'Embedder':<35} {'nDCG@10':>10} {'MRR@10':>10} {'Latency':>10} {'Store':>10}")
    print("-" * 80)

    for name, m in results.items():
        if "error" in m:
            continue
        print(f"{name:<35} {m['nDCG@10']:>10.4f} {m['MRR@10']:>10.4f} "
              f"{m['latency_mean_ms']:>10.1f} {m['store_time_ms']/1000:>9.1f}s")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    if "MiniLM (current default)" in results and "BGE-base (SOTA base)" in results:
        mini = results["MiniLM (current default)"]["nDCG@10"]
        bge = results["BGE-base (SOTA base)"]["nDCG@10"]
        improvement = bge - mini
        print(f"  Swapping embedder: MiniLM ({mini:.4f}) → BGE-base ({bge:.4f}) = +{improvement:.4f} nDCG@10")
        if improvement > 0.05:
            print(f"  ✓ CONFIRMED: Embedder swap gives +{improvement*100:.1f}pp improvement")
            print(f"  Recommendation: Replace all-MiniLM-L6-v2 with BGE-base-en-v1.5 in MATHIR default config")
        else:
            print(f"  ✗ NOT CONFIRMED: Embedder swap does not significantly improve")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mathir_embedder_swap_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

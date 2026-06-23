"""
Minimal Fast Test: Does swapping the embedder fix MATHIR V7.1 D?
================================================================

Focused experiment: only test 1-2 systems, use cached models, no reranker.
Target runtime: <2 minutes.
"""

import os
import sys
import time
import json
import ssl
import urllib.request
import zipfile
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============ Use existing cached data ============
SCIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beir_data", "scifact")


def find_scifact():
    """Find scifact in cached location."""
    actual = SCIFACT_DIR
    if not os.path.exists(os.path.join(actual, "corpus.jsonl")):
        actual = os.path.join(SCIFACT_DIR, "scifact")
    if not os.path.exists(os.path.join(actual, "corpus.jsonl")):
        # Download
        print("Downloading scifact...")
        os.makedirs(SCIFACT_DIR, exist_ok=True)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        zip_path = os.path.join(SCIFACT_DIR, "scifact.zip")
        with urllib.request.urlopen("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip", context=ctx, timeout=60) as r:
            with open(zip_path, 'wb') as f:
                f.write(r.read())
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(SCIFACT_DIR)
        os.remove(zip_path)
        if os.path.exists(os.path.join(SCIFACT_DIR, "scifact", "corpus.jsonl")):
            actual = os.path.join(SCIFACT_DIR, "scifact")
    return actual


def load_data():
    actual = find_scifact()
    corpus, queries, qrels = {}, {}, {}
    with open(os.path.join(actual, "corpus.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}
    with open(os.path.join(actual, "queries.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q.get("text", "")
    with open(os.path.join(actual, "qrels", "test.tsv"), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 3:
                qrels.setdefault(p[0], {})[p[1]] = int(p[2])
    return corpus, queries, qrels


def dcg_at_k(rel, k):
    rel = rel[:k]
    if not rel: return 0.0
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel))


def ndcg_at_k(rel, k):
    dcg = dcg_at_k(rel, k)
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved, relevant, k):
    if not relevant: return 0.0
    return len(set(retrieved[:k]) & set(relevant.keys())) / len(relevant)


def mrr_at_k(rel, k):
    for i, r in enumerate(rel[:k]):
        if r > 0: return 1.0 / (i + 1)
    return 0.0


def evaluate(results, qrels, ks=(10, 100)):
    metrics = {f"nDCG@{k}": [] for k in ks}
    metrics.update({f"MRR@{k}": [] for k in ks})
    metrics.update({f"Recall@{k}": [] for k in ks})
    for qid, ret in results.items():
        if qid not in qrels: continue
        rels = [qrels[qid].get(d, 0) for d, _ in ret]
        for k in ks:
            metrics[f"nDCG@{k}"].append(ndcg_at_k(rels, k))
            metrics[f"MRR@{k}"].append(mrr_at_k(rels, k))
            metrics[f"Recall@{k}"].append(recall_at_k([d for d, _ in ret], qrels[qid], k))
    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ============ Fast experiment: just FAISS + embedder ============
def run_faiss_with_embedder(model_name, corpus, queries, qrels):
    """Pure FAISS + embedder. NO BM25, NO MATHIR hybrid."""
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"\n=== {model_name} + FAISS (pure dense) ===")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    doc_ids = list(corpus.keys())
    doc_texts = [corpus[d]["title"] + " " + corpus[d]["text"] for d in doc_ids]

    t0 = time.perf_counter()
    doc_emb = model.encode(doc_texts, show_progress_bar=False, batch_size=64,
                            convert_to_numpy=True, normalize_embeddings=True)
    print(f"  Encode: {(time.perf_counter()-t0)*1000:.0f} ms")

    index = faiss.IndexFlatIP(dim)
    index.add(doc_emb.astype("float32"))
    print(f"  Index: {index.ntotal} vectors, dim={dim}")

    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]
    query_emb = model.encode(query_texts, show_progress_bar=False, batch_size=32,
                              convert_to_numpy=True, normalize_embeddings=True)

    # Search
    results = {}
    latencies = []
    for i, qid in enumerate(query_ids):
        t0 = time.perf_counter()
        scores, indices = index.search(query_emb[i:i+1].astype("float32"), 100)
        latencies.append((time.perf_counter()-t0)*1000)
        retrieved = []
        for j in range(indices.shape[1]):
            idx = int(indices[0, j])
            if idx >= 0:
                retrieved.append((doc_ids[idx], float(scores[0, j])))
        results[qid] = retrieved

    metrics = evaluate(results, qrels)
    metrics["latency_ms"] = float(np.mean(latencies))
    metrics["dim"] = dim
    print(f"  nDCG@10: {metrics['nDCG@10']:.4f}")
    print(f"  MRR@10:  {metrics['MRR@10']:.4f}")
    print(f"  Recall@100: {metrics['Recall@100']:.4f}")
    print(f"  Latency: {metrics['latency_ms']:.2f} ms/query")
    return metrics


def main():
    print("=" * 80)
    print("MINIMAL TEST: Does embedder matter?")
    print("=" * 80)

    print("\n[1/2] Loading scifact (cached or download)...")
    corpus, queries, qrels = load_data()
    print(f"  Loaded: {len(corpus)} docs, {len(queries)} queries, {sum(len(v) for v in qrels.values())} qrels")

    print("\n[2/2] Running experiments (only MiniLM and BGE-small - fast)...")

    all_results = {}
    for model in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
        try:
            all_results[model] = run_faiss_with_embedder(model, corpus, queries, qrels)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Pure FAISS + different embedders")
    print("=" * 80)
    print(f"\n{'Embedder':<45} {'nDCG@10':>10} {'MRR@10':>10} {'Recall@100':>12} {'Latency':>10}")
    print("-" * 90)
    for name, m in all_results.items():
        print(f"{name:<45} {m['nDCG@10']:>10.4f} {m['MRR@10']:>10.4f} {m['Recall@100']:>12.4f} {m['latency_ms']:>10.2f}")

    # Reference
    print("\nReference from previous full benchmark:")
    print(f"  BGE-base-en-v1.5 + FAISS:    0.7376 nDCG@10, 4.2 ms")
    print(f"  MATHIR V7.1 D (Hybrid MiniLM): 0.6782 nDCG@10, 83.9 ms")
    print(f"  all-MiniLM-L6-v2 + FAISS:     0.6451 nDCG@10, 2.6 ms")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimal_test_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

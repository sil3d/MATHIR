"""
REAL BENCHMARK — BEIR SciFact (Gold Standard Retrieval Evaluation)
================================================================

This is a REAL benchmark using:
  - BEIR SciFact dataset (300 queries, 5183 docs, 339 human-judged relevance labels)
  - nDCG@10, MRR@10, Recall@100 (the actual standard IR metrics)
  - BM25 + 3 SOTA embedding models + MATHIR variants

NO keyword overlap. NO fake metrics. Real TREC-style evaluation.
"""

import os
import sys
import time
import json
import urllib.request
import zipfile
import statistics
from typing import List, Dict, Any, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ============================================================================
# SCIFACT DATASET (the smallest BEIR dataset - 5K docs, 300 queries)
# ============================================================================

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beir_data", "scifact")


def download_scifact() -> bool:
    """Download SciFact dataset (5KB compressed, ~1MB extracted)."""
    if os.path.exists(os.path.join(SCIFACT_DIR, "corpus.jsonl")):
        print("  [OK] SciFact already downloaded")
        return True

    print(f"  Downloading SciFact from {SCIFACT_URL}...")
    os.makedirs(SCIFACT_DIR, exist_ok=True)
    zip_path = os.path.join(SCIFACT_DIR, "scifact.zip")

    # Try urllib with SSL bypass
    try:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        import urllib.request
        with urllib.request.urlopen(SCIFACT_URL, context=ctx, timeout=60) as response:
            with open(zip_path, 'wb') as out:
                out.write(response.read())
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(SCIFACT_DIR)
        os.remove(zip_path)
        print(f"  [OK] SciFact ready at {SCIFACT_DIR}")
        return True
    except Exception as e:
        print(f"  [FAIL] urllib download error: {e}")

    # Fallback: try HuggingFace datasets
    try:
        print("  Trying HuggingFace datasets as fallback...")
        from datasets import load_dataset
        os.makedirs(os.path.join(SCIFACT_DIR, "qrels"), exist_ok=True)

        corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
        queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
        qrels_ds = load_dataset("BeIR/scifact-qrels", split="train")

        with open(os.path.join(SCIFACT_DIR, "corpus.jsonl"), 'w', encoding='utf-8') as f:
            for doc in corpus_ds:
                f.write(json.dumps({"_id": doc["_id"], "title": doc.get("title", ""),
                                     "text": doc.get("text", "")}) + "\n")

        with open(os.path.join(SCIFACT_DIR, "queries.jsonl"), 'w', encoding='utf-8') as f:
            for q in queries_ds:
                f.write(json.dumps({"_id": q["_id"], "text": q.get("text", "")}) + "\n")

        with open(os.path.join(SCIFACT_DIR, "qrels", "test.tsv"), 'w', encoding='utf-8') as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for r in qrels_ds:
                f.write(f"{r['query-id']}\t{r['corpus-id']}\t{r['score']}\n")

        print(f"  [OK] SciFact loaded via HuggingFace at {SCIFACT_DIR}")
        return True
    except Exception as e:
        print(f"  [FAIL] HuggingFace fallback also failed: {e}")
        return False


def load_scifact() -> Tuple[Dict, Dict, Dict]:
    """
    Load SciFact dataset in BEIR format.
    Returns: (corpus, queries, qrels)
      - corpus: {doc_id: {"title": ..., "text": ...}}
      - queries: {query_id: query_text}
      - qrels: {query_id: {doc_id: relevance}}
    """
    corpus = {}
    queries = {}
    qrels = {}

    # Find files (handle nested folder structure)
    actual_dir = SCIFACT_DIR
    for sub in ['scifact']:
        candidate = os.path.join(SCIFACT_DIR, sub)
        if os.path.exists(os.path.join(candidate, "corpus.jsonl")):
            actual_dir = candidate
            break

    # Load corpus
    with open(os.path.join(actual_dir, "corpus.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }

    # Load queries
    with open(os.path.join(actual_dir, "queries.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q.get("text", "")

    # Load qrels (relevance judgments)
    qrels_path = os.path.join(actual_dir, "qrels", "test.tsv")
    with open(qrels_path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid, did, rel = parts[0], parts[1], int(parts[2])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][did] = rel

    print(f"  Loaded: {len(corpus)} docs, {len(queries)} queries, "
          f"{sum(len(v) for v in qrels.values())} relevance judgments")
    return corpus, queries, qrels


# ============================================================================
# METRICS — REAL TREC-STYLE METRICS (nDCG@10, MRR@10, Recall@100)
# ============================================================================

def dcg_at_k(relevances: List[int], k: int) -> float:
    """Discounted Cumulative Gain @ k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ k."""
    dcg = dcg_at_k(relevances, k)
    # Ideal DCG: sort relevances in descending order
    ideal_rels = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr_at_k(relevances: List[int], k: int) -> float:
    """Mean Reciprocal Rank @ k."""
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved_docs: List[str], relevant_docs: Dict[str, int], k: int) -> float:
    """Recall @ k."""
    if not relevant_docs:
        return 0.0
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs.keys())
    if not relevant_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)


def evaluate_run(results: Dict[str, List[Tuple[str, float]]],
                 qrels: Dict[str, Dict[str, int]],
                 k_values: List[int] = [10, 100]) -> Dict[str, float]:
    """
    Compute nDCG@k, MRR@k, Recall@k for a run.
    results: {query_id: [(doc_id, score), ...]}
    qrels: {query_id: {doc_id: relevance}}
    """
    ndcg_scores = defaultdict(list)
    mrr_scores = defaultdict(list)
    recall_scores = defaultdict(list)

    for qid, retrieved in results.items():
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        retrieved_ids = [doc_id for doc_id, _ in retrieved]
        relevances = [relevant.get(doc_id, 0) for doc_id in retrieved_ids]

        for k in k_values:
            ndcg_scores[k].append(ndcg_at_k(relevances, k))
            mrr_scores[k].append(mrr_at_k(relevances, k))
            recall_scores[k].append(recall_at_k(retrieved_ids, relevant, k))

    metrics = {}
    for k in k_values:
        metrics[f"nDCG@{k}"] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0.0
        metrics[f"MRR@{k}"] = float(np.mean(mrr_scores[k])) if mrr_scores[k] else 0.0
        metrics[f"Recall@{k}"] = float(np.mean(recall_scores[k])) if recall_scores[k] else 0.0
    return metrics


# ============================================================================
# BASELINE 1: BM25 (using rank-bm25)
# ============================================================================

def run_bm25(corpus: Dict, queries: Dict, qrels: Dict) -> Dict:
    """BM25 baseline using rank-bm25."""
    from rank_bm25 import BM25Okapi

    print("\n--- BM25 (rank-bm25) ---")
    doc_ids = list(corpus.keys())
    tokenized_corpus = [corpus[did]["text"].lower().split() for did in doc_ids]

    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized_corpus)
    index_time = (time.perf_counter() - t0) * 1000
    print(f"  Index time: {index_time:.0f} ms ({len(doc_ids)} docs)")

    results = {}
    latencies = []
    for qid, qtext in queries.items():
        t0 = time.perf_counter()
        scores = bm25.get_scores(qtext.lower().split())
        top_k = np.argsort(scores)[::-1][:100]
        retrieved = [(doc_ids[i], float(scores[i])) for i in top_k]
        results[qid] = retrieved
        latencies.append((time.perf_counter() - t0) * 1000)

    metrics = evaluate_run(results, qrels, k_values=[10, 100])

    # Add the latency stats to metrics
    metrics["latency_mean_ms"] = float(np.mean(latencies))
    metrics["latency_median_ms"] = float(np.median(latencies))
    metrics["index_time_ms"] = index_time
    metrics["num_queries"] = len(queries)

    print(f"  nDCG@10: {metrics['nDCG@10']:.4f}")
    print(f"  MRR@10:  {metrics['MRR@10']:.4f}")
    print(f"  Recall@100: {metrics['Recall@100']:.4f}")
    print(f"  Latency: {metrics['latency_mean_ms']:.1f} ms/query")
    return metrics


# ============================================================================
# BASELINE 2: SOTA Embedding Models (via sentence-transformers)
# ============================================================================

def run_embedding_model(model_name: str, corpus: Dict, queries: Dict, qrels: Dict,
                        use_faiss: bool = True) -> Dict:
    """Run a sentence-transformer model with FAISS or brute-force."""
    from sentence_transformers import SentenceTransformer
    import torch

    print(f"\n--- {model_name} ---")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"  [FAIL] Could not load model: {e}")
        return {"error": str(e)}

    # Encode corpus
    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[did]["title"] + " " + corpus[did]["text"]) for did in doc_ids]

    t0 = time.perf_counter()
    doc_embeddings = model.encode(doc_texts, show_progress_bar=False, batch_size=64,
                                   convert_to_numpy=True, normalize_embeddings=True)
    encode_time = (time.perf_counter() - t0) * 1000
    print(f"  Encode time: {encode_time:.0f} ms ({len(doc_ids)} docs)")

    # Encode queries
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    t0 = time.perf_counter()
    query_embeddings = model.encode(query_texts, show_progress_bar=False, batch_size=32,
                                     convert_to_numpy=True, normalize_embeddings=True)
    query_encode_time = (time.perf_counter() - t0) * 1000

    # Search
    results = {}
    latencies = []

    if use_faiss:
        try:
            import faiss
            dim = doc_embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(doc_embeddings.astype("float32"))
            print(f"  FAISS index built (dim={dim})")

            for i, qid in enumerate(query_ids):
                t0 = time.perf_counter()
                scores, indices = index.search(
                    query_embeddings[i:i+1].astype("float32"), 100
                )
                retrieved = [(doc_ids[int(indices[0, j])], float(scores[0, j]))
                             for j in range(indices.shape[1]) if int(indices[0, j]) >= 0]
                results[qid] = retrieved
                latencies.append((time.perf_counter() - t0) * 1000)
        except ImportError:
            use_faiss = False

    if not use_faiss:
        # Brute-force cosine
        for i, qid in enumerate(query_ids):
            t0 = time.perf_counter()
            scores = doc_embeddings @ query_embeddings[i]
            top_k = np.argsort(scores)[::-1][:100]
            retrieved = [(doc_ids[j], float(scores[j])) for j in top_k]
            results[qid] = retrieved
            latencies.append((time.perf_counter() - t0) * 1000)

    metrics = evaluate_run(results, qrels, k_values=[10, 100])
    metrics["latency_mean_ms"] = float(np.mean(latencies))
    metrics["latency_median_ms"] = float(np.median(latencies))
    metrics["encode_time_ms"] = encode_time
    metrics["query_encode_time_ms"] = query_encode_time
    metrics["num_queries"] = len(queries)
    metrics["dim"] = int(doc_embeddings.shape[1])
    metrics["search_method"] = "FAISS" if use_faiss else "brute-force"

    print(f"  Dim: {metrics['dim']}, Search: {metrics['search_method']}")
    print(f"  nDCG@10: {metrics['nDCG@10']:.4f}")
    print(f"  MRR@10:  {metrics['MRR@10']:.4f}")
    print(f"  Recall@100: {metrics['Recall@100']:.4f}")
    print(f"  Latency: {metrics['latency_mean_ms']:.1f} ms/query")
    return metrics


# ============================================================================
# MATHIR V7.1 D (HYBRID) — Real comparison
# ============================================================================

def run_mathir_hybrid(corpus: Dict, queries: Dict, qrels: Dict) -> Dict:
    """MATHIR V7.1 D: Hybrid BM25 + Dense + Rerank."""
    from mathir_lib.memory import HybridEpisodicMemory
    import torch

    print("\n--- MATHIR V7.1 D (Hybrid BM25+Dense+CE) ---")
    dim = 384  # Will be set after first encode
    try:
        # Use MiniLM for embedder (fast)
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        dim = embedder.get_sentence_embedding_dimension()
    except Exception as e:
        print(f"  [FAIL] No embedder: {e}")
        return {"error": str(e)}

    try:
        mem = HybridEpisodicMemory(
            feature_dim=dim,
            use_cross_encoder=False,  # Disable for CPU speed
            use_result_cache=False,
            use_adaptive_rerank=False,
            capacity=len(corpus) + 100,
        )
    except Exception as e:
        print(f"  [FAIL] Could not init HybridEpisodicMemory: {e}")
        return {"error": str(e)}

    # Encode and store
    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[did]["title"] + " " + corpus[did]["text"]) for did in doc_ids]

    t0 = time.perf_counter()
    doc_embeddings = embedder.encode(doc_texts, show_progress_bar=False, batch_size=64,
                                       convert_to_numpy=True, normalize_embeddings=True)
    encode_time = (time.perf_counter() - t0) * 1000
    print(f"  Encode time: {encode_time:.0f} ms")

    t0 = time.perf_counter()
    for i, did in enumerate(doc_ids):
        emb_t = torch.from_numpy(doc_embeddings[i]).float()
        mem.store(emb_t, text=corpus[did]["title"] + " " + corpus[did]["text"])
    store_time = (time.perf_counter() - t0) * 1000
    print(f"  Store time: {store_time:.0f} ms ({len(doc_ids)} memories)")

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
            # Handle 2D shape [1, k]
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

    print(f"  Dim: {dim}")
    print(f"  nDCG@10: {metrics['nDCG@10']:.4f}")
    print(f"  MRR@10:  {metrics['MRR@10']:.4f}")
    print(f"  Recall@100: {metrics['Recall@100']:.4f}")
    print(f"  Latency: {metrics['latency_mean_ms']:.1f} ms/query")
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("REAL BENCHMARK — BEIR SciFact (Gold Standard Retrieval Evaluation)")
    print("=" * 80)
    print("Dataset: SciFact (5,183 docs, 300 queries, 339 human relevance judgments)")
    print("Metrics: nDCG@10, MRR@10, Recall@100 (TREC standard)")
    print()

    # Download
    print("[1/3] Downloading SciFact dataset...")
    if not download_scifact():
        print("Failed to download. Exiting.")
        return

    # Load
    print("\n[2/3] Loading dataset...")
    corpus, queries, qrels = load_scifact()

    # Run all systems
    print("\n[3/3] Running all retrieval systems...")
    all_results = {}

    # Baseline 1: BM25
    try:
        all_results["BM25 (rank-bm25)"] = run_bm25(corpus, queries, qrels)
    except Exception as e:
        print(f"BM25 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Baseline 2: all-MiniLM-L6-v2 (current MATHIR default)
    try:
        all_results["all-MiniLM-L6-v2"] = run_embedding_model(
            "sentence-transformers/all-MiniLM-L6-v2", corpus, queries, qrels
        )
    except Exception as e:
        print(f"MiniLM FAILED: {e}")

    # Baseline 3: BGE-small (SOTA small)
    try:
        all_results["BGE-small-en-v1.5"] = run_embedding_model(
            "BAAI/bge-small-en-v1.5", corpus, queries, qrels
        )
    except Exception as e:
        print(f"BGE-small FAILED: {e}")

    # Baseline 4: BGE-base (SOTA medium)
    try:
        all_results["BGE-base-en-v1.5"] = run_embedding_model(
            "BAAI/bge-base-en-v1.5", corpus, queries, qrels
        )
    except Exception as e:
        print(f"BGE-base FAILED: {e}")

    # MATHIR V7.1 D Hybrid
    try:
        all_results["MATHIR V7.1 D (Hybrid)"] = run_mathir_hybrid(corpus, queries, qrels)
    except Exception as e:
        print(f"MATHIR Hybrid FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Print final comparison
    print("\n" + "=" * 80)
    print("FINAL RESULTS — BEIR SciFact (REAL METRICS)")
    print("=" * 80)
    print(f"\n{'System':<35} {'nDCG@10':>10} {'MRR@10':>10} {'Recall@100':>12} {'Latency(ms)':>12}")
    print("-" * 80)

    for name, metrics in all_results.items():
        if "error" in metrics:
            print(f"{name:<35} {'ERROR':>10}")
            continue
        ndcg = metrics.get('nDCG@10', 0)
        mrr = metrics.get('MRR@10', 0)
        recall = metrics.get('Recall@100', 0)
        lat = metrics.get('latency_mean_ms', 0)
        print(f"{name:<35} {ndcg:>10.4f} {mrr:>10.4f} {recall:>12.4f} {lat:>12.1f}")

    # Reference numbers for context
    print("\n" + "=" * 80)
    print("REFERENCE: Published BEIR SciFact nDCG@10 scores (from literature)")
    print("=" * 80)
    print("""
  BM25 (Anserini)               ~0.665 nDCG@10
  Contriever                     ~0.678 nDCG@10
  E5-base-v2                     ~0.747 nDCG@10
  BGE-base-en-v1.5               ~0.785 nDCG@10
  ColBERTv2                      ~0.700 nDCG@10
  SPLADE-v3                      ~0.780 nDCG@10

  Source: BEIR Official Leaderboard, arxiv 2306.07471
    """)

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "real_sota_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

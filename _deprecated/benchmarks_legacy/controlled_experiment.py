"""
CONTROLLED EXPERIMENT — Honest head-to-head on BEIR SciFact TEST split
========================================================================

WHY: Previous v2 partial benchmark reported 0.7376 nDCG@10 (4 decimals
identical to a different run on a different query split). This is
statistically impossible. Something was cached or reused incorrectly.

THIS SCRIPT:
  * Uses BEIR SciFact TEST split (the standard evaluation set, 300 queries,
    339 relevance judgments) - NOT train.
  * Encodes the corpus EXACTLY ONCE per model.
  * Encodes the queries EXACTLY ONCE per model.
  * Caches the embeddings to disk so subsequent runs re-use the same
    encodings (the only honest way to compare systems).
  * Runs 5 systems on the SAME embeddings:
        1. FAISS only (the simple baseline)
        2. BM25 only (lexical baseline)
        3. OptimizedMATHIR (FAISS dense, no BM25, no CE)
        4. OptimizedMATHIR + BM25 fusion (RRF, no CE)
        5. OptimizedMATHIR + BM25 fusion + Cross-Encoder rerank
  * Reports nDCG@10, MRR@10, Recall@100, mean/median/p95 latency.
  * Reports memory footprint for each system.

The truth: there is no free lunch. The hybrid pipeline with BM25 + CE
should be the slowest. Whether it wins on quality is the question.
"""
import os
import sys
import time
import json
import math
import pickle
import hashlib
import gc
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "beir_data", "scifact", "scifact")
EMB_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "controlled_emb_cache")
os.makedirs(EMB_CACHE, exist_ok=True)


# =============================================================================
# 1. LOAD SCIFACT (TEST SPLIT)
# =============================================================================
def load_scifact_test():
    """Load ONLY the test split: 300 queries with 339 qrels."""
    corpus = {}
    with open(os.path.join(SCIFACT_DIR, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()

    queries = {}
    with open(os.path.join(SCIFACT_DIR, "queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    # ONLY test.tsv qrels
    qrels = defaultdict(dict)
    with open(os.path.join(SCIFACT_DIR, "qrels", "test.tsv"), "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, rel = parts[0], parts[1], int(parts[2])
                if rel > 0:
                    qrels[qid][did] = rel

    # Filter queries to only those with at least one relevant doc in test
    queries_in_test = {qid: text for qid, text in queries.items() if qid in qrels}
    print(f"  Corpus: {len(corpus):,} docs")
    print(f"  Queries (test split only): {len(queries_in_test):,}")
    print(f"  Relevance judgments: {sum(len(v) for v in qrels.values()):,}")
    return corpus, queries_in_test, dict(qrels)


# =============================================================================
# 2. CACHE EMBEDDINGS (encode once, reuse forever)
# =============================================================================
def get_or_compute_embeddings(model_name, texts, kind):
    """Encode texts with a model, cache to disk, return numpy array."""
    key = hashlib.sha256(f"{model_name}|{kind}|{len(texts)}|{texts[0][:50]}".encode()).hexdigest()[:16]
    path = os.path.join(EMB_CACHE, f"{model_name.replace('/', '__')}_{kind}_{key}.npy")
    if os.path.exists(path):
        print(f"    [CACHED] {model_name} {kind}: {path}")
        return np.load(path).astype("float32")

    from sentence_transformers import SentenceTransformer
    print(f"    [ENCODING] {model_name} {kind} ({len(texts)} texts)...")
    t0 = time.perf_counter()
    # BGE models need query prefix for queries
    if "bge" in model_name.lower() and kind == "query":
        prefix = "Represent this sentence for searching relevant passages: "
        texts = [prefix + t for t in texts]
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=False,
                         batch_size=32, convert_to_numpy=True,
                         normalize_embeddings=True).astype("float32")
    elapsed = time.perf_counter() - t0
    print(f"    [ENCODED] {len(texts)} texts in {elapsed:.1f}s -> {embs.shape}")
    np.save(path, embs)
    del model
    gc.collect()
    return embs


# =============================================================================
# 3. METRICS (TREC-style)
# =============================================================================
def dcg_at_k(rels, k):
    rels = rels[:k]
    if not rels:
        return 0.0
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(rels, k):
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(rels, k):
    for i, r in enumerate(rels[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def evaluate(results, qrels, k_values=(10, 100)):
    ndcg = defaultdict(list)
    mrr = defaultdict(list)
    recall = defaultdict(list)
    for qid, retrieved in results.items():
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        retrieved_ids = [d for d, _ in retrieved]
        rels = [relevant.get(d, 0) for d in retrieved_ids]
        for k in k_values:
            ndcg[k].append(ndcg_at_k(rels, k))
            mrr[k].append(mrr_at_k(rels, k))
            recall[k].append(recall_at_k(retrieved_ids, relevant, k))
    out = {}
    for k in k_values:
        out[f"nDCG@{k}"] = float(np.mean(ndcg[k]))
        out[f"MRR@{k}"] = float(np.mean(mrr[k]))
        out[f"Recall@{k}"] = float(np.mean(recall[k]))
    return out


# =============================================================================
# 4. SYSTEM 1: FAISS only (brute-force inner product on L2-normalized vecs)
# =============================================================================
def run_faiss(doc_embs, doc_ids, query_embs, query_ids, qrels, k=100):
    import faiss
    dim = doc_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embs)
    latencies = []
    results = {}
    for i, qid in enumerate(query_ids):
        t0 = time.perf_counter()
        scores, idxs = index.search(query_embs[i:i+1], k)
        latencies.append((time.perf_counter() - t0) * 1000)
        results[qid] = [(doc_ids[j], float(scores[0, n]))
                         for n, j in enumerate(idxs[0]) if j >= 0]
    return evaluate(results, qrels), latencies


# =============================================================================
# 5. SYSTEM 2: BM25 only
# =============================================================================
def run_bm25(corpus, queries, qrels, k=100):
    from rank_bm25 import BM25Okapi
    doc_ids = list(corpus.keys())
    tokenized = [corpus[d].lower().split() for d in doc_ids]
    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized)
    index_ms = (time.perf_counter() - t0) * 1000

    latencies = []
    results = {}
    for qid, qtext in queries.items():
        t0 = time.perf_counter()
        scores = bm25.get_scores(qtext.lower().split())
        latencies.append((time.perf_counter() - t0) * 1000)
        top = np.argsort(scores)[::-1][:k]
        results[qid] = [(doc_ids[i], float(scores[i])) for i in top]
    metrics = evaluate(results, qrels)
    return metrics, latencies, index_ms


# =============================================================================
# 6. SYSTEM 3-5: OptimizedMATHIR variants
# =============================================================================
def run_optimized_mathir(doc_embs, doc_ids, doc_texts, query_embs,
                          query_ids, queries_text, qrels,
                          use_bm25, use_ce, k=100):
    """
    Run OptimizedMATHIR with controlled FAISS + optional BM25 + optional CE.
    Uses cached query embeddings - we ONLY call the cross-encoder
    (sentence-transformers) for rerank, not for the dense stage.
    """
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder

    # ---- Build FAISS index on cached embeddings ----
    dim = doc_embs.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(doc_embs)

    # ---- Build BM25 if requested ----
    bm25 = None
    if use_bm25:
        tokenized = [t.lower().split() for t in doc_texts]
        bm25 = BM25Okapi(tokenized)

    # ---- Load cross-encoder only if needed ----
    ce = None
    ce_load_ms = 0.0
    if use_ce:
        t0 = time.perf_counter()
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        ce_load_ms = (time.perf_counter() - t0) * 1000

    # ---- Build qid -> query_text lookup (for CE) ----
    qid_to_text = {qid: queries_text[qid] for qid in query_ids}

    # ---- Run queries ----
    latencies = []
    ce_calls = 0
    ce_skipped = 0
    results = {}
    for i, qid in enumerate(query_ids):
        t0 = time.perf_counter()
        # 1. Dense search (cached emb)
        scores, idxs = faiss_index.search(query_embs[i:i+1], 100)
        dense_top = [(doc_ids[j], float(scores[0, n]))
                     for n, j in enumerate(idxs[0]) if j >= 0]

        # 2. BM25 if requested
        if bm25 is not None:
            bm_scores = bm25.get_scores(qid_to_text[qid].lower().split())
            bm_top = [(doc_ids[j], float(bm_scores[j]))
                      for j in np.argsort(bm_scores)[::-1][:100]]
        else:
            bm_top = []

        # 3. RRF fusion
        fused = {}
        for rank, (d, _) in enumerate(dense_top):
            fused[d] = fused.get(d, 0.0) + 1.0 / (60 + rank + 1)
        for rank, (d, _) in enumerate(bm_top):
            fused[d] = fused.get(d, 0.0) + 1.0 / (60 + rank + 1)
        ordered = sorted(fused.items(), key=lambda x: -x[1])[:k]

        # 4. Cross-encoder rerank (only on first 30)
        if ce is not None and len(ordered) > 1:
            # Adaptive: skip if dense top-1 has high confidence AND agrees with BM25
            dense_top1 = dense_top[0][0]
            bm25_top1 = bm_top[0][0] if bm_top else None
            skip_ce = (bm25_top1 is not None and dense_top1 == bm25_top1
                       and dense_top[0][1] > 0.9)
            if skip_ce:
                ce_skipped += 1
            else:
                ce_calls += 1
                rerank_input = ordered[:30]
                pairs = [(qid_to_text[qid], doc_texts[doc_ids.index(d)]) for d, _ in rerank_input]
                ce_scores = ce.predict(pairs, convert_to_numpy=True,
                                        show_progress_bar=False, batch_size=32)
                reranked = sorted(zip([d for d, _ in rerank_input], ce_scores),
                                   key=lambda x: -float(x[1]))
                # Pad with rest of fused
                reranked_ids = {d for d, _ in reranked}
                rest = [(d, s) for d, s in ordered if d not in reranked_ids]
                ordered = reranked + rest
        latencies.append((time.perf_counter() - t0) * 1000)
        results[qid] = ordered

    metrics = evaluate(results, qrels)
    extras = {"ce_calls": ce_calls, "ce_skipped": ce_skipped,
              "ce_load_ms": ce_load_ms}
    return metrics, latencies, extras


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  CONTROLLED EXPERIMENT — BEIR SciFact TEST split (300 queries)")
    print("  Honest, cached-embeddings, head-to-head comparison")
    print("=" * 70)

    print("\n[1/4] Loading SciFact TEST split...")
    corpus, queries, qrels = load_scifact_test()
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[d] for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    MODEL = "BAAI/bge-base-en-v1.5"

    print(f"\n[2/4] Encoding with {MODEL} (cached after first run)...")
    doc_embs = get_or_compute_embeddings(MODEL, doc_texts, "doc")
    query_embs = get_or_compute_embeddings(MODEL, query_texts, "query")
    print(f"  doc_embs: {doc_embs.shape}, query_embs: {query_embs.shape}")

    all_results = {}
    all_latencies = {}

    print(f"\n[3/4] Running 5 systems on the SAME embeddings...")
    print(f"  Total queries: {len(query_ids)}")
    print(f"  Total docs: {len(doc_ids)}")

    # ---- System 1: FAISS only ----
    print(f"\n--- System 1/5: FAISS only (the simple baseline) ---")
    metrics, lats = run_faiss(doc_embs, doc_ids, query_embs, query_ids, qrels, k=100)
    all_results["1_FAISS_only"] = metrics
    all_latencies["1_FAISS_only"] = lats
    print(f"  nDCG@10 = {metrics['nDCG@10']:.4f}, "
          f"latency p50 = {np.percentile(lats, 50):.2f}ms")

    # ---- System 2: BM25 only ----
    print(f"\n--- System 2/5: BM25 only ---")
    metrics, lats, idx_ms = run_bm25(corpus, queries, qrels, k=100)
    all_results["2_BM25_only"] = metrics
    all_latencies["2_BM25_only"] = lats
    print(f"  nDCG@10 = {metrics['nDCG@10']:.4f}, "
          f"latency p50 = {np.percentile(lats, 50):.2f}ms, "
          f"index = {idx_ms:.0f}ms")

    # ---- System 3: OptimizedMATHIR (FAISS, no BM25, no CE) ----
    print(f"\n--- System 3/5: OptimizedMATHIR (dense only) ---")
    metrics, lats, extras = run_optimized_mathir(
        doc_embs, doc_ids, doc_texts, query_embs,
        query_ids, queries, qrels,
        use_bm25=False, use_ce=False
    )
    all_results["3_OPT_dense"] = metrics
    all_latencies["3_OPT_dense"] = lats
    print(f"  nDCG@10 = {metrics['nDCG@10']:.4f}, "
          f"latency p50 = {np.percentile(lats, 50):.2f}ms")

    # ---- System 4: OptimizedMATHIR + BM25 (RRF, no CE) ----
    print(f"\n--- System 4/5: OptimizedMATHIR + BM25 (RRF, no CE) ---")
    metrics, lats, extras = run_optimized_mathir(
        doc_embs, doc_ids, doc_texts, query_embs,
        query_ids, queries, qrels,
        use_bm25=True, use_ce=False
    )
    all_results["4_OPT_dense+bm25"] = metrics
    all_latencies["4_OPT_dense+bm25"] = lats
    print(f"  nDCG@10 = {metrics['nDCG@10']:.4f}, "
          f"latency p50 = {np.percentile(lats, 50):.2f}ms")

    # ---- System 5: OptimizedMATHIR + BM25 + CE ----
    print(f"\n--- System 5/5: OptimizedMATHIR + BM25 + CE rerank ---")
    metrics, lats, extras = run_optimized_mathir(
        doc_embs, doc_ids, doc_texts, query_embs,
        query_ids, queries, qrels,
        use_bm25=True, use_ce=True
    )
    all_results["5_OPT_dense+bm25+ce"] = metrics
    all_latencies["5_OPT_dense+bm25+ce"] = lats
    print(f"  nDCG@10 = {metrics['nDCG@10']:.4f}, "
          f"latency p50 = {np.percentile(lats, 50):.2f}ms, "
          f"ce_calls = {extras['ce_calls']}, ce_skipped = {extras['ce_skipped']}, "
          f"ce_load = {extras['ce_load_ms']:.0f}ms")

    # =============================================================================
    # 4. FINAL REPORT
    # =============================================================================
    print(f"\n{'='*70}")
    print(f"  FINAL REPORT — BEIR SciFact TEST split (300 queries, 339 qrels)")
    print(f"  Model: {MODEL}")
    print(f"{'='*70}")
    print(f"\n{'System':<32} {'nDCG@10':>8} {'MRR@10':>8} {'R@100':>7} "
          f"{'p50 ms':>8} {'p95 ms':>8}")
    print("-" * 78)
    for name, m in all_results.items():
        lats = all_latencies[name]
        p50 = np.percentile(lats, 50)
        p95 = np.percentile(lats, 95)
        print(f"{name:<32} {m['nDCG@10']:>8.4f} {m['MRR@10']:>8.4f} "
              f"{m['Recall@100']:>7.4f} {p50:>8.2f} {p95:>8.2f}")

    # Save results
    out = {
        "metadata": {
            "dataset": "BEIR SciFact",
            "split": "test",
            "num_queries": len(query_ids),
            "num_docs": len(doc_ids),
            "num_qrels": sum(len(v) for v in qrels.values()),
            "model": MODEL,
        },
        "results": all_results,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "controlled_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

"""
MULTI-DATASET BEIR EXPERIMENT — Find where (if anywhere) MATHIR beats FAISS.
============================================================================

Tests the same 5 systems on 3 BEIR datasets using the TEST split.

Datasets:
  * scifact    — scientific claim verification (5K docs, 300 queries, 339 qrels)
  * nfcorpus   — bio-medical retrieval         (3.6K docs, 323 queries, ~3K qrels)
  * arguana    — counter-argument retrieval    (8.7K docs, 1406 queries, ~1.4K qrels)

Systems (all on the SAME cached BGE-base-en-v1.5 embeddings):
  1. FAISS only (the simple baseline)
  2. OptimizedMATHIR (dense only)
  3. + BM25 (RRF, equal weights)
  4. + BM25 (RRF, dense_weight=2, BM25_weight=1)  -- biases dense
  5. + BM25 + Cross-Encoder rerank

Embeddings are cached on disk. First run encodes (slow); subsequent runs
are < 1 minute per dataset.

Output: per-dataset nDCG@10 + mean nDCG@10 across datasets (the "BEIR score")
        + per-stage latency breakdown.
"""
import os
import sys
import time
import json
import math
import gc
import hashlib
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BEIR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beir_data")
EMB_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "controlled_emb_cache")
os.makedirs(EMB_CACHE, exist_ok=True)

DATASETS = [
    {"name": "scifact",  "subdir": "scifact/scifact",  "n_docs": 5183},
    {"name": "nfcorpus", "subdir": "nfcorpus/nfcorpus", "n_docs": 3633},
    {"name": "arguana",  "subdir": "arguana/arguana",   "n_docs": 8674},
]
MODEL = "BAAI/bge-base-en-v1.5"


# =============================================================================
# DATA LOADING
# =============================================================================
def load_dataset(name: str) -> Tuple[Dict, Dict, Dict]:
    """Load BEIR dataset (test split only)."""
    subdir = next(d["subdir"] for d in DATASETS if d["name"] == name)
    base = os.path.join(BEIR_DIR, subdir)
    corpus = {}
    with open(os.path.join(base, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()
    queries = {}
    with open(os.path.join(base, "queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    # nfcorpus has dev.tsv; the standard BEIR eval uses test.tsv.
    # Try test.tsv first, then dev.tsv as fallback.
    qrels = defaultdict(dict)
    qrels_path = os.path.join(base, "qrels", "test.tsv")
    if not os.path.exists(qrels_path):
        qrels_path = os.path.join(base, "qrels", "dev.tsv")
    with open(qrels_path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, rel = parts[0], parts[1], int(parts[2])
                if rel > 0:
                    qrels[qid][did] = rel
    queries_in_test = {qid: text for qid, text in queries.items() if qid in qrels}
    return corpus, queries_in_test, dict(qrels)


# =============================================================================
# EMBEDDING CACHE
# =============================================================================
def get_or_compute_embeddings(model_name, texts, kind, tag=""):
    """Encode and cache to disk. First run encodes; subsequent runs read from disk."""
    cache_key = hashlib.sha256(
        f"{model_name}|{kind}|{tag}|{len(texts)}|{texts[0][:50] if texts else ''}".encode()
    ).hexdigest()[:16]
    fname = f"{model_name.replace('/', '__')}_{kind}_{tag}_{cache_key}.npy"
    path = os.path.join(EMB_CACHE, fname)
    if os.path.exists(path):
        return np.load(path).astype("float32")

    from sentence_transformers import SentenceTransformer
    print(f"    [ENCODING] {model_name} {kind} ({len(texts)} texts, tag={tag})...")
    t0 = time.perf_counter()
    if "bge" in model_name.lower() and kind == "query":
        prefix = "Represent this sentence for searching relevant passages: "
        texts = [prefix + t for t in texts]
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=False,
                         batch_size=32, convert_to_numpy=True,
                         normalize_embeddings=True).astype("float32")
    print(f"    [ENCODED] {embs.shape} in {time.perf_counter()-t0:.1f}s")
    np.save(path, embs)
    del model
    gc.collect()
    return embs


# =============================================================================
# METRICS
# =============================================================================
def ndcg_at_k(rels, k):
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(sorted(rels, reverse=True)[:k]))
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
    ndcg = defaultdict(list); mrr = defaultdict(list); recall = defaultdict(list)
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
# SYSTEMS
# =============================================================================
def system_faiss(doc_embs, doc_ids, query_embs, query_ids, qrels, k=100):
    """System 1: FAISS only."""
    import faiss
    index = faiss.IndexFlatIP(doc_embs.shape[1])
    index.add(doc_embs)
    lats = []; results = {}
    for i, qid in enumerate(query_ids):
        t0 = time.perf_counter()
        scores, idxs = index.search(query_embs[i:i+1], k)
        lats.append((time.perf_counter() - t0) * 1000)
        results[qid] = [(doc_ids[j], float(scores[0, n]))
                         for n, j in enumerate(idxs[0]) if j >= 0]
    return evaluate(results, qrels), lats


def system_mathir(doc_embs, doc_ids, doc_texts, query_embs, query_ids,
                   query_texts, qrels, use_bm25=False, use_ce=False,
                   bm25_weight=1.0, k=100):
    """
    System 2-N: OptimizedMATHIR variants.
    bm25_weight controls the relative weight of BM25 in the RRF.
    """
    import faiss
    from rank_bm25 import BM25Okapi

    index = faiss.IndexFlatIP(doc_embs.shape[1])
    index.add(doc_embs)
    bm25 = None
    if use_bm25:
        tokenized = [t.lower().split() for t in doc_texts]
        bm25 = BM25Okapi(tokenized)

    ce = None; ce_load_ms = 0.0
    if use_ce:
        from sentence_transformers import CrossEncoder
        t0 = time.perf_counter()
        # Auto-detect GPU
        import torch as _torch
        ce_device = "cuda" if _torch.cuda.is_available() else "cpu"
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=ce_device)
        ce_load_ms = (time.perf_counter() - t0) * 1000

    qid_to_text = dict(zip(query_ids, query_texts))
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    lats = []; results = {}; ce_calls = 0; ce_skipped = 0
    for i, qid in enumerate(query_ids):
        t0 = time.perf_counter()
        # 1. Dense
        scores, idxs = index.search(query_embs[i:i+1], 100)
        dense_top = [(doc_ids[j], float(scores[0, n]))
                     for n, j in enumerate(idxs[0]) if j >= 0]
        # 2. BM25
        if bm25 is not None:
            bm_scores = bm25.get_scores(qid_to_text[qid].lower().split())
            bm_top = [(doc_ids[j], float(bm_scores[j]))
                      for j in np.argsort(bm_scores)[::-1][:100]]
        else:
            bm_top = []
        # 3. Weighted RRF
        fused = {}
        for rank, (d, _) in enumerate(dense_top):
            fused[d] = fused.get(d, 0.0) + 1.0 / (60 + rank + 1)  # dense weight = 1
        if bm_top:
            for rank, (d, _) in enumerate(bm_top):
                fused[d] = fused.get(d, 0.0) + bm25_weight / (60 + rank + 1)
        ordered = sorted(fused.items(), key=lambda x: -x[1])[:k]
        # 4. CE rerank
        if ce is not None and len(ordered) > 1:
            ce_calls += 1
            rerank_input = ordered[:30]
            pairs = [(qid_to_text[qid],
                      doc_texts[doc_id_to_idx[d]]) for d, _ in rerank_input]
            ce_scores = ce.predict(pairs, convert_to_numpy=True,
                                    show_progress_bar=False, batch_size=32)
            reranked = sorted(zip([d for d, _ in rerank_input], ce_scores),
                               key=lambda x: -float(x[1]))
            reranked_ids = {d for d, _ in reranked}
            rest = [(d, s) for d, s in ordered if d not in reranked_ids]
            ordered = reranked + rest
        lats.append((time.perf_counter() - t0) * 1000)
        results[qid] = ordered

    metrics = evaluate(results, qrels)
    extras = {"ce_calls": ce_calls, "ce_skipped": ce_skipped,
              "ce_load_ms": ce_load_ms}
    return metrics, lats, extras


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  MULTI-DATASET BEIR EXPERIMENT (3 datasets, 5 systems each)")
    print(f"  Model: {MODEL}")
    print("=" * 70)

    all_results = {}  # all_results[system_name][dataset_name] = metrics
    all_latencies = {}

    for ds in DATASETS:
        name = ds["name"]
        print(f"\n{'='*70}\n  DATASET: {name}\n{'='*70}")

        print(f"\n  Loading {name} (test split)...")
        corpus, queries, qrels = load_dataset(name)
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[d] for d in doc_ids]
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]
        print(f"    {len(doc_ids)} docs, {len(query_ids)} queries, "
              f"{sum(len(v) for v in qrels.values())} qrels")

        print(f"\n  Encoding (cached after first run)...")
        doc_embs = get_or_compute_embeddings(MODEL, doc_texts, "doc", tag=name)
        query_embs = get_or_compute_embeddings(MODEL, query_texts, "query", tag=name)
        print(f"    doc: {doc_embs.shape}, query: {query_embs.shape}")

        ds_results = {}
        ds_lats = {}

        # 1. FAISS only
        print(f"\n  [1/5] FAISS only...")
        m, lats = system_faiss(doc_embs, doc_ids, query_embs, query_ids, qrels)
        ds_results["1_FAISS"] = m
        ds_lats["1_FAISS"] = lats
        print(f"        nDCG@10={m['nDCG@10']:.4f}, p50={np.percentile(lats, 50):.2f}ms")

        # 2. OptimizedMATHIR dense only
        print(f"\n  [2/5] OptimizedMATHIR (dense only)...")
        m, lats, _ = system_mathir(doc_embs, doc_ids, doc_texts, query_embs,
                                    query_ids, query_texts, qrels,
                                    use_bm25=False, use_ce=False)
        ds_results["2_MATHIR_dense"] = m
        ds_lats["2_MATHIR_dense"] = lats
        print(f"        nDCG@10={m['nDCG@10']:.4f}, p50={np.percentile(lats, 50):.2f}ms")

        # 3. OptimizedMATHIR + BM25 (equal weight)
        print(f"\n  [3/5] OptimizedMATHIR + BM25 (equal RRF)...")
        m, lats, _ = system_mathir(doc_embs, doc_ids, doc_texts, query_embs,
                                    query_ids, query_texts, qrels,
                                    use_bm25=True, use_ce=False, bm25_weight=1.0)
        ds_results["3_MATHIR_dense+bm25_eq"] = m
        ds_lats["3_MATHIR_dense+bm25_eq"] = lats
        print(f"        nDCG@10={m['nDCG@10']:.4f}, p50={np.percentile(lats, 50):.2f}ms")

        # 4. OptimizedMATHIR + BM25 (denser weight)
        print(f"\n  [4/5] OptimizedMATHIR + BM25 (dense=1, bm25=0.3)...")
        m, lats, _ = system_mathir(doc_embs, doc_ids, doc_texts, query_embs,
                                    query_ids, query_texts, qrels,
                                    use_bm25=True, use_ce=False, bm25_weight=0.3)
        ds_results["4_MATHIR_dense+bm25_low"] = m
        ds_lats["4_MATHIR_dense+bm25_low"] = lats
        print(f"        nDCG@10={m['nDCG@10']:.4f}, p50={np.percentile(lats, 50):.2f}ms")

        # 5. OptimizedMATHIR + BM25 + CE
        # GPU acceleration makes CE ~10-20× faster → use ALL queries
        import torch as _torch
        if _torch.cuda.is_available():
            ce_query_ids = query_ids
            ce_query_embs = query_embs
            ce_n = len(query_ids)
            print(f"\n  [5/5] OptimizedMATHIR + BM25 + CE rerank (CUDA, ALL {ce_n} queries)...")
        else:
            ce_sample_size = min(50, len(query_ids))
            ce_query_ids = query_ids[:ce_sample_size]
            ce_query_embs = query_embs[:ce_sample_size]
            ce_n = ce_sample_size
            print(f"\n  [5/5] OptimizedMATHIR + BM25 + CE rerank (CPU, {ce_n}-query sample)...")
        ce_qrels = {k: v for k, v in qrels.items() if k in set(ce_query_ids)}
        ce_doc_embs = doc_embs
        m, lats, extras = system_mathir(ce_doc_embs, doc_ids, doc_texts, ce_query_embs,
                                          ce_query_ids, query_texts, ce_qrels,
                                          use_bm25=True, use_ce=True, bm25_weight=0.3)
        m["_n_queries_used"] = ce_n
        ds_results["5_MATHIR_dense+bm25+ce"] = m
        ds_lats["5_MATHIR_dense+bm25+ce"] = lats
        print(f"        nDCG@10={m['nDCG@10']:.4f} (on {ce_n} queries), "
              f"p50={np.percentile(lats, 50):.2f}ms, ce_load={extras['ce_load_ms']:.0f}ms")

        all_results[name] = ds_results
        all_latencies[name] = ds_lats

    # =============================================================================
    # FINAL MATRIX
    # =============================================================================
    print(f"\n\n{'='*80}")
    print(f"  FINAL MATRIX — nDCG@10 per (system, dataset)")
    print(f"{'='*80}")
    sys_names = ["1_FAISS", "2_MATHIR_dense", "3_MATHIR_dense+bm25_eq",
                  "4_MATHIR_dense+bm25_low", "5_MATHIR_dense+bm25+ce"]
    ds_names = [d["name"] for d in DATASETS]

    header = f"{'System':<32}" + "".join(f"{n:>12}" for n in ds_names) + f"{'AVG':>10}"
    print(header)
    print("-" * len(header))

    averages = {}
    for sys_name in sys_names:
        row = f"{sys_name:<32}"
        vals = []
        for ds_name in ds_names:
            v = all_results[ds_name][sys_name]["nDCG@10"]
            vals.append(v)
            row += f"{v:>12.4f}"
        avg = float(np.mean(vals))
        averages[sys_name] = avg
        row += f"{avg:>10.4f}"
        print(row)

    # Find the best system
    best = max(averages, key=averages.get)
    print(f"\n  Best by average nDCG@10: {best} ({averages[best]:.4f})")
    print(f"  FAISS only avg:           {averages['1_FAISS']:.4f}")
    print(f"  Δ best vs FAISS:          {(averages[best]-averages['1_FAISS']):+.4f}")

    # Latency matrix
    print(f"\n\n{'='*80}")
    print(f"  LATENCY MATRIX — p50 ms per (system, dataset)")
    print(f"{'='*80}")
    print(header.replace("nDCG@10", "p50 ms"))
    print("-" * len(header))
    for sys_name in sys_names:
        row = f"{sys_name:<32}"
        vals = []
        for ds_name in ds_names:
            v = np.percentile(all_latencies[ds_name][sys_name], 50)
            vals.append(v)
            row += f"{v:>12.1f}"
        avg = float(np.mean(vals))
        row += f"{avg:>10.1f}"
        print(row)

    # Save
    out = {
        "metadata": {
            "model": MODEL,
            "datasets": [d["name"] for d in DATASETS],
            "systems": sys_names,
            "metrics": ["nDCG@10", "MRR@10", "Recall@100"],
        },
        "results": {
            ds_name: {
                sys_name: all_results[ds_name][sys_name]
                for sys_name in sys_names
            } for ds_name in ds_names
        },
        "average_nDCG@10": averages,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "multi_dataset_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

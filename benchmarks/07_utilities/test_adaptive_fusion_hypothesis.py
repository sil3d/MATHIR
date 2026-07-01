#!/usr/bin/env python3
"""
Validates, empirically and locally (no API needed), the hypothesis behind
a proposed "confidence-gated adaptive fusion" architecture for MATHIR's
hybrid_search, before implementing it in the live server.

Hypothesis: RRF fusion with BM25 should only be invoked when the dense
(vector) ranking is AMBIGUOUS -- i.e. when its top-1/top-2 score margin is
small. When the dense ranking is already confident (large margin), fusing
in BM25 should on average hurt more than it helps (consistent with the
already-established finding that hybrid trails pure-dense on nfcorpus/
scifact with a stronger embedder).

Method: bucket queries by their dense top1-top2 score margin (low vs high
confidence), then compare "trust dense-only" vs "always fuse with BM25"
nDCG@10 *within each bucket* on nfcorpus, using MATHIR's real embedder and
real rrf_fusion() function. If the hypothesis holds, hybrid should win (or
be neutral) in the low-confidence bucket and lose in the high-confidence
bucket.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MCP_ROOT = Path(__file__).resolve().parent.parent.parent / "mathir_mcp"
sys.path.insert(0, str(MCP_ROOT / "mathir_lib"))
from mathir_search import rrf_fusion, _tokenize  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEIR_DATA_DIR = Path(__file__).resolve().parent.parent / "05_test_data" / "beir_data"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATASET = "nfcorpus"


def load_dataset(name):
    data_path = BEIR_DATA_DIR / name
    nested = data_path / name
    if nested.exists():
        data_path = nested
    return GenericDataLoader(data_path).load(split="test")


def main():
    corpus, queries, qrels = load_dataset(DATASET)
    corpus_ids = list(corpus.keys())
    corpus_texts = [(corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip() for cid in corpus_ids]
    q_ids = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]

    print(f"Loading embedder: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME, device=DEVICE)
    corpus_embs = encoder.encode(corpus_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    query_embs = encoder.encode(q_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    print("Building FAISS dense ranking + margins...")
    import faiss
    dim = corpus_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embs.astype(np.float32))
    scores, indices = index.search(query_embs.astype(np.float32), 100)

    margins = scores[:, 0] - scores[:, 1]  # top1 - top2 raw score gap per query

    print("Building BM25 ranking...")
    tokenized_corpus = [_tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Build per-query dense-only and hybrid-fused result dicts.
    dense_results = {}
    hybrid_results = {}
    for i, qid in enumerate(q_ids):
        vector_results = [(corpus_ids[indices[i][j]], float(scores[i][j])) for j in range(100) if indices[i][j] < len(corpus_ids)]
        dense_results[qid] = {mid: sc for mid, sc in vector_results}

        bm25_scores = bm25.get_scores(_tokenize(q_texts[i]))
        bm25_results = sorted(
            [(corpus_ids[k], float(bm25_scores[k])) for k in range(len(corpus_ids)) if bm25_scores[k] > 0],
            key=lambda x: x[1], reverse=True,
        )[:100]
        fused = rrf_fusion(vector_results, bm25_results, vector_weight=1.0, bm25_weight=1.0)
        hybrid_results[qid] = {mid: sc for mid, sc in fused[:100]}

    # Bucket queries by margin (median split -- low vs high confidence).
    median_margin = float(np.median(margins))
    low_conf_qids = [q_ids[i] for i in range(len(q_ids)) if margins[i] <= median_margin]
    high_conf_qids = [q_ids[i] for i in range(len(q_ids)) if margins[i] > median_margin]
    print(f"\nMedian top1-top2 margin: {median_margin:.4f}")
    print(f"Low-confidence bucket (margin <= median):  {len(low_conf_qids)} queries")
    print(f"High-confidence bucket (margin > median):  {len(high_conf_qids)} queries")

    evaluator = EvaluateRetrieval()

    def eval_subset(results: dict, qids: list, sub_qrels: dict):
        sub_results = {q: results[q] for q in qids if q in results}
        ndcg, _map, _rc, _r = evaluator.evaluate(sub_qrels, sub_results, k_values=[10])
        return ndcg["NDCG@10"]

    for label, qids in [("LOW-confidence (ambiguous dense ranking)", low_conf_qids),
                         ("HIGH-confidence (clear dense winner)", high_conf_qids)]:
        sub_qrels = {q: qrels[q] for q in qids if q in qrels}
        dense_ndcg = eval_subset(dense_results, qids, sub_qrels)
        hybrid_ndcg = eval_subset(hybrid_results, qids, sub_qrels)
        delta = hybrid_ndcg - dense_ndcg
        verdict = "hybrid HELPS" if delta > 0.01 else ("hybrid HURTS" if delta < -0.01 else "no meaningful difference")
        print(f"\n--- {label} (n={len(qids)}) ---")
        print(f"  dense-only nDCG@10:  {dense_ndcg:.4f}")
        print(f"  hybrid nDCG@10:      {hybrid_ndcg:.4f}")
        print(f"  delta: {delta:+.4f}  ({verdict})")

    print("\n=== HYPOTHESIS CHECK ===")
    print("Expected if adaptive fusion is a sound architecture: hybrid helps (or is neutral)")
    print("in the LOW-confidence bucket, and hurts (or is neutral) in the HIGH-confidence bucket.")


if __name__ == "__main__":
    main()

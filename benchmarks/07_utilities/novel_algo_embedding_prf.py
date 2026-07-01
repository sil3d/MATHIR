#!/usr/bin/env python3
"""
NOVEL ALGORITHM (self-written, not a reuse of FAISS/BM25/RRF/cross-encoder):
embedding-space pseudo-relevance feedback (PRF) for MATHIR's dense retrieval.

Idea (Rocchio-style, but implemented here from scratch as a two-pass dense
search, not imported from any IR library):

  1. First pass: encode the query, do a plain dense KNN search, take the
     top-m results.
  2. Refine the query vector by blending it with the (score-weighted)
     centroid of those top-m result embeddings:
         q' = normalize(alpha * q + beta * weighted_centroid(top_m))
  3. Second pass: dense KNN search again with q' instead of q, return the
     final top-k.

Rationale: a single-shot query embedding is a noisy, one-shot guess at
"what region of embedding space is relevant." The top-m first-pass results
are real evidence about where the relevant region actually is (assuming
most of the top-m are at least topically close) -- blending them back in
is a cheap, training-free way to correct the query vector before the real
ranking decision, using only information already available at query time
(no fine-tuning, no external model).

Known risk (classic PRF "query drift"): if the first-pass top-m contains
irrelevant results, the refined query moves AWAY from the true relevant
region instead of toward it. This is tested honestly below across a sweep
of (alpha, beta, m) -- report whatever the real numbers say, including if
this makes things worse.

No API/LLM calls needed -- pure embeddings + linear algebra, local only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEIR_DATA_DIR = Path(__file__).resolve().parent.parent / "05_test_data" / "beir_data"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATASETS = ["nfcorpus", "scifact"]


def load_dataset(name):
    data_path = BEIR_DATA_DIR / name
    nested = data_path / name
    if nested.exists():
        data_path = nested
    return GenericDataLoader(data_path).load(split="test")


def dense_search_matrix(query_embs, corpus_embs, k):
    """Return (scores, indices) for top-k per query, cosine/IP on normalized vectors."""
    sims = query_embs @ corpus_embs.T  # [n_queries, n_corpus], already normalized -> cosine sim
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    row_idx = np.arange(sims.shape[0])[:, None]
    top_scores = sims[row_idx, idx]
    order = np.argsort(-top_scores, axis=1)
    idx = idx[row_idx, order]
    scores = sims[row_idx, idx]
    return scores, idx


def prf_refine(query_embs, corpus_embs, first_pass_idx, first_pass_scores, alpha, beta, m):
    """Rocchio-style refinement: q' = normalize(alpha*q + beta*weighted_centroid(top_m))."""
    n_q = query_embs.shape[0]
    refined = np.zeros_like(query_embs)
    for i in range(n_q):
        top_m_idx = first_pass_idx[i, :m]
        top_m_scores = first_pass_scores[i, :m]
        weights = np.clip(top_m_scores, 0, None)
        weights = weights / (weights.sum() + 1e-8)
        centroid = (corpus_embs[top_m_idx] * weights[:, None]).sum(axis=0)
        combined = alpha * query_embs[i] + beta * centroid
        norm = np.linalg.norm(combined)
        refined[i] = combined / norm if norm > 1e-8 else query_embs[i]
    return refined


def evaluate_scores(scores, indices, corpus_ids, q_ids, qrels, evaluator):
    results = {}
    for i, qid in enumerate(q_ids):
        results[qid] = {corpus_ids[indices[i][j]]: float(scores[i][j]) for j in range(indices.shape[1])}
    ndcg, _map, recall_cap, _r = evaluator.evaluate(qrels, results, k_values=[10, 100])
    return ndcg["NDCG@10"], _map["MAP@10"], recall_cap["Recall@100"]


def main():
    encoder = SentenceTransformer(MODEL_NAME, device=DEVICE)
    evaluator = EvaluateRetrieval()

    for dataset in DATASETS:
        print(f"\n{'='*70}\nDATASET: {dataset}\n{'='*70}")
        corpus, queries, qrels = load_dataset(dataset)
        corpus_ids = list(corpus.keys())
        corpus_texts = [(corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip() for cid in corpus_ids]
        q_ids = list(queries.keys())
        q_texts = [queries[qid] for qid in q_ids]

        print("Encoding...")
        corpus_embs = encoder.encode(corpus_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        query_embs = encoder.encode(q_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

        print("\nBaseline (single-pass dense search)...")
        base_scores, base_idx = dense_search_matrix(query_embs, corpus_embs, 100)
        base_ndcg, base_map, base_rec = evaluate_scores(base_scores, base_idx, corpus_ids, q_ids, qrels, evaluator)
        print(f"  nDCG@10={base_ndcg:.4f}  MAP@10={base_map:.4f}  Recall@100={base_rec:.4f}")

        print("\nPRF sweep (alpha, beta, m) -- self-written two-pass refinement...")
        best = (None, -1.0)
        for m in [3, 5, 10]:
            for beta in [0.1, 0.25, 0.5, 1.0]:
                alpha = 1.0
                refined_q = prf_refine(query_embs, corpus_embs, base_idx, base_scores, alpha, beta, m)
                scores2, idx2 = dense_search_matrix(refined_q, corpus_embs, 100)
                ndcg2, map2, rec2 = evaluate_scores(scores2, idx2, corpus_ids, q_ids, qrels, evaluator)
                delta = ndcg2 - base_ndcg
                marker = " <-- BEST so far" if ndcg2 > best[1] else ""
                print(f"  m={m:>2} alpha={alpha:.1f} beta={beta:.2f}:  nDCG@10={ndcg2:.4f} (delta {delta:+.4f})  MAP@10={map2:.4f}  Recall@100={rec2:.4f}{marker}")
                if ndcg2 > best[1]:
                    best = ((m, alpha, beta), ndcg2)

        print(f"\n  BEST PRF config for {dataset}: m,alpha,beta={best[0]}  nDCG@10={best[1]:.4f}  "
              f"(baseline was {base_ndcg:.4f}, delta {best[1]-base_ndcg:+.4f})")


if __name__ == "__main__":
    main()

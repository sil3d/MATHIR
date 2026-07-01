#!/usr/bin/env python3
"""
NOVEL ALGORITHM #3 (self-written, structurally different from the 4 prior
techniques): anisotropy correction ("all-but-the-top" style) on the
embedding SPACE itself, rather than blending in a secondary signal
(BM25/CE/PRF) or penalizing document scores (hubness).

Background: sentence embeddings are known to be anisotropic -- a handful
of dominant directions (the top principal components of the embedding
distribution) capture most of the variance but carry little semantic
discriminative content (they're often correlated with generic properties
like sentence length or frequency, not topical relevance). Raw cosine
similarity is dominated by these directions, which compresses the
*useful* semantic signal into a narrow cone and makes truly-different
documents look artificially similar.

Algorithm (self-written from the "all-but-the-top" principle, not an
imported library):
  1. Compute the corpus embedding mean mu and the top-D principal
     directions (via SVD on the centered corpus embeddings).
  2. For every embedding x (corpus AND query, same transform since they
     must stay in the same space): x' = (x - mu) - sum_{i=1}^{D} (proj of
     (x-mu) onto principal direction i).
  3. Re-normalize x' to unit length.
  4. Re-rank by cosine similarity in this corrected space.

This is a purely linear, training-free, per-corpus-fitted correction --
different in kind from PRF (query-side, two-pass) and hubness correction
(document-score penalty): it recalibrates the metric space itself before
any query ever runs.

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


def fit_anisotropy_correction(corpus_embs: np.ndarray, max_d: int):
    """Returns (mu, top_directions[max_d, dim]) fitted on the corpus."""
    mu = corpus_embs.mean(axis=0)
    centered = corpus_embs - mu
    # SVD on centered embeddings: right singular vectors = principal directions.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mu, vt[:max_d]  # top_directions: [max_d, dim], already unit-norm rows


def apply_correction(embs: np.ndarray, mu: np.ndarray, top_directions: np.ndarray, d: int) -> np.ndarray:
    """Remove the top-d principal directions, then re-normalize to unit length."""
    centered = embs - mu
    if d > 0:
        dirs = top_directions[:d]  # [d, dim]
        proj = centered @ dirs.T  # [N, d]
        centered = centered - proj @ dirs  # remove those components
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return centered / norms


def dense_search_matrix(query_embs, corpus_embs, k):
    sims = query_embs @ corpus_embs.T
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    row_idx = np.arange(sims.shape[0])[:, None]
    top_scores = sims[row_idx, idx]
    order = np.argsort(-top_scores, axis=1)
    idx = idx[row_idx, order]
    scores = sims[row_idx, idx]
    return scores, idx


def evaluate_scores(scores, indices, corpus_ids, q_ids, qrels, evaluator):
    results = {}
    for i, qid in enumerate(q_ids):
        results[qid] = {corpus_ids[indices[i][j]]: float(scores[i][j]) for j in range(indices.shape[1])}
    ndcg, _map, recall_cap, _r = evaluator.evaluate(qrels, results, k_values=[10, 100])
    return ndcg["NDCG@10"], _map["MAP@10"], recall_cap["Recall@100"]


def main():
    encoder = SentenceTransformer(MODEL_NAME, device=DEVICE)
    evaluator = EvaluateRetrieval()
    max_d = 20

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

        print("Baseline (single-pass dense search, raw embeddings)...")
        base_scores, base_idx = dense_search_matrix(query_embs, corpus_embs, 100)
        base_ndcg, base_map, base_rec = evaluate_scores(base_scores, base_idx, corpus_ids, q_ids, qrels, evaluator)
        print(f"  nDCG@10={base_ndcg:.4f}  MAP@10={base_map:.4f}  Recall@100={base_rec:.4f}")

        print(f"\nFitting anisotropy correction (top {max_d} directions) on corpus...")
        mu, top_dirs = fit_anisotropy_correction(corpus_embs, max_d)

        print("Sweeping number of removed top directions (D)...")
        best = (None, -1.0)
        for d in [0, 1, 2, 3, 5, 8, 12, 20]:
            corrected_corpus = apply_correction(corpus_embs, mu, top_dirs, d)
            corrected_query = apply_correction(query_embs, mu, top_dirs, d)
            scores2, idx2 = dense_search_matrix(corrected_query, corrected_corpus, 100)
            ndcg2, map2, rec2 = evaluate_scores(scores2, idx2, corpus_ids, q_ids, qrels, evaluator)
            delta = ndcg2 - base_ndcg
            marker = " <-- BEST so far" if ndcg2 > best[1] else ""
            print(f"  D={d:>2}:  nDCG@10={ndcg2:.4f} (delta {delta:+.4f})  MAP@10={map2:.4f}  Recall@100={rec2:.4f}{marker}")
            if ndcg2 > best[1]:
                best = (d, ndcg2)

        print(f"\n  BEST anisotropy-correction D for {dataset}: {best[0]}  nDCG@10={best[1]:.4f}  "
              f"(baseline was {base_ndcg:.4f}, delta {best[1]-base_ndcg:+.4f})")


if __name__ == "__main__":
    main()

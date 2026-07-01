#!/usr/bin/env python3
"""
NOVEL ALGORITHM #2 (self-written, different failure mode than PRF):
document-side "hubness" bias correction for dense retrieval.

Background: in high-dimensional embedding spaces, some documents become
disproportionately-frequent nearest neighbors across many unrelated
queries ("hubs") -- an intrinsic-dimensionality artifact, not a semantic
property. If a document is a structural hub, it gets an inflated
similarity score for many queries regardless of true relevance, pushing
genuinely relevant documents out of the top-k. This is a *document-side*
bias (unlike PRF/BM25/CE, which all operate on the query side or blend in
an external signal) -- worth testing because it targets a completely
different failure mode than the three already-rejected ideas.

Algorithm (self-written, not an import of an existing hubness-reduction
library):
  1. For each corpus document d, precompute its "hub score" h(d) = mean
     cosine similarity of d to a random sample of R other corpus documents.
     A document embedded near the "center of mass" of the corpus (close to
     everything) gets a high h(d); a document in a genuinely distinctive
     region gets a low h(d).
  2. At query time, adjust the raw similarity score:
         adjusted_score(q, d) = cos(q, d) - lambda * h(d)
     penalizing documents that are hubs (structurally close to everything)
     proportionally to how hub-like they are.
  3. Re-rank by adjusted_score instead of raw cosine similarity.

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
R_SAMPLE = 500  # random sample size for hub-score estimation (cheap, not O(N^2))


def load_dataset(name):
    data_path = BEIR_DATA_DIR / name
    nested = data_path / name
    if nested.exists():
        data_path = nested
    return GenericDataLoader(data_path).load(split="test")


def compute_hub_scores(corpus_embs: np.ndarray, r_sample: int, seed: int = 42) -> np.ndarray:
    """h(d) = mean cosine similarity of d to a random sample of r_sample other docs."""
    rng = np.random.RandomState(seed)
    n = corpus_embs.shape[0]
    sample_idx = rng.choice(n, size=min(r_sample, n), replace=False)
    sample_embs = corpus_embs[sample_idx]  # [R, dim], already normalized
    sims = corpus_embs @ sample_embs.T  # [N, R] cosine sims (normalized vectors)
    return sims.mean(axis=1)  # [N]


def dense_search_matrix(query_embs, corpus_embs, k):
    sims = query_embs @ corpus_embs.T
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    row_idx = np.arange(sims.shape[0])[:, None]
    top_scores = sims[row_idx, idx]
    order = np.argsort(-top_scores, axis=1)
    idx = idx[row_idx, order]
    scores = sims[row_idx, idx]
    return scores, idx, sims


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

        print("Baseline (single-pass dense search)...")
        base_scores, base_idx, full_sims = dense_search_matrix(query_embs, corpus_embs, 100)
        base_ndcg, base_map, base_rec = evaluate_scores(base_scores, base_idx, corpus_ids, q_ids, qrels, evaluator)
        print(f"  nDCG@10={base_ndcg:.4f}  MAP@10={base_map:.4f}  Recall@100={base_rec:.4f}")

        print(f"\nComputing hub scores (R={R_SAMPLE} sample)...")
        hub_scores = compute_hub_scores(corpus_embs, R_SAMPLE)
        print(f"  hub_score range: [{hub_scores.min():.4f}, {hub_scores.max():.4f}], mean={hub_scores.mean():.4f}, std={hub_scores.std():.4f}")

        print("\nHubness-correction sweep (lambda)...")
        best = (None, -1.0)
        for lam in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
            adjusted_sims = full_sims - lam * hub_scores[None, :]
            k = 100
            idx = np.argpartition(-adjusted_sims, kth=min(k, adjusted_sims.shape[1] - 1), axis=1)[:, :k]
            row_idx = np.arange(adjusted_sims.shape[0])[:, None]
            top_adj = adjusted_sims[row_idx, idx]
            order = np.argsort(-top_adj, axis=1)
            idx2 = idx[row_idx, order]
            scores2 = top_adj[row_idx, order]  # report the RAW cosine-derived rank via adjusted score for evaluation
            ndcg2, map2, rec2 = evaluate_scores(scores2, idx2, corpus_ids, q_ids, qrels, evaluator)
            delta = ndcg2 - base_ndcg
            marker = " <-- BEST so far" if ndcg2 > best[1] else ""
            print(f"  lambda={lam:.2f}:  nDCG@10={ndcg2:.4f} (delta {delta:+.4f})  MAP@10={map2:.4f}  Recall@100={rec2:.4f}{marker}")
            if ndcg2 > best[1]:
                best = (lam, ndcg2)

        print(f"\n  BEST hubness-correction lambda for {dataset}: {best[0]}  nDCG@10={best[1]:.4f}  "
              f"(baseline was {base_ndcg:.4f}, delta {best[1]-base_ndcg:+.4f})")


if __name__ == "__main__":
    main()

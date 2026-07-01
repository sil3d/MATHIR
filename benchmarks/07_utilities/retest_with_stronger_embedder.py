#!/usr/bin/env python3
"""
Re-tests the "helps weak baseline / hurts strong baseline" pattern found
across all five rejected techniques (see report.md section 4 / DIMENSIONS.md)
using intfloat/multilingual-e5-small (the stronger, same-footprint
candidate embedder from compare_embedding_models.py) instead of the
current weaker default.

Prediction being tested: with a stronger baseline dense signal, hybrid
BM25 fusion and embedding-space PRF should now HURT (not help) -- flipping
the pattern observed with the weaker embedder. If confirmed, this is a
clean, generalizable rule: "disable augmentation techniques once the
embedder is strong enough" -- actionable and simple, no per-query gating
needed (consistent with the already-rejected confidence-gating hypothesis).

No API/LLM calls needed -- pure embeddings + linear algebra, local only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import sys
MCP_ROOT = Path(__file__).resolve().parent.parent.parent / "mathir_mcp"
sys.path.insert(0, str(MCP_ROOT / "mathir_lib"))
from mathir_search import rrf_fusion, _tokenize  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEIR_DATA_DIR = Path(__file__).resolve().parent.parent / "05_test_data" / "beir_data"
MODEL_NAME = "intfloat/multilingual-e5-small"  # the STRONGER same-footprint candidate
DATASETS = ["nfcorpus", "scifact"]
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def load_dataset(name):
    data_path = BEIR_DATA_DIR / name
    nested = data_path / name
    if nested.exists():
        data_path = nested
    return GenericDataLoader(data_path).load(split="test")


def dense_search_matrix(query_embs, corpus_embs, k):
    sims = query_embs @ corpus_embs.T
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    row_idx = np.arange(sims.shape[0])[:, None]
    top_scores = sims[row_idx, idx]
    order = np.argsort(-top_scores, axis=1)
    idx = idx[row_idx, order]
    scores = sims[row_idx, idx]
    return scores, idx


def prf_refine(query_embs, corpus_embs, first_pass_idx, first_pass_scores, beta, m):
    n_q = query_embs.shape[0]
    refined = np.zeros_like(query_embs)
    for i in range(n_q):
        top_m_idx = first_pass_idx[i, :m]
        top_m_scores = first_pass_scores[i, :m]
        weights = np.clip(top_m_scores, 0, None)
        weights = weights / (weights.sum() + 1e-8)
        centroid = (corpus_embs[top_m_idx] * weights[:, None]).sum(axis=0)
        combined = query_embs[i] + beta * centroid
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
    print(f"Loading STRONGER embedder: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME, device=DEVICE)
    evaluator = EvaluateRetrieval()

    for dataset in DATASETS:
        print(f"\n{'='*70}\nDATASET: {dataset}\n{'='*70}")
        corpus, queries, qrels = load_dataset(dataset)
        corpus_ids = list(corpus.keys())
        corpus_texts_raw = [(corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip() for cid in corpus_ids]
        corpus_texts = [PASSAGE_PREFIX + t for t in corpus_texts_raw]
        q_ids = list(queries.keys())
        q_texts = [QUERY_PREFIX + queries[qid] for qid in q_ids]

        print("Encoding (E5 prefixes applied)...")
        corpus_embs = encoder.encode(corpus_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        query_embs = encoder.encode(q_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

        print("\n--- Baseline: dense-only (e5-small) ---")
        base_scores, base_idx = dense_search_matrix(query_embs, corpus_embs, 100)
        base_ndcg, base_map, base_rec = evaluate_scores(base_scores, base_idx, corpus_ids, q_ids, qrels, evaluator)
        print(f"  nDCG@10={base_ndcg:.4f}  MAP@10={base_map:.4f}  Recall@100={base_rec:.4f}")

        print("\n--- Test A: RRF hybrid fusion (BM25 + e5-small dense, weights 1.0/1.0) ---")
        tokenized_corpus = [_tokenize(t) for t in corpus_texts_raw]
        bm25 = BM25Okapi(tokenized_corpus)
        hybrid_results = {}
        for i, qid in enumerate(q_ids):
            vector_results = [(corpus_ids[base_idx[i][j]], float(base_scores[i][j])) for j in range(100)]
            bm25_scores = bm25.get_scores(_tokenize(queries[qid]))
            bm25_results = sorted(
                [(corpus_ids[k], float(bm25_scores[k])) for k in range(len(corpus_ids)) if bm25_scores[k] > 0],
                key=lambda x: x[1], reverse=True,
            )[:100]
            fused = rrf_fusion(vector_results, bm25_results, vector_weight=1.0, bm25_weight=1.0)
            hybrid_results[qid] = {mid: sc for mid, sc in fused[:100]}
        ndcg_dict, map_dict, recall_cap_dict, _r = evaluator.evaluate(qrels, hybrid_results, k_values=[10, 100])
        ndcg_h, map_h, rec_h = ndcg_dict["NDCG@10"], map_dict["MAP@10"], recall_cap_dict["Recall@100"]
        delta_h = ndcg_h - base_ndcg
        print(f"  nDCG@10={ndcg_h:.4f} (delta {delta_h:+.4f})  MAP@10={map_h:.4f}  Recall@100={rec_h:.4f}")
        print(f"  PREDICTION CHECK: {'CONFIRMED (hybrid now hurts, as predicted)' if delta_h < -0.005 else 'NOT CONFIRMED (hybrid still helps or neutral even with stronger embedder)'}")

        print("\n--- Test B: embedding PRF (beta=0.25, m=5, e5-small) ---")
        refined_q = prf_refine(query_embs, corpus_embs, base_idx, base_scores, beta=0.25, m=5)
        scores2, idx2 = dense_search_matrix(refined_q, corpus_embs, 100)
        ndcg_p, map_p, rec_p = evaluate_scores(scores2, idx2, corpus_ids, q_ids, qrels, evaluator)
        delta_p = ndcg_p - base_ndcg
        print(f"  nDCG@10={ndcg_p:.4f} (delta {delta_p:+.4f})  MAP@10={map_p:.4f}  Recall@100={rec_p:.4f}")
        print(f"  PREDICTION CHECK: {'CONFIRMED (PRF now hurts, as predicted)' if delta_p < -0.005 else 'NOT CONFIRMED (PRF still helps or neutral even with stronger embedder)'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Completes the 3-embedder x 2-dataset matrix for the hybrid-fusion-flip
investigation (retest_with_stronger_embedder.py + investigate_hybrid_flip_
factor.py only covered some cells). Runs all 3 embedders (default,
e5-small, bge-base-en-v1.5) on BOTH nfcorpus and scifact, reporting
baseline nDCG@10, hybrid nDCG@10, delta, and several candidate explanatory
statistics (top1-top10 gap, BM25/dense score-scale ratio) side by side,
to look for ANY consistent pattern across the full matrix rather than
2-3 cherry-picked comparisons.

No API/LLM calls needed -- pure embeddings + linear algebra, local only.
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
DATASETS = ["nfcorpus", "scifact"]

EMBEDDERS = {
    "default": {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "q_prefix": "", "p_prefix": ""},
    "e5-small": {"name": "intfloat/multilingual-e5-small", "q_prefix": "query: ", "p_prefix": "passage: "},
    "bge-base": {"name": "BAAI/bge-base-en-v1.5", "q_prefix": "", "p_prefix": ""},
}


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


def evaluate_scores(scores, indices, corpus_ids, q_ids, qrels, evaluator):
    results = {}
    for i, qid in enumerate(q_ids):
        results[qid] = {corpus_ids[indices[i][j]]: float(scores[i][j]) for j in range(indices.shape[1])}
    ndcg, _map, recall_cap, _r = evaluator.evaluate(qrels, results, k_values=[10, 100])
    return ndcg["NDCG@10"]


def main():
    rows = []
    for dataset in DATASETS:
        corpus, queries, qrels = load_dataset(dataset)
        corpus_ids = list(corpus.keys())
        corpus_texts_raw = [(corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip() for cid in corpus_ids]
        q_ids = list(queries.keys())
        q_texts_raw = [queries[qid] for qid in q_ids]

        evaluator = EvaluateRetrieval()
        tokenized_corpus = [_tokenize(t) for t in corpus_texts_raw]
        bm25 = BM25Okapi(tokenized_corpus)
        # BM25 raw-score scale reference (independent of embedder).
        sample_bm25_scores = np.concatenate([bm25.get_scores(_tokenize(q_texts_raw[i])) for i in range(min(20, len(q_ids)))])
        bm25_scale = float(np.median(sample_bm25_scores[sample_bm25_scores > 0])) if (sample_bm25_scores > 0).any() else 0.0

        for emb_label, cfg in EMBEDDERS.items():
            print(f"[{dataset}] Loading {emb_label} ({cfg['name']})...")
            encoder = SentenceTransformer(cfg["name"], device=DEVICE)
            corpus_texts = [cfg["p_prefix"] + t for t in corpus_texts_raw]
            q_texts = [cfg["q_prefix"] + t for t in q_texts_raw]
            corpus_embs = encoder.encode(corpus_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            query_embs = encoder.encode(q_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

            base_scores, base_idx = dense_search_matrix(query_embs, corpus_embs, 100)
            base_ndcg = evaluate_scores(base_scores, base_idx, corpus_ids, q_ids, qrels, evaluator)

            top1 = base_scores[:, 0]
            top10 = base_scores[:, 9]
            gap = float(((top1 - top10) / (np.abs(top1) + 1e-8)).mean())
            dense_scale = float(np.median(top1))

            hybrid_results = {}
            for i, qid in enumerate(q_ids):
                vector_results = [(corpus_ids[base_idx[i][j]], float(base_scores[i][j])) for j in range(100)]
                bm25_scores = bm25.get_scores(_tokenize(q_texts_raw[i]))
                bm25_r = sorted(
                    [(corpus_ids[k], float(bm25_scores[k])) for k in range(len(corpus_ids)) if bm25_scores[k] > 0],
                    key=lambda x: x[1], reverse=True,
                )[:100]
                fused = rrf_fusion(vector_results, bm25_r, vector_weight=1.0, bm25_weight=1.0)
                hybrid_results[qid] = {mid: sc for mid, sc in fused[:100]}
            ndcg_dict, _m, _rc, _r = evaluator.evaluate(qrels, hybrid_results, k_values=[10, 100])
            hybrid_ndcg = ndcg_dict["NDCG@10"]

            delta = hybrid_ndcg - base_ndcg
            rows.append((dataset, emb_label, base_ndcg, hybrid_ndcg, delta, gap, dense_scale, bm25_scale))

            del encoder
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print(f"\n{'dataset':<10}{'embedder':<12}{'baseline':>10}{'hybrid':>10}{'delta':>9}{'top1-10 gap':>13}{'dense scale':>13}{'bm25 scale':>12}")
    print("-" * 90)
    for r in rows:
        print(f"{r[0]:<10}{r[1]:<12}{r[2]:>10.4f}{r[3]:>10.4f}{r[4]:>+9.4f}{r[5]:>13.4f}{r[6]:>13.4f}{r[7]:>12.4f}")

    print("\nLooking for ANY consistent ordering between delta and the candidate stats above.")


if __name__ == "__main__":
    main()

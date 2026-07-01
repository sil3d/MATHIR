#!/usr/bin/env python3
"""
Follow-up to retest_with_stronger_embedder.py: investigates WHAT differs
between intfloat/multilingual-e5-small (hybrid fusion HELPS, per the
previous test) and BAAI/bge-base-en-v1.5 (hybrid fusion HURTS, per the
pre-existing multi_dataset_efficient.py benchmark), since embedder
"strength" alone doesn't explain the difference (bge-base is the
stronger of the two on scifact: baseline dense-only ~0.744 vs e5-small's
0.677).

Hypothesis being tested: the deciding factor is the SCORE DISTRIBUTION
SHAPE of the dense ranking, not overall retrieval quality. RRF only uses
RANK, not the raw score value, so if a dense ranking's top results are
already sharply separated from the rest (a "peaked" score distribution),
BM25's rank-based contribution is more likely to demote a correct answer
that the dense model was already confident about. If the dense ranking is
"flatter" (many closely-scored candidates), BM25's independent signal adds
real information rather than just noise.

Measures, for each embedder: the score gap between top-1 and top-10
(normalized), and re-confirms the hybrid-fusion delta, to see if they
correlate across the three embedders tested this session.

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
DATASET = "scifact"  # the dataset where the e5-small/bge-base split is clearest

EMBEDDERS = {
    "default (paraphrase-multilingual-MiniLM-L12-v2)": {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "q_prefix": "", "p_prefix": ""},
    "e5-small (hybrid HELPS, per prior test)": {"name": "intfloat/multilingual-e5-small", "q_prefix": "query: ", "p_prefix": "passage: "},
    "bge-base (hybrid HURTS, per pre-existing benchmark)": {"name": "BAAI/bge-base-en-v1.5", "q_prefix": "", "p_prefix": ""},
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
    return ndcg["NDCG@10"], _map["MAP@10"], recall_cap["Recall@100"]


def main():
    corpus, queries, qrels = load_dataset(DATASET)
    corpus_ids = list(corpus.keys())
    corpus_texts_raw = [(corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip() for cid in corpus_ids]
    q_ids = list(queries.keys())
    q_texts_raw = [queries[qid] for qid in q_ids]

    evaluator = EvaluateRetrieval()
    tokenized_corpus_cache = [_tokenize(t) for t in corpus_texts_raw]
    bm25 = BM25Okapi(tokenized_corpus_cache)

    print(f"{'Embedder':<55}{'baseline':>10}{'hybrid':>10}{'delta':>9}{'top1-top10 gap (norm)':>24}")
    print("-" * 108)

    for label, cfg in EMBEDDERS.items():
        encoder = SentenceTransformer(cfg["name"], device=DEVICE)
        corpus_texts = [cfg["p_prefix"] + t for t in corpus_texts_raw]
        q_texts = [cfg["q_prefix"] + t for t in q_texts_raw]

        corpus_embs = encoder.encode(corpus_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        query_embs = encoder.encode(q_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

        base_scores, base_idx = dense_search_matrix(query_embs, corpus_embs, 100)
        base_ndcg, _map, _rec = evaluate_scores(base_scores, base_idx, corpus_ids, q_ids, qrels, evaluator)

        # Score distribution shape: normalized gap between top-1 and top-10
        # score, averaged across all queries. Larger gap = more "peaked"/
        # confident dense ranking; smaller gap = "flatter" ranking.
        top1 = base_scores[:, 0]
        top10 = base_scores[:, 9]
        gap = (top1 - top10) / (np.abs(top1) + 1e-8)
        mean_gap = float(gap.mean())

        hybrid_results = {}
        for i, qid in enumerate(q_ids):
            vector_results = [(corpus_ids[base_idx[i][j]], float(base_scores[i][j])) for j in range(100)]
            bm25_scores = bm25.get_scores(_tokenize(q_texts_raw[i]))
            bm25_results = sorted(
                [(corpus_ids[k], float(bm25_scores[k])) for k in range(len(corpus_ids)) if bm25_scores[k] > 0],
                key=lambda x: x[1], reverse=True,
            )[:100]
            fused = rrf_fusion(vector_results, bm25_results, vector_weight=1.0, bm25_weight=1.0)
            hybrid_results[qid] = {mid: sc for mid, sc in fused[:100]}
        ndcg_dict, _md, _rcd, _r = evaluator.evaluate(qrels, hybrid_results, k_values=[10, 100])
        hybrid_ndcg = ndcg_dict["NDCG@10"]
        delta = hybrid_ndcg - base_ndcg

        print(f"{label:<55}{base_ndcg:>10.4f}{hybrid_ndcg:>10.4f}{delta:>+9.4f}{mean_gap:>24.4f}")

        del encoder
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\nIf the 'top1-top10 gap' hypothesis is right, the embedder with the")
    print("LARGEST gap (most 'peaked'/confident dense ranking) should show hybrid")
    print("HURTING the most (most negative delta), and the smallest gap should")
    print("show hybrid HELPING the most.")


if __name__ == "__main__":
    main()

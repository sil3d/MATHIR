#!/usr/bin/env python3
"""
Tests whether MATHIR's hybrid_search RRF fusion default weights
(vector_weight=1.0, bm25_weight=1.0) are the reason hybrid trails
pure-dense FAISS on semantically-rich corpora (scifact/nfcorpus per the
already-established real benchmark numbers), by sweeping vector_weight
while holding bm25_weight fixed, on the exact same embeddings used
throughout this investigation (MATHIR's real default embedder).

Uses mathir_search.rrf_fusion directly (MATHIR's actual fusion function,
not a reimplementation) against a real vector-search ranking (FAISS) and
a real BM25Okapi ranking on nfcorpus.
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

    print("Building FAISS dense ranking (vector_results equivalent)...")
    import faiss
    dim = corpus_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embs.astype(np.float32))
    scores, indices = index.search(query_embs.astype(np.float32), 100)

    print("Building BM25 ranking (bm25_results equivalent)...")
    tokenized_corpus = [_tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    evaluator = EvaluateRetrieval()

    def run_fusion(vector_weight: float, bm25_weight: float):
        fused_results = {}
        for i, qid in enumerate(q_ids):
            vector_results = [(corpus_ids[indices[i][j]], float(scores[i][j])) for j in range(100) if indices[i][j] < len(corpus_ids)]
            bm25_scores = bm25.get_scores(_tokenize(q_texts[i]))
            bm25_results = sorted(
                [(corpus_ids[k], float(bm25_scores[k])) for k in range(len(corpus_ids)) if bm25_scores[k] > 0],
                key=lambda x: x[1], reverse=True,
            )[:100]
            fused = rrf_fusion(vector_results, bm25_results, vector_weight=vector_weight, bm25_weight=bm25_weight)
            fused_results[qid] = {mid: sc for mid, sc in fused[:100]}
        ndcg, map_, recall_cap, _recall = evaluator.evaluate(qrels, fused_results, k_values=[10, 100])
        return ndcg["NDCG@10"], map_["MAP@10"], recall_cap["Recall@100"]

    print(f"\n{'='*60}\nDATASET: {DATASET} -- sweeping vector_weight (bm25_weight=1.0)\n{'='*60}")

    print("\n--- baselines ---")
    pure_vector_results = {q_ids[i]: {corpus_ids[indices[i][j]]: float(scores[i][j]) for j in range(100) if indices[i][j] < len(corpus_ids)} for i in range(len(q_ids))}
    ndcg_v, _map_v, _recall_cap_v, _r_v = evaluator.evaluate(qrels, pure_vector_results, k_values=[10, 100])
    print(f"  Pure FAISS dense-only:  nDCG@10={ndcg_v['NDCG@10']:.4f}")

    print("\n--- MATHIR's real rrf_fusion(), sweeping weights ---")
    for vw, bw in [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (5.0, 1.0), (10.0, 1.0), (1.0, 0.3), (1.0, 0.1)]:
        ndcg10, map10, rec100 = run_fusion(vw, bw)
        print(f"  vector_weight={vw:>5.1f} bm25_weight={bw:>4.1f}:  nDCG@10={ndcg10:.4f}  MAP@10={map10:.4f}  Recall@100={rec100:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Real A/B comparison of MATHIR's current default embedding model
(sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, trained for
paraphrase/STS similarity) against a same-footprint retrieval-focused
alternative (intfloat/multilingual-e5-small) -- both 384-dim, ~118M params,
same edge-device resource class. No API/LLM calls needed: this is pure
local embedding + FAISS-dense retrieval, scored with the real
beir.retrieval.evaluation.EvaluateRetrieval (same methodology already used
in multi_dataset_efficient.py for nDCG@10/MRR@10/Recall@100).

Rationale: the live MATHIR-vs-FAISS benchmark showed MATHIR trailing FAISS
on real BEIR data partly because MATHIR's default embedder was trained for
paraphrase similarity, not retrieval. E5 models are explicitly trained for
retrieval and require a "query: " / "passage: " prefix convention on inputs
-- applied here for the E5 side only, since that's how E5 is meant to be used.

Datasets: scifact, nfcorpus, arguana (already present locally, no download).

Usage:
    python compare_embedding_models.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASETS = ["scifact", "nfcorpus", "arguana"]
BEIR_DATA_DIR = Path(__file__).resolve().parent.parent / "05_test_data" / "beir_data"
OUT_FILE = Path(__file__).resolve().parent.parent / "06_results" / "current" / "embedding_model_comparison.json"

MODELS = {
    "current_default (paraphrase-multilingual-MiniLM-L12-v2)": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "candidate (multilingual-e5-small)": {
        "name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}


def load_dataset(name: str):
    data_path = BEIR_DATA_DIR / name
    nested = data_path / name
    if nested.exists():
        data_path = nested
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset {name} not found at {data_path}")
    return GenericDataLoader(data_path).load(split="test")


def dense_search(encoder, corpus_texts, corpus_ids, queries, k=100):
    import faiss

    corpus_embs = encoder.encode(corpus_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    q_ids = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]
    query_embs = encoder.encode(q_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    dim = corpus_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embs.astype(np.float32))
    scores, indices = index.search(query_embs.astype(np.float32), k)

    results = {}
    for i, qid in enumerate(q_ids):
        results[qid] = {}
        for j in range(k):
            if indices[i][j] < len(corpus_ids):
                results[qid][corpus_ids[indices[i][j]]] = float(scores[i][j])
    return results


def main():
    evaluator = EvaluateRetrieval()
    all_results = {}

    for dataset in DATASETS:
        print(f"\n{'='*70}\nDATASET: {dataset}\n{'='*70}")
        corpus, queries, qrels = load_dataset(dataset)
        corpus_ids = list(corpus.keys())
        print(f"  corpus={len(corpus_ids)} queries={len(queries)} qrels={len(qrels)}")

        dataset_results = {}
        for label, cfg in MODELS.items():
            print(f"\n  --- {label} ({cfg['name']}) ---")
            encoder = SentenceTransformer(cfg["name"], device=DEVICE)

            corpus_texts = [cfg["passage_prefix"] + (corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip() for cid in corpus_ids]
            queries_prefixed = {qid: cfg["query_prefix"] + qtext for qid, qtext in queries.items()}

            t0 = time.time()
            scores = dense_search(encoder, corpus_texts, corpus_ids, queries_prefixed, k=100)
            elapsed = time.time() - t0

            ndcg, _map, recall_cap, _recall = evaluator.evaluate(qrels, scores, k_values=[10, 100])
            ndcg10 = ndcg["NDCG@10"]
            map10 = _map["MAP@10"]
            rec100 = recall_cap["Recall@100"]
            print(f"    nDCG@10={ndcg10:.4f}  MAP@10={map10:.4f}  Recall@100={rec100:.4f}  time={elapsed:.1f}s")
            dataset_results[label] = {"nDCG@10": ndcg10, "MAP@10": map10, "Recall@100": rec100, "time_s": elapsed}

            del encoder
            torch.cuda.empty_cache() if DEVICE == "cuda" else None

        all_results[dataset] = dataset_results

    print(f"\n\n{'='*70}\nSUMMARY (nDCG@10)\n{'='*70}")
    print(f"{'dataset':<14}" + "".join(f"{label[:28]:>30}" for label in MODELS))
    for dataset, res in all_results.items():
        print(f"{dataset:<14}" + "".join(f"{res[label]['nDCG@10']:>30.4f}" for label in MODELS))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump({"models": {k: v["name"] for k, v in MODELS.items()}, "results": all_results}, f, indent=2)
    print(f"\nWrote {OUT_FILE}")
    return all_results


if __name__ == "__main__":
    main()

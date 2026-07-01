#!/usr/bin/env python3
"""
Isolates whether MATHIR's quality gap vs FAISS is an embedder-choice issue
or an actual retrieval-mechanism bug, by holding the embedder FIXED and
comparing raw FAISS IndexFlatIP against MATHIR's own VecMemory.search()
on the exact same embeddings, for the exact same corpus/queries/qrels.

Uses a throwaway temp VecMemory (isolated SQLite file) -- does NOT touch
the live MATHIR daemon or its already-running benchmark job.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer

MCP_ROOT = Path(__file__).resolve().parent.parent.parent / "mathir_mcp"
sys.path.insert(0, str(MCP_ROOT / "mathir_lib"))
from mathir_vec import VecMemory  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEIR_DATA_DIR = Path(__file__).resolve().parent.parent / "05_test_data" / "beir_data"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # MATHIR's actual default
DATASET = "nfcorpus"  # smaller corpus, faster full-fidelity test


def load_dataset(name):
    data_path = BEIR_DATA_DIR / name
    nested = data_path / name
    if nested.exists():
        data_path = nested
    return GenericDataLoader(data_path).load(split="test")


def main():
    print(f"Loading {DATASET}...")
    corpus, queries, qrels = load_dataset(DATASET)
    corpus_ids = list(corpus.keys())
    print(f"  corpus={len(corpus_ids)} queries={len(queries)}")

    print(f"Loading MATHIR's actual default embedder: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME, device=DEVICE)
    dim = encoder.get_sentence_embedding_dimension()
    print(f"  dim={dim}")

    corpus_texts = [(corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip() for cid in corpus_ids]
    print("Encoding corpus...")
    corpus_embs = encoder.encode(corpus_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    q_ids = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]
    print("Encoding queries...")
    query_embs = encoder.encode(q_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    evaluator = EvaluateRetrieval()

    # --- A: raw FAISS IndexFlatIP on these exact embeddings ---
    print("\n=== A: raw FAISS IndexFlatIP (same embedder, same embeddings) ===")
    import faiss
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embs.astype(np.float32))
    t0 = time.time()
    scores, indices = index.search(query_embs.astype(np.float32), 100)
    faiss_time = time.time() - t0
    faiss_results = {}
    for i, qid in enumerate(q_ids):
        faiss_results[qid] = {}
        for j in range(100):
            if indices[i][j] < len(corpus_ids):
                faiss_results[qid][corpus_ids[indices[i][j]]] = float(scores[i][j])
    ndcg, _map, recall_cap, _r = evaluator.evaluate(qrels, faiss_results, k_values=[10, 100])
    print(f"  nDCG@10={ndcg['NDCG@10']:.4f}  MAP@10={_map['MAP@10']:.4f}  Recall@100={recall_cap['Recall@100']:.4f}  time={faiss_time:.2f}s")

    # --- B: MATHIR's own VecMemory.search() on the exact same embeddings ---
    print("\n=== B: MATHIR VecMemory.search() (same embedder, same embeddings, real code path) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "isolate_test.db"
        vm = VecMemory(db_path, embedding_dim=dim)
        print(f"  Inserting {len(corpus_ids)} docs into isolated VecMemory...")
        for i, cid in enumerate(corpus_ids):
            vm.store(f"doc_{cid}", corpus_embs[i], {
                "content": corpus_texts[i][:2000], "agent": "test",
                "block_type": "episodic", "label": "", "priority": 5,
            })

        mathir_results = {}
        t0 = time.time()
        for i, qid in enumerate(q_ids):
            hits = vm.search(query_embedding=query_embs[i], k=100)
            mathir_results[qid] = {}
            for h in hits:
                orig_id = h["memory_id"].replace("doc_", "", 1)
                # VecMemory.search() already returns "score" as a similarity
                # (1.0 - distance internally), higher = better -- same
                # convention as FAISS's IndexFlatIP.
                mathir_results[qid][orig_id] = h["score"]
        mathir_time = time.time() - t0
        vm.close()

    ndcg2, map2, recall_cap2, _r2 = evaluator.evaluate(qrels, mathir_results, k_values=[10, 100])
    print(f"  nDCG@10={ndcg2['NDCG@10']:.4f}  MAP@10={map2['MAP@10']:.4f}  Recall@100={recall_cap2['Recall@100']:.4f}  time={mathir_time:.2f}s")

    print("\n=== VERDICT ===")
    gap = ndcg["NDCG@10"] - ndcg2["NDCG@10"]
    print(f"FAISS nDCG@10:  {ndcg['NDCG@10']:.4f}")
    print(f"MATHIR nDCG@10: {ndcg2['NDCG@10']:.4f}")
    print(f"Gap: {gap:.4f} ({'MATHIR retrieval bug confirmed -- same embeddings, worse ranking' if gap > 0.02 else 'no meaningful gap -- MATHIR retrieval mechanism is fine, the earlier gap was purely the embedder choice'})")


if __name__ == "__main__":
    main()

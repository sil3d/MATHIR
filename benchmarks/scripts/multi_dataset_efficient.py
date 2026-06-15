"""
Efficient multi-dataset BEIR benchmark with GPU-accelerated CE reranking.
Tests: FAISS-only, BM25-only, MATHIR dense-only, hybrid (dense+BM25 RRF), hybrid+CE(top-20).
Datasets: SciFact, NFCorpus, ArguAna
"""

import json
import time
import numpy as np
from pathlib import Path
import torch
from rank_bm25 import BM25Okapi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import CrossEncoder, SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DATASETS = ["scifact", "nfcorpus", "arguana"]
MODEL_NAME = "BAAI/bge-base-en-v1.5"
BEIR_DATA_DIR = Path(__file__).parent.parent / "beir_data"
CACHE_DIR = Path(__file__).parent.parent / "controlled_emb_cache"
RESULTS_FILE = Path(__file__).parent.parent / "results_final" / "beir" / "multi_dataset_efficient_results.json"
TOP_K = 20

def load_dataset(name):
    data_path = BEIR_DATA_DIR / name
    nested = data_path / name
    if nested.exists():
        data_path = nested
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset {name} not found at {data_path}. Please download manually.")
    return GenericDataLoader(data_path).load(split="test")

class DenseFAISS:
    def __init__(self, model_name, device):
        self.encoder = SentenceTransformer(model_name, device=device)
        self.device = device

    def search(self, corpus_embs, doc_ids, query_embs, top_k=100):
        import faiss
        dim = corpus_embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        if self.device == "cuda":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(corpus_embs.astype(np.float32))
        scores, indices = index.search(query_embs.astype(np.float32), top_k)
        results = {}
        for i, qid in enumerate(doc_ids):
            results[qid] = {}
            for j in range(top_k):
                if indices[i][j] < len(doc_ids):
                    results[qid][doc_ids[indices[i][j]]] = float(scores[i][j])
        return results

class BM25:
    def __init__(self, corpus, device="cpu"):
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.tokenized_corpus = [corpus[did].get("text", "").lower().split() for did in self.doc_ids]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, queries, top_k=100):
        results = {}
        for qid, q_text in queries.items():
            scores = self.bm25.get_scores(q_text.lower().split())
            doc_scores = [(self.doc_ids[i], scores[i]) for i in np.argsort(scores)[::-1][:top_k]]
            results[qid] = {d: float(s) for d, s in doc_scores}
        return results

class CrossEncoderReranker:
    def __init__(self, model_name, device, top_k=20):
        self.reranker = CrossEncoder(model_name, max_length=512, device=device)
        self.top_k = top_k

    def rerank(self, queries, corpus, hybrid_scores):
        results = {}
        for qid, q_text in queries.items():
            top_docs = sorted(hybrid_scores.get(qid, {}).items(), key=lambda x: x[1], reverse=True)[:100]
            pairs = [(q_text, corpus.get(doc_id, {}).get("text", "")) for doc_id, _ in top_docs[:self.top_k]]
            if not pairs:
                results[qid] = {}
                continue
            ce_scores = self.reranker.predict(pairs, show_progress_bar=False)
            ce_ranked = [(doc_id, ce_scores[i]) for i, (doc_id, _) in enumerate(top_docs[:self.top_k])]
            ce_ranked += [(doc_id, score) for doc_id, score in top_docs[self.top_k:]]
            ce_ranked.sort(key=lambda x: x[1], reverse=True)
            results[qid] = {doc_id: float(score) for doc_id, score in ce_ranked[:100]}
        return results

def main():
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE) as f:
                results = json.load(f)
            # Ensure metadata datasets contains all datasets we want
            for ds in DATASETS:
                if ds not in results["metadata"]["datasets"]:
                    results["metadata"]["datasets"].append(ds)
        except Exception as e:
            print(f"Error loading existing results: {e}. Starting fresh.")
            results = {"metadata": {"model": MODEL_NAME, "datasets": DATASETS, "device": DEVICE}, "results": {}}
    else:
        results = {"metadata": {"model": MODEL_NAME, "datasets": DATASETS, "device": DEVICE}, "results": {}}
    
    evaluator = EvaluateRetrieval()

    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset}")
        print('='*60)

        corpus, queries, qrels = load_dataset(dataset)
        print(f"  Corpus: {len(corpus)}, Queries: {len(queries)}, Qrels: {len(qrels)}")

        safe_name = MODEL_NAME.replace("/", "_")
        corpus_emb_file = CACHE_DIR / f"{dataset}_{safe_name}_test_corpus_emb.npy"
        doc_ids_file = CACHE_DIR / f"{dataset}_{safe_name}_test_doc_ids.json"

        encoder = SentenceTransformer(MODEL_NAME, device=DEVICE)

        if corpus_emb_file.exists() and doc_ids_file.exists():
            print(f"  Loading cached corpus embeddings...")
            corpus_embs = np.load(corpus_emb_file)
            with open(doc_ids_file) as f:
                cached_doc_ids = json.load(f)
            doc_id_to_idx = {d: i for i, d in enumerate(cached_doc_ids)}
        else:
            print(f"  Encoding corpus (no cache)...")
            cached_doc_ids = list(corpus.keys())
            doc_texts = [corpus[did].get("text", "") for did in cached_doc_ids]
            corpus_embs = encoder.encode(doc_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
            CACHE_DIR.mkdir(exist_ok=True)
            np.save(corpus_emb_file, corpus_embs)
            with open(doc_ids_file, "w") as f:
                json.dump(cached_doc_ids, f)
            doc_id_to_idx = {d: i for i, d in enumerate(cached_doc_ids)}

        print(f"  Encoding queries...")
        q_ids = list(queries.keys())
        q_texts = [queries[qid] for qid in q_ids]
        query_embs = encoder.encode(q_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

        print(f"  FAISS GPU search...")
        import faiss
        dim = corpus_embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_embs.astype(np.float32))
        start = time.time()
        scores_np, indices_np = index.search(query_embs.astype(np.float32), 100)
        dense_scores = {}
        for i, qid in enumerate(q_ids):
            dense_scores[qid] = {}
            for j in range(100):
                if indices_np[i][j] < len(cached_doc_ids):
                    dense_scores[qid][cached_doc_ids[indices_np[i][j]]] = float(scores_np[i][j])
        dense_time = time.time() - start
        print(f"    Dense time: {dense_time:.2f}s")

        print(f"  BM25 search...")
        start = time.time()
        bm25 = BM25(corpus)
        bm25_scores = bm25.search(queries)
        bm25_time = time.time() - start
        print(f"    BM25 time: {bm25_time:.2f}s")

        reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", DEVICE, top_k=TOP_K)

        dataset_results = {}

        def evaluate(name, scores, time_s):
            ndcg, mrr, recall_cap, recall = evaluator.evaluate(qrels, scores, k_values=[10, 100])
            ndcg10 = ndcg["NDCG@10"]
            mrr10 = mrr["MAP@10"]
            rec100 = recall_cap["Recall@100"]
            print(f"  [{name}] nDCG@10: {ndcg10:.4f}, MRR@10: {mrr10:.4f}, Recall@100: {rec100:.4f}, Time: {time_s:.2f}s")
            dataset_results[name] = {
                "nDCG@10": ndcg10, "MRR@10": mrr10, "Recall@100": rec100, "time_s": time_s
            }

        evaluate("1_FAISS_only", dense_scores, dense_time)
        evaluate("2_BM25_only", bm25_scores, bm25_time)

        print(f"  Hybrid RRF (k=60)...")
        start = time.time()
        hybrid_scores = {}
        all_docs = set(corpus.keys())
        for qid in queries:
            d_ranked = sorted(dense_scores.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
            b_ranked = sorted(bm25_scores.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
            d_ranks = {d: r+1 for r, (d, _) in enumerate(d_ranked)}
            b_ranks = {d: r+1 for r, (d, _) in enumerate(b_ranked)}
            rrf = {}
            for doc_id in all_docs:
                s = (1/(60+d_ranks[doc_id]) if doc_id in d_ranks else 0) + \
                    (1/(60+b_ranks[doc_id]) if doc_id in b_ranks else 0)
                rrf[doc_id] = s
            hybrid_scores[qid] = dict(sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:100])
        hybrid_total_time = time.time() - start + dense_time + bm25_time
        evaluate("3_hybrid_RRF", hybrid_scores, hybrid_total_time)

        print(f"  Hybrid + CE rerank (top-{TOP_K})...")
        start = time.time()
        ce_scores = reranker.rerank(queries, corpus, hybrid_scores)
        ce_total_time = time.time() - start + hybrid_total_time
        evaluate("4_hybrid_CE", ce_scores, ce_total_time)

        results["results"][dataset] = dataset_results

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {RESULTS_FILE}")

    print(f"\n\n{'='*60}")
    print("FINAL RESULTS")
    print('='*60)
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    main()

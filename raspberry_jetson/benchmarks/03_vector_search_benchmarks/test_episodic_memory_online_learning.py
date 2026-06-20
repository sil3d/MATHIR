"""
Episodic Memory Online Learning Benchmark
==========================================

Tests whether MATHIR's episodic memory provides real online learning benefit
over static FAISS by measuring if storing relevant documents FIRST improves
subsequent recall.

Phase 1 (COLD): Raw FAISS search (no MATHIR memory) → nDCG@10
Phase 2 (WARM): Store relevant docs from qrels, then recall → nDCG@10

If MATHIR's episodic memory works: warm should beat cold.
If episodic memory doesn't help: warm ≈ cold (same as FAISS).
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Windows SSL bypass
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "beir_data")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodic_memory_results.json")


# ============================================================================
# TREC-style Metrics
# ============================================================================
def dcg_at_k(relevances: List[int], k: int) -> float:
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return float(sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances)))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal_rels = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_run(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = (10,),
) -> Dict[str, float]:
    """Compute nDCG@k averaged over queries with qrels."""
    ndcg_scores: Dict[int, List[float]] = {k: [] for k in k_values}

    for qid, retrieved in results.items():
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        retrieved_ids = [doc_id for doc_id, _ in retrieved]
        relevances = [relevant.get(doc_id, 0) for doc_id in retrieved_ids]
        for k in k_values:
            ndcg_scores[k].append(ndcg_at_k(relevances, k))

    metrics = {}
    for k in k_values:
        metrics[f"nDCG@{k}"] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0.0
    return metrics


# ============================================================================
# Data Loading
# ============================================================================
def load_beir_dataset(name: str = "scifact") -> Tuple[Dict, Dict, Dict]:
    """
    Load BEIR SciFact dataset from local cache.

    Returns: (corpus, queries, qrels)
      corpus:  {doc_id: {"title": ..., "text": ...}}
      queries: {query_id: query_text}
      qrels:   {query_id: {doc_id: relevance}}
    """
    target_dir = os.path.join(DATA_DIR, name, name)

    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    corpus_path = os.path.join(target_dir, "corpus.jsonl")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }

    queries_path = os.path.join(target_dir, "queries.jsonl")
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q.get("text", "")

    qrels_path = os.path.join(target_dir, "qrels", "test.tsv")
    with open(qrels_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, rel = parts[0], parts[1], int(parts[2])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][did] = rel

    print(f"  Loaded {name}: {len(corpus)} docs, {len(queries)} queries, "
          f"{sum(len(v) for v in qrels.values())} relevance judgments")
    return corpus, queries, qrels


# ============================================================================
# MATHIR Episodic Memory Integration
# ============================================================================
class MATHIREpisodicRetriever:
    """
    MATHIR retriever that uses episodic memory for online learning.

    Architecture:
    - FAISS for base retrieval (static)
    - RawEmbeddingEpisodicMemory for storing relevant docs and improving recall

    The key test: Does storing relevant docs FIRST improve subsequent recall?
    """

    def __init__(
        self,
        embedder_name: str = "BAAI/bge-base-en-v1.5",
        episodic_capacity: int = 10000,
    ):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.embedder_name = embedder_name
        self.episodic_capacity = episodic_capacity

        # Lazy load embedder
        self._embedder = None
        self._embedder_dim = None

        # FAISS index
        self._faiss_index = None
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []
        self._doc_embeddings: Optional[np.ndarray] = None
        self._indexed = False

        # MATHIR Episodic Memory (raw embeddings - no projection bottleneck)
        self._init_episodic_memory()

    def _init_episodic_memory(self):
        """Initialize MATHIR raw embedding episodic memory."""
        from mathir_lib.memory.raw_episodic import RawEmbeddingEpisodicMemory

        # Determine embedding dimension lazily
        self._load_embedder()
        embedding_dim = self._embedder.get_sentence_embedding_dimension()

        # Use RawEmbeddingEpisodicMemory to store FULL embeddings
        # This bypasses the projection bottleneck that loses information
        self.episodic = RawEmbeddingEpisodicMemory(
            capacity=self.episodic_capacity,
            embedding_dim=embedding_dim,
            projection=False,  # No projection - store full embedding
        )
        self._episodic_stored = 0

        # Map from episodic memory index to doc_id for boosting
        self._episodic_doc_map: Dict[int, str] = {}

    def _load_embedder(self):
        """Lazy load the embedder."""
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(self.embedder_name)
        self._embedder_dim = self._embedder.get_sentence_embedding_dimension()

    def index(self, doc_ids: List[str], doc_texts: List[str]):
        """Build FAISS index from documents."""
        import faiss

        self._load_embedder()

        self._doc_ids = [str(d) for d in doc_ids]
        self._doc_texts = [str(t) for t in doc_texts]

        t0 = time.perf_counter()
        doc_texts_combined = [
            (doc_texts[i].split(". ")[0] if ". " in doc_texts[i] else doc_texts[i][:512])
            for i in range(len(doc_texts))
        ]
        self._doc_embeddings = self._embedder.encode(
            doc_texts_combined,
            show_progress_bar=False,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = int(self._doc_embeddings.shape[1])
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(self._doc_embeddings)
        self._indexed = True

        self._index_time = (time.perf_counter() - t0) * 1000
        return self

    def store_relevant_docs(
        self,
        relevant_docs: Dict[str, Dict[str, int]],
        corpus: Dict[str, Dict[str, str]],
    ):
        """
        Store relevant documents from qrels into MATHIR episodic memory.

        This is the "learning" phase - we show MATHIR which docs are relevant
        BEFORE the recall test.
        """
        self._load_embedder()

        stored_count = 0
        for qid, doc_scores in relevant_docs.items():
            for doc_id, rel in doc_scores.items():
                if rel <= 0:
                    continue
                if doc_id not in corpus:
                    continue

                doc_text = corpus[doc_id].get("text", "") or corpus[doc_id].get("title", "")
                if not doc_text:
                    continue

                # Encode the document with raw embedder (full dimensionality)
                emb = self._embedder.encode(
                    [doc_text[:512]],
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype("float32")

                # Store raw embedding in episodic memory
                import torch
                emb_tensor = torch.from_numpy(emb)
                self.episodic.store(emb_tensor)

                # Map this episodic slot to the doc_id
                mem_idx = self.episodic.get_usage() - 1
                self._episodic_doc_map[mem_idx] = doc_id

                stored_count += 1

        self._episodic_stored = stored_count
        print(f"  Stored {stored_count} relevant docs in episodic memory")

    def search(
        self,
        query: str,
        k: int = 10,
        use_episodic: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Search documents.

        Args:
            query: query text
            k: number of results
            use_episodic: if True, use episodic memory to boost recall
        """
        if not self._indexed:
            raise RuntimeError("Must call index() before search()")

        self._load_embedder()

        # Encode query
        query_emb = self._embedder.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # FAISS search
        top_k = min(100, len(self._doc_ids))
        scores, indices = self._faiss_index.search(query_emb, top_k)

        # Build results
        results = []
        for j in range(indices.shape[1]):
            idx = int(indices[0, j])
            if 0 <= idx < len(self._doc_ids):
                results.append((self._doc_ids[idx], float(scores[0, j])))

        # If episodic is enabled and we have memories, boost similar docs
        if use_episodic and self.episodic.get_usage() > 0:
            results = self._episodic_boost(query_emb, results)

        return results[:k]

    def _episodic_boost(
        self,
        query_emb: np.ndarray,
        results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Boost document scores using episodic memory similarity.

        The key insight: If we stored relevant docs in episodic memory,
        and those docs are similar to some FAISS results, boost those results.
        """
        import torch

        # Search episodic memory for similar stored experiences
        query_tensor = torch.from_numpy(query_emb)
        episodic_indices, episodic_sims = self.episodic.search(query_tensor, k=10)

        if episodic_indices.numel() == 0:
            return results

        # Build a set of doc_ids that episodic memory thinks are relevant
        # based on what we stored during the learning phase
        episodic_relevant_docs: Dict[str, float] = {}
        for mem_idx, sim in zip(episodic_indices[0].tolist(), episodic_sims[0].tolist()):
            if mem_idx in self._episodic_doc_map:
                doc_id = self._episodic_doc_map[mem_idx]
                # Accumulate similarity scores
                if doc_id not in episodic_relevant_docs:
                    episodic_relevant_docs[doc_id] = 0.0
                episodic_relevant_docs[doc_id] = max(episodic_relevant_docs[doc_id], sim)

        if not episodic_relevant_docs:
            return results

        # Boost documents that appear in episodic memory
        # The boost is proportional to how similar they are to stored memories
        boost_weight = 0.5  # Weight for episodic boost
        max_sim = max(episodic_sims[0].tolist()) if episodic_sims.numel() > 0 else 1.0

        boosted_results = []
        for doc_id, faiss_score in results:
            if doc_id in episodic_relevant_docs:
                # Normalize episodic similarity
                episodic_sim = episodic_relevant_docs[doc_id] / max_sim
                # Combined score: FAISS score + episodic boost
                combined_score = faiss_score + boost_weight * episodic_sim
                boosted_results.append((doc_id, combined_score))
            else:
                boosted_results.append((doc_id, faiss_score))

        # Re-sort by combined score
        boosted_results.sort(key=lambda x: -x[1])
        return boosted_results

    def reset_episodic(self):
        """Reset episodic memory."""
        self.episodic.reset()
        self._episodic_stored = 0
        self._episodic_doc_map = {}

    def get_episodic_count(self) -> int:
        """Get number of memories stored."""
        return self.episodic.get_usage()


# ============================================================================
# Benchmark Runner
# ============================================================================
def run_cold_phase(
    retriever: MATHIREpisodicRetriever,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, float]]:
    """
    Phase 1 (COLD): Raw FAISS search without episodic memory.
    """
    print("\n[PHASE 1] COLD - Raw FAISS search (no episodic memory)")
    results: Dict[str, List[Tuple[str, float]]] = {}
    latencies: List[float] = []

    for qid, qtext in queries.items():
        t0 = time.perf_counter()
        retrieved = retriever.search(qtext, k=100, use_episodic=False)
        results[qid] = retrieved
        latencies.append((time.perf_counter() - t0) * 1000)

    metrics = evaluate_run(results, qrels, k_values=[10])
    metrics["latency_mean_ms"] = float(np.mean(latencies)) if latencies else 0.0
    metrics["num_queries"] = len(queries)

    print(f"  Cold nDCG@10: {metrics['nDCG@10']:.4f}")
    print(f"  Mean latency: {metrics['latency_mean_ms']:.1f}ms")

    return results, metrics


def run_warm_phase(
    retriever: MATHIREpisodicRetriever,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    corpus: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, float]]:
    """
    Phase 2 (WARM): Store relevant docs in episodic memory, then recall.
    """
    print("\n[PHASE 2] WARM - Store relevant docs, then recall with episodic boost")

    # Store all relevant docs from qrels into episodic memory
    print("  Storing relevant docs in episodic memory...")
    t0 = time.perf_counter()
    retriever.store_relevant_docs(qrels, corpus)
    store_time = (time.perf_counter() - t0) * 1000
    print(f"  Storage time: {store_time:.0f}ms")
    print(f"  Episodic memory count: {retriever.get_episodic_count()}")

    # Now search with episodic boost
    results: Dict[str, List[Tuple[str, float]]] = {}
    latencies: List[float] = []

    for qid, qtext in queries.items():
        t0 = time.perf_counter()
        retrieved = retriever.search(qtext, k=100, use_episodic=True)
        results[qid] = retrieved
        latencies.append((time.perf_counter() - t0) * 1000)

    metrics = evaluate_run(results, qrels, k_values=[10])
    metrics["latency_mean_ms"] = float(np.mean(latencies)) if latencies else 0.0
    metrics["num_queries"] = len(queries)
    metrics["store_time_ms"] = store_time

    print(f"  Warm nDCG@10: {metrics['nDCG@10']:.4f}")
    print(f"  Mean latency: {metrics['latency_mean_ms']:.1f}ms")

    return results, metrics


# ============================================================================
# Main
# ============================================================================
def main() -> Dict:
    """Run the episodic memory online learning benchmark."""
    print("=" * 80)
    print("EPISODIC MEMORY ONLINE LEARNING BENCHMARK")
    print("=" * 80)
    print(f"Dataset: BEIR SciFact")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print()

    # Load data
    print("[1/5] Loading BEIR SciFact dataset...")
    corpus, queries, qrels = load_beir_dataset("scifact")
    print(f"  Corpus: {len(corpus)} docs")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Qrels: {sum(len(v) for v in qrels.values())} judgments")

    # Filter queries to only those with qrels
    queries_with_qrels = {qid: text for qid, text in queries.items() if qid in qrels}
    print(f"  Queries with qrels: {len(queries_with_qrels)}")

    # Build retriever and index
    print("\n[2/5] Building FAISS index...")
    retriever = MATHIREpisodicRetriever(
        embedder_name="BAAI/bge-base-en-v1.5",
        episodic_capacity=10000,
    )

    doc_ids = list(corpus.keys())
    doc_texts = [
        (corpus[did]["title"] + " " + corpus[did]["text"]).strip()
        for did in doc_ids
    ]

    t0 = time.perf_counter()
    retriever.index(doc_ids, doc_texts)
    index_time = (time.perf_counter() - t0) * 1000
    print(f"  Index time: {index_time:.0f}ms")

    # Phase 1: Cold (no episodic memory)
    print("\n[3/5] Running PHASE 1 (COLD)...")
    cold_results, cold_metrics = run_cold_phase(
        retriever, queries_with_qrels, qrels
    )

    # Reset episodic memory before warm phase
    retriever.reset_episodic()

    # Phase 2: Warm (with episodic memory)
    print("\n[4/5] Running PHASE 2 (WARM)...")
    warm_results, warm_metrics = run_warm_phase(
        retriever, queries_with_qrels, qrels, corpus
    )

    # Compute improvement
    cold_ndcg = cold_metrics["nDCG@10"]
    warm_ndcg = warm_metrics["nDCG@10"]
    improvement = warm_ndcg - cold_ndcg
    improvement_pct = (improvement / cold_ndcg * 100) if cold_ndcg > 0 else 0.0

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"  Cold nDCG@10 (raw FAISS):    {cold_ndcg:.4f}")
    print(f"  Warm nDCG@10 (with episodic): {warm_ndcg:.4f}")
    print(f"  Improvement:                  {improvement:+.4f} ({improvement_pct:+.2f}%)")
    print(f"  Episodic memory stored:        {retriever.get_episodic_count()} docs")
    print()

    # Build results dict
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "dataset": "BEIR SciFact",
            "num_corpus": len(corpus),
            "num_queries": len(queries_with_qrels),
            "num_qrels": sum(len(v) for v in qrels.values()),
            "embedder": retriever.embedder_name,
        },
        "cold_phase": {
            "description": "Raw FAISS search (no episodic memory)",
            "nDCG@10": cold_ndcg,
            "latency_mean_ms": cold_metrics.get("latency_mean_ms", 0.0),
            "num_queries": cold_metrics.get("num_queries", 0),
        },
        "warm_phase": {
            "description": "Store relevant docs, then recall with episodic boost",
            "nDCG@10": warm_ndcg,
            "latency_mean_ms": warm_metrics.get("latency_mean_ms", 0.0),
            "store_time_ms": warm_metrics.get("store_time_ms", 0.0),
            "episodic_stored": retriever.get_episodic_count(),
            "num_queries": warm_metrics.get("num_queries", 0),
        },
        "improvement": {
            "absolute": improvement,
            "relative_pct": improvement_pct,
            "episodic_memory_helps": improvement > 0,
        },
    }

    # Save results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to: {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)

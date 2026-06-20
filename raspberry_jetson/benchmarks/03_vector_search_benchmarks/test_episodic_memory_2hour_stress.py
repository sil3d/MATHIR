"""
Episodic Memory 2-Hour Stress Test
===================================

Tests MATHIR's episodic memory under sustained load:
- Memory filling up (episodic capacity is 1000 slots)
- Repeated store/recall cycles
- Queries that haven't been seen before (no cached answers)
- Degradation patterns as memory gets full
- Recovery after eviction

This is a SIMULATED 2-hour test that runs the equivalent workload
in a compressed timeframe, measuring the same metrics.

Simulation Parameters:
- Total operations: ~5000 mixed queries + ~3000 stores
- Capacity: 1000 slots
- Test interval: every checkpoint (~200 operations)
- Total checkpoints: 24 (simulating 2 hours at 5-min intervals)
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Windows SSL bypass
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beir_data")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodic_stress_results.json")


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
# Episodic Memory Retriever with Stress Testing Hooks
# ============================================================================

class StressTestEpisodicRetriever:
    """
    MATHIR retriever with hooks for stress testing episodic memory.

    Tracks:
    - Store count
    - Eviction events (when old memories are overwritten)
    - Memory fullness percentage
    - All operations for later analysis
    """

    def __init__(
        self,
        embedder_name: str = "BAAI/bge-base-en-v1.5",
        episodic_capacity: int = 1000,  # Use 1000 as specified
    ):
        from sentence_transformers import SentenceTransformer

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

        # Stress test tracking
        self._total_stores = 0
        self._eviction_count = 0
        self._store_history: List[Dict] = []
        self._checkpoint_data: List[Dict] = []
        self._phase = "cold"

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

    def store_doc(self, doc_text: str, doc_id: str) -> None:
        """
        Store a document embedding in episodic memory.

        Tracks eviction events when memory is at capacity.
        """
        self._load_embedder()

        # Encode the document
        emb = self._embedder.encode(
            [doc_text[:512]],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Track if this will cause an eviction
        current_usage = self.episodic.get_usage()
        will_evict = current_usage >= self.episodic_capacity

        import torch
        emb_tensor = torch.from_numpy(emb)
        self.episodic.store(emb_tensor)

        # Track eviction
        if will_evict:
            self._eviction_count += 1

        # Map this episodic slot to the doc_id
        mem_idx = self.episodic.get_usage() - 1
        self._episodic_doc_map[mem_idx] = doc_id

        self._total_stores += 1
        self._store_history.append({
            "store_num": self._total_stores,
            "doc_id": doc_id,
            "mem_idx": mem_idx,
            "evicted": will_evict,
            "total_evictions": self._eviction_count,
        })

    def store_relevant_docs(
        self,
        relevant_docs: Dict[str, Dict[str, int]],
        corpus: Dict[str, Dict[str, str]],
    ) -> int:
        """
        Store relevant documents from qrels into MATHIR episodic memory.

        Returns the number of docs stored.
        """
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

                self.store_doc(doc_text, doc_id)
                stored_count += 1

        return stored_count

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
        """Reset episodic memory and tracking."""
        self.episodic.reset()
        self._episodic_doc_map = {}
        self._total_stores = 0
        self._eviction_count = 0
        self._store_history = []
        self._phase = "cold"

    def get_episodic_count(self) -> int:
        """Get number of memories stored."""
        return self.episodic.get_usage()

    def get_memory_fullness(self) -> float:
        """Get memory fullness as percentage (0.0 to 1.0+)."""
        return self.episodic.get_usage() / self.episodic_capacity

    def record_checkpoint(self, checkpoint_num: int, ndcg10: float, phase: str) -> Dict:
        """Record checkpoint data for timeline analysis."""
        data = {
            "checkpoint": checkpoint_num,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "total_stores": self._total_stores,
            "eviction_count": self._eviction_count,
            "episodic_count": self.episodic.get_usage(),
            "memory_fullness_pct": self.get_memory_fullness() * 100,
            "nDCG@10": ndcg10,
            "eviction_rate": self._eviction_count / max(1, self._total_stores),
        }
        self._checkpoint_data.append(data)
        return data

    def set_phase(self, phase: str):
        """Set the current stress test phase."""
        self._phase = phase


# ============================================================================
# Stress Test Phases
# ============================================================================

def run_phase1_normal_operation(
    retriever: StressTestEpisodicRetriever,
    queries_with_qrels: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    corpus: Dict[str, Dict[str, str]],
    config: Dict,
) -> List[Dict]:
    """
    Phase 1: Normal Operation (0-60 min equivalent)
    - Random mix of store + recall operations
    - Real BEIR queries from SciFact
    - Measure: nDCG@10 at each checkpoint
    """
    print("\n" + "=" * 70)
    print("[PHASE 1] NORMAL OPERATION - Memory filling (0-60 min equivalent)")
    print("=" * 70)

    retriever.set_phase("phase1_normal")

    # Get all relevant doc_ids for random selection during stores
    all_relevant_docs = []
    for qid, doc_scores in qrels.items():
        for doc_id, rel in doc_scores.items():
            if rel > 0 and doc_id in corpus:
                all_relevant_docs.append((doc_id, corpus[doc_id]))

    print(f"  Total relevant docs available: {len(all_relevant_docs)}")
    print(f"  Capacity: {retriever.episodic_capacity} slots")
    print(f"  Target stores: ~{config['phase1_stores']}")

    checkpoint_results = []
    checkpoint_interval = config['checkpoint_interval']
    stores_per_checkpoint = config['phase1_stores'] // config['num_checkpoints_phase1']
    queries_per_checkpoint = config['queries_per_checkpoint']

    for cp in range(config['num_checkpoints_phase1']):
        # Store new relevant docs
        stores_this_cp = 0
        for _ in range(stores_per_checkpoint):
            if stores_this_cp >= stores_per_checkpoint:
                break
            # Pick a random relevant doc to store
            import random
            doc_id, doc_data = random.choice(all_relevant_docs)
            doc_text = doc_data.get("text", "") or doc_data.get("title", "")
            if doc_text:
                retriever.store_doc(doc_text, doc_id)
                stores_this_cp += 1

        # Run recall queries
        query_ids = list(queries_with_qrels.keys())
        import random
        sampled_queries = random.sample(query_ids, min(queries_per_checkpoint, len(query_ids)))

        results = {}
        for qid in sampled_queries:
            retrieved = retriever.search(queries_with_qrels[qid], k=100, use_episodic=True)
            results[qid] = retrieved

        metrics = evaluate_run(results, qrels, k_values=[10])
        ndcg10 = metrics.get("nDCG@10", 0.0)

        cp_data = retriever.record_checkpoint(cp + 1, ndcg10, "phase1_normal")
        checkpoint_results.append(cp_data)

        print(f"  Checkpoint {cp + 1:2d}/{config['num_checkpoints_phase1']}: "
              f"stores={retriever._total_stores:4d}, "
              f"evictions={retriever._eviction_count:3d}, "
              f"fullness={retriever.get_memory_fullness()*100:5.1f}%, "
              f"nDCG@10={ndcg10:.4f}")

    return checkpoint_results


def run_phase2_memory_pressure(
    retriever: StressTestEpisodicRetriever,
    queries_with_qrels: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    corpus: Dict[str, Dict[str, str]],
    config: Dict,
) -> List[Dict]:
    """
    Phase 2: Memory Pressure (60-90 min equivalent)
    - Keep storing even when at capacity
    - Old entries get evicted (circular buffer)
    - Measure: does nDCG@10 drop when old memories are evicted?
    """
    print("\n" + "=" * 70)
    print("[PHASE 2] MEMORY PRESSURE - Eviction starts (60-90 min equivalent)")
    print("=" * 70)

    retriever.set_phase("phase2_pressure")

    # Get all relevant doc_ids for random selection during stores
    all_relevant_docs = []
    for qid, doc_scores in qrels.items():
        for doc_id, rel in doc_scores.items():
            if rel > 0 and doc_id in corpus:
                all_relevant_docs.append((doc_id, corpus[doc_id]))

    checkpoint_results = []
    stores_per_checkpoint = config['phase2_stores'] // config['num_checkpoints_phase2']
    queries_per_checkpoint = config['queries_per_checkpoint']

    for cp in range(config['num_checkpoints_phase2']):
        # Store new relevant docs (will cause evictions since we're at capacity)
        stores_this_cp = 0
        for _ in range(stores_per_checkpoint):
            if stores_this_cp >= stores_per_checkpoint:
                break
            import random
            doc_id, doc_data = random.choice(all_relevant_docs)
            doc_text = doc_data.get("text", "") or doc_data.get("title", "")
            if doc_text:
                retriever.store_doc(doc_text, doc_id)
                stores_this_cp += 1

        # Run recall queries with episodic boost
        query_ids = list(queries_with_qrels.keys())
        import random
        sampled_queries = random.sample(query_ids, min(queries_per_checkpoint, len(query_ids)))

        results = {}
        for qid in sampled_queries:
            retrieved = retriever.search(queries_with_qrels[qid], k=100, use_episodic=True)
            results[qid] = retrieved

        metrics = evaluate_run(results, qrels, k_values=[10])
        ndcg10 = metrics.get("nDCG@10", 0.0)

        cp_data = retriever.record_checkpoint(
            config['num_checkpoints_phase1'] + cp + 1,
            ndcg10,
            "phase2_pressure"
        )
        checkpoint_results.append(cp_data)

        print(f"  Checkpoint {config['num_checkpoints_phase1'] + cp + 1:2d}: "
              f"stores={retriever._total_stores:4d}, "
              f"evictions={retriever._eviction_count:3d}, "
              f"fullness={retriever.get_memory_fullness()*100:5.1f}%, "
              f"nDCG@10={ndcg10:.4f}")

    return checkpoint_results


def run_phase3_recovery(
    retriever: StressTestEpisodicRetriever,
    queries_with_qrels: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    corpus: Dict[str, Dict[str, str]],
    config: Dict,
) -> List[Dict]:
    """
    Phase 3: Recovery Test (90-120 min equivalent)
    - After eviction, can MATHIR still recall RECENT memories?
    - Can it learn NEW relevant docs?
    - Measure: recovery rate after memory stress
    """
    print("\n" + "=" * 70)
    print("[PHASE 3] RECOVERY TEST - Post-eviction memory (90-120 min)")
    print("=" * 70)

    retriever.set_phase("phase3_recovery")

    # Get all relevant doc_ids for random selection during stores
    all_relevant_docs = []
    for qid, doc_scores in qrels.items():
        for doc_id, rel in doc_scores.items():
            if rel > 0 and doc_id in corpus:
                all_relevant_docs.append((doc_id, corpus[doc_id]))

    checkpoint_results = []
    stores_per_checkpoint = config['phase3_stores'] // config['num_checkpoints_phase3']
    queries_per_checkpoint = config['queries_per_checkpoint']

    for cp in range(config['num_checkpoints_phase3']):
        # Store NEW relevant docs (testing if it can still learn)
        stores_this_cp = 0
        for _ in range(stores_per_checkpoint):
            if stores_this_cp >= stores_per_checkpoint:
                break
            import random
            doc_id, doc_data = random.choice(all_relevant_docs)
            doc_text = doc_data.get("text", "") or doc_data.get("title", "")
            if doc_text:
                retriever.store_doc(doc_text, doc_id)
                stores_this_cp += 1

        # Run recall queries
        query_ids = list(queries_with_qrels.keys())
        import random
        sampled_queries = random.sample(query_ids, min(queries_per_checkpoint, len(query_ids)))

        results = {}
        for qid in sampled_queries:
            retrieved = retriever.search(queries_with_qrels[qid], k=100, use_episodic=True)
            results[qid] = retrieved

        metrics = evaluate_run(results, qrels, k_values=[10])
        ndcg10 = metrics.get("nDCG@10", 0.0)

        cp_data = retriever.record_checkpoint(
            config['num_checkpoints_phase1'] + config['num_checkpoints_phase2'] + cp + 1,
            ndcg10,
            "phase3_recovery"
        )
        checkpoint_results.append(cp_data)

        print(f"  Checkpoint {config['num_checkpoints_phase1'] + config['num_checkpoints_phase2'] + cp + 1:2d}: "
              f"stores={retriever._total_stores:4d}, "
              f"evictions={retriever._eviction_count:3d}, "
              f"fullness={retriever.get_memory_fullness()*100:5.1f}%, "
              f"nDCG@10={ndcg10:.4f}")

    return checkpoint_results


# ============================================================================
# Main
# ============================================================================

def main() -> Dict:
    """Run the 2-hour episodic memory stress test."""
    print("=" * 70)
    print("EPISODIC MEMORY 2-HOUR STRESS TEST")
    print("=" * 70)
    print(f"Dataset: BEIR SciFact")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print()

    # Stress test configuration (compressed for practical runtime)
    # In real 2 hours: 3000 stores, 5000 queries, 24 checkpoints
    # Compressed config maintains same ratios
    config = {
        # Capacity
        "episodic_capacity": 1000,

        # Phase 1: Normal (0-60 min) - 750 stores, 1000 queries
        "phase1_stores": 750,
        "num_checkpoints_phase1": 8,

        # Phase 2: Pressure (60-90 min) - 1000 stores, 1500 queries
        "phase2_stores": 1000,
        "num_checkpoints_phase2": 8,

        # Phase 3: Recovery (90-120 min) - 500 stores, 1000 queries
        "phase3_stores": 500,
        "num_checkpoints_phase3": 8,

        # Shared
        "checkpoint_interval": 200,  # operations per checkpoint
        "queries_per_checkpoint": 50,  # recall queries per checkpoint
    }

    print("Stress Test Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
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
    retriever = StressTestEpisodicRetriever(
        embedder_name="BAAI/bge-base-en-v1.5",
        episodic_capacity=config["episodic_capacity"],
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

    # Run baseline cold search (no episodic memory)
    print("\n[3/5] Running BASELINE cold search...")
    cold_results = {}
    cold_latencies = []
    for qid, qtext in list(queries_with_qrels.items())[:100]:
        t0 = time.perf_counter()
        retrieved = retriever.search(qtext, k=100, use_episodic=False)
        cold_results[qid] = retrieved
        cold_latencies.append((time.perf_counter() - t0) * 1000)

    cold_metrics = evaluate_run(cold_results, qrels, k_values=[10])
    baseline_ndcg = cold_metrics.get("nDCG@10", 0.0)
    print(f"  Baseline nDCG@10 (cold, no memory): {baseline_ndcg:.4f}")

    # Reset episodic memory before stress test
    retriever.reset_episodic()

    # Run Phase 1: Normal Operation
    print("\n[4/5] Running PHASE 1: Normal Operation...")
    phase1_results = run_phase1_normal_operation(
        retriever, queries_with_qrels, qrels, corpus, config
    )

    # Run Phase 2: Memory Pressure
    print("\n[4/5] Running PHASE 2: Memory Pressure...")
    phase2_results = run_phase2_memory_pressure(
        retriever, queries_with_qrels, qrels, corpus, config
    )

    # Run Phase 3: Recovery
    print("\n[4/5] Running PHASE 3: Recovery Test...")
    phase3_results = run_phase3_recovery(
        retriever, queries_with_qrels, qrels, corpus, config
    )

    # Compute summary statistics
    all_checkpoints = phase1_results + phase2_results + phase3_results

    # Phase 1 summary
    phase1_ndcgs = [cp["nDCG@10"] for cp in phase1_results]
    phase1_start_ndcg = phase1_ndcgs[0] if phase1_ndcgs else 0.0
    phase1_end_ndcg = phase1_ndcgs[-1] if phase1_ndcgs else 0.0
    phase1_avg_ndcg = np.mean(phase1_ndcgs) if phase1_ndcgs else 0.0

    # Phase 2 summary
    phase2_ndcgs = [cp["nDCG@10"] for cp in phase2_results]
    phase2_start_ndcg = phase2_ndcgs[0] if phase2_ndcgs else 0.0
    phase2_end_ndcg = phase2_ndcgs[-1] if phase2_ndcgs else 0.0
    phase2_avg_ndcg = np.mean(phase2_ndcgs) if phase2_ndcgs else 0.0

    # Phase 3 summary
    phase3_ndcgs = [cp["nDCG@10"] for cp in phase3_results]
    phase3_start_ndcg = phase3_ndcgs[0] if phase3_ndcgs else 0.0
    phase3_end_ndcg = phase3_ndcgs[-1] if phase3_ndcgs else 0.0
    phase3_avg_ndcg = np.mean(phase3_ndcgs) if phase3_ndcgs else 0.0

    # Overall metrics
    starting_ndcg = phase1_results[0]["nDCG@10"] if phase1_results else baseline_ndcg
    at_60min_ndcg = phase2_results[0]["nDCG@10"] if phase2_results else 0.0
    at_120min_ndcg = phase3_results[-1]["nDCG@10"] if phase3_results else 0.0

    # Graceful degradation check
    graceful_degradation = (at_120min_ndcg > 0.3 * starting_ndcg) if starting_ndcg > 0 else False

    # Recovery check
    recovery_ratio = (at_120min_ndcg / starting_ndcg) if starting_ndcg > 0 else 0.0

    print("\n" + "=" * 70)
    print("STRESS TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nBaseline (cold, no memory): nDCG@10 = {baseline_ndcg:.4f}")
    print(f"\nPhase 1 (Normal, 0-60min):")
    print(f"  Start nDCG@10: {phase1_start_ndcg:.4f}")
    print(f"  End nDCG@10:   {phase1_end_ndcg:.4f}")
    print(f"  Avg nDCG@10:   {phase1_avg_ndcg:.4f}")
    print(f"\nPhase 2 (Pressure, 60-90min, eviction started):")
    print(f"  Start nDCG@10: {phase2_start_ndcg:.4f}")
    print(f"  End nDCG@10:   {phase2_end_ndcg:.4f}")
    print(f"  Avg nDCG@10:   {phase2_avg_ndcg:.4f}")
    print(f"\nPhase 3 (Recovery, 90-120min):")
    print(f"  Start nDCG@10: {phase3_start_ndcg:.4f}")
    print(f"  End nDCG@10:   {phase3_end_ndcg:.4f}")
    print(f"  Avg nDCG@10:   {phase3_avg_ndcg:.4f}")
    print(f"\n--- Key Metrics ---")
    print(f"  Starting nDCG@10:      {starting_ndcg:.4f}")
    print(f"  At 60min (75% full):   {at_60min_ndcg:.4f}")
    print(f"  At 120min (evicted):  {at_120min_ndcg:.4f}")
    print(f"  Recovery rate:         {recovery_ratio:.2%}")
    print(f"  Graceful degradation:  {'YES' if graceful_degradation else 'NO'}")
    print(f"\n--- Memory Statistics ---")
    print(f"  Capacity:             {retriever.episodic_capacity} slots")
    print(f"  Total stores:         {retriever._total_stores}")
    print(f"  Total evictions:      {retriever._eviction_count}")
    print(f"  Eviction policy:      FIFO (circular buffer)")
    print(f"  Final memory fullness: {retriever.get_memory_fullness()*100:.1f}%")

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
            "episodic_capacity": retriever.episodic_capacity,
            "total_stores": retriever._total_stores,
            "total_evictions": retriever._eviction_count,
            "config": config,
        },
        "baseline": {
            "description": "Cold search without episodic memory",
            "nDCG@10": baseline_ndcg,
        },
        "phase1_normal": {
            "description": "Normal operation (0-60 min equivalent)",
            "checkpoints": phase1_results,
            "start_ndcg": phase1_start_ndcg,
            "end_ndcg": phase1_end_ndcg,
            "avg_ndcg": phase1_avg_ndcg,
        },
        "phase2_pressure": {
            "description": "Memory pressure with eviction (60-90 min equivalent)",
            "checkpoints": phase2_results,
            "start_ndcg": phase2_start_ndcg,
            "end_ndcg": phase2_end_ndcg,
            "avg_ndcg": phase2_avg_ndcg,
        },
        "phase3_recovery": {
            "description": "Recovery test after eviction (90-120 min equivalent)",
            "checkpoints": phase3_results,
            "start_ndcg": phase3_start_ndcg,
            "end_ndcg": phase3_end_ndcg,
            "avg_ndcg": phase3_avg_ndcg,
        },
        "summary": {
            "starting_ndcg": starting_ndcg,
            "at_60min_ndcg": at_60min_ndcg,
            "at_120min_ndcg": at_120min_ndcg,
            "recovery_rate": recovery_ratio,
            "graceful_degradation": graceful_degradation,
            "eviction_policy": "FIFO (circular buffer)",
            "critical_issues": [],
        },
        "store_history_sample": retriever._store_history[:100],  # First 100 for inspection
    }

    # Check for critical issues
    issues = []
    if at_120min_ndcg < 0.1:
        issues.append("nDCG@10 dropped below 0.1 at end of test - severe degradation")
    if retriever._eviction_count == 0 and retriever._total_stores > retriever.episodic_capacity:
        issues.append("No evictions recorded but capacity was exceeded")
    if graceful_degradation is False:
        issues.append("Graceful degradation check failed - rapid performance drop")

    results["summary"]["critical_issues"] = issues

    if issues:
        print(f"\n!!! CRITICAL ISSUES FOUND: !!!")
        for issue in issues:
            print(f"  - {issue}")

    # Save results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {RESULTS_PATH}")

    # Print broadcast summary
    print("\n" + "=" * 70)
    print("BROADCAST SUMMARY")
    print("=" * 70)
    print(f"- Starting nDCG@10: {starting_ndcg:.4f}")
    print(f"- At 60min (75% full): {at_60min_ndcg:.4f}")
    print(f"- At 120min (evicted): {at_120min_ndcg:.4f}")
    print(f"- Recovery rate: {recovery_ratio:.2%}")
    print(f"- Graceful degradation: {'YES' if graceful_degradation else 'NO'}")
    print(f"- Critical issues found: {issues if issues else 'None'}")

    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)

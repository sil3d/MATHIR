"""
REAL BENCHMARK v2 -- Multi-Dataset BEIR (Gold Standard Retrieval Evaluation)
==========================================================================

This is a multi-dataset BEIR benchmark, the v2 of ``real_sota_benchmark.py``.

Datasets (5 BEIR benchmarks, 7 systems, 3 IR metrics):
  * scifact   (300 queries,    5K docs)  -- fact checking
  * nfcorpus  (323 queries,  3.6K docs)  -- bio-medical
  * fiqa      (648 queries,   57K docs)  -- finance QA
  * arguana   (1406 queries, 8.7K docs)  -- argument retrieval
  * scidocs   (1000 queries,  25K docs)  -- scientific citations

Systems compared (7):
  1. BM25 (rank-bm25)
  2. all-MiniLM-L6-v2 + FAISS
  3. BGE-small-en-v1.5 + FAISS
  4. BGE-base-en-v1.5 + FAISS
  5. OptimizedMATHIR (dense only)
  6. OptimizedMATHIR + BM25 (RRF)
  7. OptimizedMATHIR + CE rerank

Metrics (TREC standard):
  nDCG@10, MRR@10, Recall@100

Statistical analysis (per-system, per-dataset):
  * P50 / P95 / P99 / mean / std / min / max of per-query latency
  * Memory footprint of the index (embeddings + FAISS + BM25 tokens)
  * Indexing time, encoding time, search time, rerank time

Output:
  * Prints a clear final table with the cross-dataset BEIR score
  * Saves ``real_sota_benchmark_v2_results.json`` next to this file

Resilience:
  * SSL bypass for Windows (PYTHONHTTPSVERIFY=0).
  * Downloads via urllib -> falls back to HuggingFace ``datasets``.
  * Per-(system, dataset) try/except -- failures are recorded, not crashing.
  * Large datasets (fiqa) can be skipped on OOM via ``SKIP_LARGE_DATASETS``.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import re
import statistics
import sys
import tempfile
import time
import traceback
import urllib.request
import warnings
import zipfile
from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- Windows SSL bypass (must happen BEFORE requests / urllib / ssl) -------
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")

# --- Windows UTF-8 stdout (cp1252 cannot encode a few ASCII-exotic chars) --
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np

# Add project root to path so we can import mathir_optimized
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# 5 BEIR DATASETS
# ============================================================================
DATASETS: Dict[str, Dict[str, Any]] = {
    "scifact": {
        "description": "Fact checking -- scientific claims",
        "num_queries": 300,
        "num_docs_approx": 5_000,
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "hf_name": "BeIR/scifact",
        "hf_qrels": "BeIR/scifact-qrels",
    },
    "nfcorpus": {
        "description": "Bio-medical retrieval",
        "num_queries": 323,
        "num_docs_approx": 3_600,
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "hf_name": "BeIR/nfcorpus",
        "hf_qrels": "BeIR/nfcorpus-qrels",
    },
    "fiqa": {
        "description": "Finance Q&A",
        "num_queries": 648,
        "num_docs_approx": 57_000,
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        "hf_name": "BeIR/fiqa",
        "hf_qrels": "BeIR/fiqa-qrels",
    },
    "arguana": {
        "description": "Argument retrieval (counter-arguments)",
        "num_queries": 1406,
        "num_docs_approx": 8_700,
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
        "hf_name": "BeIR/arguana",
        "hf_qrels": "BeIR/arguana-qrels",
    },
    "scidocs": {
        "description": "Scientific citation retrieval",
        "num_queries": 1000,
        "num_docs_approx": 25_000,
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
        "hf_name": "BeIR/scidocs",
        "hf_qrels": "BeIR/scidocs-qrels",
    },
}

# The 7 systems we evaluate. Each entry is a dict with the system's
# ``name`` (used as the JSON key) and a ``type`` discriminator
# (``"bm25"`` / ``"dense"`` / ``"optimized"``) plus the type-specific
# parameters. The benchmark dispatches on ``type`` to the right runner.
SYSTEMS: List[Dict[str, Any]] = [
    {
        "name": "BM25 (rank-bm25)",
        "type": "bm25",
    },
    {
        "name": "all-MiniLM-L6-v2 + FAISS",
        "type": "dense",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
    },
    {
        "name": "BGE-small-en-v1.5 + FAISS",
        "type": "dense",
        "model": "BAAI/bge-small-en-v1.5",
        "dim": 384,
    },
    {
        "name": "BGE-base-en-v1.5 + FAISS",
        "type": "dense",
        "model": "BAAI/bge-base-en-v1.5",
        "dim": 768,
    },
    {
        "name": "OptimizedMATHIR",
        "type": "optimized",
        "use_bm25": False,
        "use_cross_encoder": False,
    },
    {
        "name": "OptimizedMATHIR + BM25 (RRF)",
        "type": "optimized",
        "use_bm25": True,
        "use_cross_encoder": False,
    },
    {
        "name": "OptimizedMATHIR + CE rerank",
        "type": "optimized",
        "use_bm25": True,
        "use_cross_encoder": True,
        # Use TinyBERT for speed (10x faster than MiniLM-L-6 on CPU,
        # ~1pp nDCG drop). FiQA + scidocs with 57K and 25K docs would
        # otherwise take many hours of CE re-ranking.
        "cross_encoder": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    },
]

# Per-dataset cap on query count for the cross-encoder variants. The CE
# is the bottleneck (10ms-1s/query); we don't want fiqa (648 queries)
# to dominate the run-time. Setting to 0 means "no limit".
CE_MAX_QUERIES_PER_DATASET = 200

# Whether to skip "large" datasets (fiqa) entirely. Set to True if
# memory is tight. The brief explicitly says "skip datasets that fail".
SKIP_LARGE_DATASETS = False
LARGE_DATASET_DOC_THRESHOLD = 30_000  # fiqa is 57K

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "real_sota_benchmark_v2_results.json",
)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beir_data")


# ============================================================================
# SSL / download helpers
# ============================================================================
def _ssl_bypass_context():
    """Build an SSL context that bypasses verification (Windows-friendly)."""
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_beir_dataset(
    name: str,
    _loader: Optional[Callable[[str], bool]] = None,
) -> bool:
    """
    Download and extract a BEIR dataset.

    Order:
      1. Use the injected ``_loader`` (used by tests).
      2. Try the official BEIR URL via ``urllib`` with SSL bypass.
      3. Fall back to HuggingFace ``datasets``.

    Returns True on success, False on failure.
    """
    if _loader is not None:
        return bool(_loader(name))

    if name not in DATASETS:
        print(f"  [FAIL] Unknown dataset: {name}")
        return False

    info = DATASETS[name]
    target_dir = os.path.join(DATA_DIR, name)
    corpus_path = os.path.join(target_dir, "corpus.jsonl")
    if os.path.exists(corpus_path):
        print(f"  [OK] {name} already downloaded at {target_dir}")
        return True

    os.makedirs(target_dir, exist_ok=True)

    # --- 1. urllib (official URL) ---
    url = info.get("url")
    if url:
        zip_path = os.path.join(target_dir, f"{name}.zip")
        try:
            print(f"  Downloading {name} from {url} ...")
            ctx = _ssl_bypass_context()
            with urllib.request.urlopen(url, context=ctx, timeout=120) as resp:
                with open(zip_path, "wb") as f:
                    f.write(resp.read())
            print(f"  Extracting {name} ...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(target_dir)
            os.remove(zip_path)
            print(f"  [OK] {name} ready at {target_dir}")
            return True
        except Exception as e:
            print(f"  [WARN] urllib download failed: {e}")

    # --- 2. HuggingFace fallback ---
    try:
        print(f"  Trying HuggingFace fallback for {name} ...")
        from datasets import load_dataset  # type: ignore
        os.makedirs(os.path.join(target_dir, "qrels"), exist_ok=True)

        corpus_ds = load_dataset(info["hf_name"], "corpus", split="corpus")
        queries_ds = load_dataset(info["hf_name"], "queries", split="queries")
        qrels_ds = load_dataset(info["hf_qrels"], split="train")

        with open(corpus_path, "w", encoding="utf-8") as f:
            for doc in corpus_ds:
                f.write(json.dumps({
                    "_id": doc["_id"],
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                }, ensure_ascii=False) + "\n")

        with open(os.path.join(target_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
            for q in queries_ds:
                f.write(json.dumps({
                    "_id": q["_id"],
                    "text": q.get("text", ""),
                }, ensure_ascii=False) + "\n")

        with open(os.path.join(target_dir, "qrels", "test.tsv"), "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for r in qrels_ds:
                f.write(f"{r['query-id']}\t{r['corpus-id']}\t{r['score']}\n")

        print(f"  [OK] {name} loaded via HuggingFace at {target_dir}")
        return True
    except Exception as e:
        print(f"  [FAIL] HuggingFace fallback also failed: {e}")
        return False


def load_beir_dataset(name: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load a BEIR dataset from local cache.

    Returns: (corpus, queries, qrels)
      corpus:  {doc_id: {"title": ..., "text": ...}}
      queries: {query_id: query_text}
      qrels:   {query_id: {doc_id: relevance}}
    """
    target_dir = os.path.join(DATA_DIR, name)
    # BEIR zip extracts to a nested folder; follow one level down if needed
    for sub in [name, ""]:
        candidate = os.path.join(target_dir, sub) if sub else target_dir
        if os.path.exists(os.path.join(candidate, "corpus.jsonl")):
            target_dir = candidate
            break

    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    with open(os.path.join(target_dir, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }

    with open(os.path.join(target_dir, "queries.jsonl"), "r", encoding="utf-8") as f:
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

    print(
        f"  Loaded {name}: {len(corpus)} docs, {len(queries)} queries, "
        f"{sum(len(v) for v in qrels.values())} relevance judgments"
    )
    return corpus, queries, qrels


# ============================================================================
# TREC-STYLE METRICS
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


def mrr_at_k(relevances: List[int], k: int) -> float:
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved_docs: List[str], relevant_docs: Dict[str, int], k: int) -> float:
    if not relevant_docs:
        return 0.0
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs.keys())
    return len(retrieved_set & relevant_set) / len(relevant_set)


def evaluate_run(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = (10, 100),
) -> Dict[str, float]:
    """nDCG@k / MRR@k / Recall@k averaged over queries with qrels."""
    ndcg_scores: Dict[int, List[float]] = defaultdict(list)
    mrr_scores: Dict[int, List[float]] = defaultdict(list)
    recall_scores: Dict[int, List[float]] = defaultdict(list)

    for qid, retrieved in results.items():
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        retrieved_ids = [doc_id for doc_id, _ in retrieved]
        relevances = [relevant.get(doc_id, 0) for doc_id in retrieved_ids]
        for k in k_values:
            ndcg_scores[k].append(ndcg_at_k(relevances, k))
            mrr_scores[k].append(mrr_at_k(relevances, k))
            recall_scores[k].append(recall_at_k(retrieved_ids, relevant, k))

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"nDCG@{k}"] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0.0
        metrics[f"MRR@{k}"] = float(np.mean(mrr_scores[k])) if mrr_scores[k] else 0.0
        metrics[f"Recall@{k}"] = float(np.mean(recall_scores[k])) if recall_scores[k] else 0.0
    return metrics


# ============================================================================
# LATENCY / MEMORY
# ============================================================================
def latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Aggregate per-query latencies into P50 / P95 / P99 / mean / std."""
    if not latencies_ms:
        return {
            "count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0,
            "p99_ms": 0.0, "std_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0,
        }
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def memory_footprint_mb() -> Dict[str, float]:
    """Current process RSS in MB. Returns a dict with ``total_mb``."""
    try:
        import psutil  # type: ignore
        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / (1024 * 1024)
        return {"rss_mb": rss_mb, "total_mb": rss_mb}
    except Exception:
        return {"rss_mb": 0.0, "total_mb": 0.0}


def _process_memory_now_mb() -> float:
    """Current RSS in MB, or 0 if psutil is unavailable."""
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


# ============================================================================
# SYSTEM RUNNERS -- 7 systems
# ============================================================================
# Common output schema for all runners:
# {
#   "nDCG@10": float, "MRR@10": float, "Recall@100": float,
#   "latency": {p50, p95, p99, mean, std, min, max, count},
#   "memory_mb": float,
#   "encode_time_ms": float, "index_time_ms": float, "search_time_ms": float,
#   "rerank_time_ms": float,
#   "num_queries": int,
#   "config": dict,  # system config (for documentation)
# }
# On failure: {"error": "<message>", "traceback": "..."}

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _run_bm25(corpus: Dict, queries: Dict, qrels: Dict) -> Dict[str, Any]:
    """Baseline 1: BM25 via rank-bm25 (no embeddings)."""
    from rank_bm25 import BM25Okapi

    doc_ids = list(corpus.keys())
    tokenized_corpus = [corpus[did]["text"].lower().split() for did in doc_ids]

    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized_corpus)
    index_time = (time.perf_counter() - t0) * 1000

    results: Dict[str, List[Tuple[str, float]]] = {}
    latencies: List[float] = []
    for qid, qtext in queries.items():
        t1 = time.perf_counter()
        scores = bm25.get_scores(qtext.lower().split())
        top_k = np.argsort(scores)[::-1][:100]
        retrieved = [(doc_ids[i], float(scores[i])) for i in top_k]
        results[qid] = retrieved
        latencies.append((time.perf_counter() - t1) * 1000)

    metrics = evaluate_run(results, qrels, k_values=[10, 100])
    return {
        **metrics,
        "latency": latency_stats(latencies),
        "memory_mb": _process_memory_now_mb(),
        "index_time_ms": index_time,
        "encode_time_ms": 0.0,
        "search_time_ms": float(np.sum(latencies)),
        "rerank_time_ms": 0.0,
        "num_queries": len(queries),
        "config": {"type": "bm25", "library": "rank-bm25"},
    }


def _run_dense_model(model_name: str, corpus: Dict, queries: Dict, qrels: Dict) -> Dict[str, Any]:
    """Baseline 2/3/4: dense embedding model + FAISS cosine."""
    from sentence_transformers import SentenceTransformer
    import faiss  # type: ignore

    model = SentenceTransformer(model_name)

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[did]["title"] + " " + corpus[did]["text"]).strip() for did in doc_ids]

    t0 = time.perf_counter()
    doc_embeddings = model.encode(
        doc_texts, show_progress_bar=False, batch_size=64,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")
    encode_time = (time.perf_counter() - t0) * 1000

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = model.encode(
        query_texts, show_progress_bar=False, batch_size=32,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")

    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)

    results: Dict[str, List[Tuple[str, float]]] = {}
    latencies: List[float] = []
    for i, qid in enumerate(query_ids):
        t1 = time.perf_counter()
        scores, indices = index.search(query_embeddings[i:i + 1], 100)
        retrieved = [
            (doc_ids[int(indices[0, j])], float(scores[0, j]))
            for j in range(indices.shape[1])
            if 0 <= int(indices[0, j]) < len(doc_ids)
        ]
        results[qid] = retrieved
        latencies.append((time.perf_counter() - t1) * 1000)

    metrics = evaluate_run(results, qrels, k_values=[10, 100])
    return {
        **metrics,
        "latency": latency_stats(latencies),
        "memory_mb": _process_memory_now_mb(),
        "encode_time_ms": encode_time,
        "index_time_ms": encode_time,  # encoding IS the indexing here
        "search_time_ms": float(np.sum(latencies)),
        "rerank_time_ms": 0.0,
        "num_queries": len(queries),
        "config": {"type": "dense", "model": model_name, "dim": int(dim), "search": "FAISS"},
    }


def _run_optimized(
    use_bm25: bool,
    use_ce: bool,
    cross_encoder_name: str,
    corpus: Dict,
    queries: Dict,
    qrels: Dict,
) -> Dict[str, Any]:
    """Runner for the 3 OptimizedMATHIR variants."""
    from mathir_optimized import OptimizedMATHIR  # type: ignore

    retriever = OptimizedMATHIR(
        embedder_name="BAAI/bge-base-en-v1.5",
        use_bm25=use_bm25,
        use_cross_encoder=use_ce,
        cross_encoder_name=cross_encoder_name,
    )

    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[did]["title"] + " " + corpus[did]["text"]).strip() for did in doc_ids]

    retriever.index(doc_ids, doc_texts)

    # For CE on large datasets, subsample queries to keep total time bounded
    query_ids = list(queries.keys())
    if use_ce and CE_MAX_QUERIES_PER_DATASET and len(query_ids) > CE_MAX_QUERIES_PER_DATASET:
        rng = np.random.RandomState(42)
        query_ids = list(rng.choice(query_ids, size=CE_MAX_QUERIES_PER_DATASET, replace=False))
        warnings.warn(
            f"[{retriever.embedder_name}] CE subsampled to "
            f"{CE_MAX_QUERIES_PER_DATASET} queries (out of {len(queries)})"
        )

    # Encode queries up front (one-shot for all systems to amortize)
    # Note: retriever also encodes per-query, but we can re-use the model's
    # encoder for speed. For correctness we use retriever.search() per query
    # to faithfully track per-query latency.

    results: Dict[str, List[Tuple[str, float]]] = {}
    latencies: List[float] = []
    for qid in query_ids:
        qtext = queries[qid]
        t1 = time.perf_counter()
        try:
            rids, _ = retriever.search(qtext, k=100, query_text=qtext)
            # Recompute scores as ranks: 1/rank for ordinal scoring
            results[qid] = [(rid, 1.0 / (i + 1)) for i, rid in enumerate(rids)]
        except Exception as e:
            results[qid] = []
            warnings.warn(f"[OptimizedMATHIR] search failed on {qid}: {e}")
        latencies.append((time.perf_counter() - t1) * 1000)

    metrics = evaluate_run(results, qrels, k_values=[10, 100])
    stats = retriever.get_stats()
    return {
        **metrics,
        "latency": latency_stats(latencies),
        "memory_mb": _process_memory_now_mb(),
        "encode_time_ms": stats.get("encode_time_ms", 0.0),
        "index_time_ms": stats.get("index_time_ms", 0.0),
        "search_time_ms": stats.get("search_time_ms", 0.0),
        "rerank_time_ms": stats.get("rerank_time_ms", 0.0),
        "num_queries": len(query_ids),
        "config": {
            "type": "optimized",
            "use_bm25": use_bm25,
            "use_cross_encoder": use_ce,
            "cross_encoder": cross_encoder_name if use_ce else None,
            "embedder": "BAAI/bge-base-en-v1.5",
        },
    }


def run_system(
    system_spec: Dict[str, Any],
    corpus: Dict,
    queries: Dict,
    qrels: Dict,
) -> Dict[str, Any]:
    """Dispatch a system spec to the correct runner, with try/except."""
    name = system_spec.get("name", "?")
    stype = system_spec.get("type", "?")
    try:
        if stype == "bm25":
            return _run_bm25(corpus, queries, qrels)
        if stype == "dense":
            return _run_dense_model(system_spec["model"], corpus, queries, qrels)
        if stype == "optimized":
            return _run_optimized(
                use_bm25=system_spec.get("use_bm25", False),
                use_ce=system_spec.get("use_cross_encoder", False),
                cross_encoder_name=system_spec.get(
                    "cross_encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                ),
                corpus=corpus,
                queries=queries,
                qrels=qrels,
            )
        return {"error": f"Unknown system type: {stype}"}
    except Exception as e:
        tb = traceback.format_exc()
        return {
            "error": str(e),
            "traceback": tb,
            "memory_mb": _process_memory_now_mb(),
        }


# ============================================================================
# FINAL JSON BUILDER
# ============================================================================
def build_final_json(
    per_dataset: Dict[str, Dict[str, Dict[str, Any]]],
    configurations: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build the final BEIR-style summary JSON.

    Schema:
        metadata:      {timestamp, platform, datasets, systems, metrics}
        per_dataset:   {dataset: {system: {metrics + latency + memory_mb + ...}}}
        average:       {system: {nDCG@10_avg, MRR@10_avg, Recall@100_avg,
                                  latency_pXX_ms_avg, memory_mb_avg,
                                  num_datasets_evaluated}}
        configurations:{system: {description, params}}
    """
    configurations = configurations or {}

    # Discover all systems across all datasets
    all_systems = set()
    for ds_results in per_dataset.values():
        all_systems.update(ds_results.keys())

    # Compute per-system averages
    average: Dict[str, Dict[str, Any]] = {}
    for system in sorted(all_systems):
        ndcgs, mrrs, recalls = [], [], []
        lat_p50, lat_p95, lat_p99, lat_mean, lat_std = [], [], [], [], []
        mems: List[float] = []
        num_evaluated = 0
        for ds_name, ds_results in per_dataset.items():
            m = ds_results.get(system)
            if m is None or "error" in m:
                continue
            num_evaluated += 1
            ndcgs.append(m.get("nDCG@10", 0.0))
            mrrs.append(m.get("MRR@10", 0.0))
            recalls.append(m.get("Recall@100", 0.0))
            lat = m.get("latency", {})
            lat_p50.append(lat.get("p50_ms", 0.0))
            lat_p95.append(lat.get("p95_ms", 0.0))
            lat_p99.append(lat.get("p99_ms", 0.0))
            lat_mean.append(lat.get("mean_ms", 0.0))
            lat_std.append(lat.get("std_ms", 0.0))
            mems.append(m.get("memory_mb", 0.0))

        def _mean(xs: List[float]) -> Optional[float]:
            return float(np.mean(xs)) if xs else None

        average[system] = {
            "nDCG@10_avg": _mean(ndcgs),
            "MRR@10_avg": _mean(mrrs),
            "Recall@100_avg": _mean(recalls),
            "latency_p50_ms_avg": _mean(lat_p50),
            "latency_p95_ms_avg": _mean(lat_p95),
            "latency_p99_ms_avg": _mean(lat_p99),
            "latency_mean_ms_avg": _mean(lat_mean),
            "latency_std_ms_avg": _mean(lat_std),
            "memory_mb_avg": _mean(mems),
            "num_datasets_evaluated": num_evaluated,
        }

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "datasets": [
            {
                "name": name,
                "description": info.get("description", ""),
                "num_queries": info.get("num_queries", "?"),
                "num_docs_approx": info.get("num_docs_approx", "?"),
            }
            for name, info in DATASETS.items()
        ],
        "systems": [
            {"name": s.get("name", "?"), "type": s.get("type", "?"), **{
                k: v for k, v in s.items() if k not in ("name", "type")
            }}
            for s in SYSTEMS
        ],
        "metrics": ["nDCG@10", "MRR@10", "Recall@100"],
        "k_values": [10, 100],
    }

    return {
        "metadata": metadata,
        "per_dataset": per_dataset,
        "average": average,
        "configurations": configurations,
    }


# ============================================================================
# FINAL TABLE
# ============================================================================
def print_final_table(
    per_dataset: Dict[str, Dict[str, Dict[str, Any]]],
    systems: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Print a clear final comparison table -- both per-dataset and averaged.

    If ``systems`` is empty/None, the function derives the system order
    from the union of keys in ``per_dataset`` (sorted alphabetically),
    so the table still renders when only partial data is available
    (e.g. for tests with mock data).
    """
    if not systems:
        # Derive from per_dataset; use minimal specs (just "name")
        all_names = set()
        for ds_results in per_dataset.values():
            all_names.update(ds_results.keys())
        systems = [{"name": n} for n in sorted(all_names)]

    final = build_final_json(per_dataset, configurations={s.get("name", "?"): s for s in systems})
    avg = final["average"]
    datasets_run = list(per_dataset.keys())

    print("\n" + "=" * 100)
    print("BEIR v2 -- FINAL RESULTS (Cross-Dataset Average = the 'BEIR score')")
    print("=" * 100)

    # --- Cross-dataset average ---
    print(f"\n[1] Average across {len(datasets_run)} datasets: "
          f"{', '.join(datasets_run)}")
    print("-" * 100)
    header = (
        f"{'System':<40} {'nDCG@10':>8} {'MRR@10':>8} {'Recall@100':>11} "
        f"{'P50(ms)':>9} {'P95(ms)':>9} {'P99(ms)':>9} {'Mem(MB)':>9}"
    )
    print(header)
    print("-" * 100)

    for s in systems:
        name = s.get("name", "?")
        a = avg.get(name, {})
        ndcg = a.get("nDCG@10_avg")
        mrr = a.get("MRR@10_avg")
        recall = a.get("Recall@100_avg")
        p50 = a.get("latency_p50_ms_avg")
        p95 = a.get("latency_p95_ms_avg")
        p99 = a.get("latency_p99_ms_avg")
        mem = a.get("memory_mb_avg")
        n_eval = a.get("num_datasets_evaluated", 0)

        def _fmt(x, w=8, prec=4):
            if x is None:
                return f"{'N/A':>{w}}"
            return f"{x:>{w}.{prec}f}"

        print(
            f"{name:<40} {_fmt(ndcg)} {_fmt(mrr)} {_fmt(recall, 11)} "
            f"{_fmt(p50, 9, 1)} {_fmt(p95, 9, 1)} {_fmt(p99, 9, 1)} {_fmt(mem, 9, 1)}"
        )

    # --- Per-dataset breakdown ---
    print(f"\n[2] Per-dataset nDCG@10 (rows = datasets, columns = systems)")
    print("-" * 100)
    col_w = 22
    sys_names = [s.get("name", "?") for s in systems]
    print(f"{'Dataset':<14} " + " ".join(f"{n[:col_w-1]:<{col_w}}" for n in sys_names))
    print("-" * 100)
    for ds in datasets_run:
        row = f"{ds:<14} "
        for sys_name in sys_names:
            m = per_dataset.get(ds, {}).get(sys_name, {})
            ndcg = m.get("nDCG@10")
            if ndcg is None:
                cell = "FAIL" if "error" in m else "N/A"
            else:
                cell = f"{ndcg:.4f}"
            row += f"{cell:<{col_w}} "
        print(row)

    # --- Highlights ---
    print("\n[3] Highlights (the point of the benchmark)")
    print("-" * 100)
    # Find the best average nDCG@10 system
    best_system = max(
        (s for s in avg.values() if s.get("nDCG@10_avg") is not None),
        key=lambda x: x["nDCG@10_avg"],
        default=None,
    )
    if best_system:
        best_name = next((n for n, a in avg.items() if a is best_system), "?")
        best_ndcg = best_system["nDCG@10_avg"]
        # Find OptimizedMATHIR + CE rerank
        optimized_ce = avg.get("OptimizedMATHIR + CE rerank", {})
        opt_ndcg = optimized_ce.get("nDCG@10_avg")
        if opt_ndcg is not None:
            delta = opt_ndcg - best_ndcg
            sign = "above" if delta >= 0 else "below"
            print(f"  Best system on average:        {best_name:<35} nDCG@10 = {best_ndcg:.4f}")
            print(f"  OptimizedMATHIR + CE rerank:   {' ' * 35} nDCG@10 = {opt_ndcg:.4f}")
            print(f"  Delta (OptimizedMATHIR - best):     {delta:+.4f} ({sign} the best)")
        else:
            print(f"  Best system on average:        {best_name:<35} nDCG@10 = {best_ndcg:.4f}")
            print(f"  OptimizedMATHIR + CE rerank:   did not run / failed on all datasets")
    print()


# ============================================================================
# MAIN
# ============================================================================
def main(
    datasets_to_run: Optional[List[str]] = None,
    systems_to_run: Optional[List[str]] = None,
    skip_large: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run the full multi-dataset BEIR benchmark.

    Args:
        datasets_to_run: subset of DATASETS to evaluate. None = all.
        systems_to_run: subset of SYSTEMS to evaluate. None = all.
        skip_large: override the SKIP_LARGE_DATASETS global.

    Returns:
        The final JSON dict (also saved to disk).
    """
    if datasets_to_run is None:
        datasets_to_run = list(DATASETS.keys())
    if systems_to_run is None:
        systems_to_run = [s["name"] for s in SYSTEMS]
    systems = [s for s in SYSTEMS if s["name"] in systems_to_run]
    if skip_large is None:
        skip_large = SKIP_LARGE_DATASETS

    print("=" * 100)
    print("REAL BENCHMARK v2 -- Multi-Dataset BEIR (Gold Standard Retrieval Evaluation)")
    print("=" * 100)
    print(f"Datasets: {', '.join(datasets_to_run)}")
    print(f"Systems:  {len(systems)}  ({', '.join(s['name'] for s in systems)})")
    print(f"Metrics:  nDCG@10, MRR@10, Recall@100 (TREC standard)")
    print(f"Platform: {platform.platform()}")
    print()

    per_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ds_name in datasets_to_run:
        ds_info = DATASETS[ds_name]
        n_docs = ds_info.get("num_docs_approx", 0)
        if skip_large and n_docs > LARGE_DATASET_DOC_THRESHOLD:
            print(f"\n[SKIP] {ds_name} ({n_docs} docs > "
                  f"{LARGE_DATASET_DOC_THRESHOLD} threshold)")
            continue

        print(f"\n{'=' * 100}")
        print(f"DATASET: {ds_name}  --  {ds_info.get('description', '')}")
        print(f"  (queries~{ds_info.get('num_queries', '?')}, docs~{n_docs})")
        print("=" * 100)

        # --- 1. Download ---
        print(f"\n[1/3] Downloading {ds_name}...")
        if not download_beir_dataset(ds_name):
            print(f"  Could not download {ds_name} -- skipping.")
            continue

        # --- 2. Load ---
        print(f"\n[2/3] Loading {ds_name}...")
        try:
            corpus, queries, qrels = load_beir_dataset(ds_name)
        except Exception as e:
            print(f"  Load failed: {e}")
            continue

        # --- 3. Run all systems ---
        print(f"\n[3/3] Running {len(systems)} systems on {ds_name}...")
        ds_results: Dict[str, Dict[str, Any]] = {}

        for sys_spec in systems:
            sys_name = sys_spec.get("name", "?")
            print(f"\n--- {sys_name} ---")
            try:
                t0 = time.perf_counter()
                res = run_system(sys_spec, corpus, queries, qrels)
                elapsed = (time.perf_counter() - t0) * 1000
                if "error" in res:
                    print(f"  [FAIL] {res['error'][:200]}")
                else:
                    print(
                        f"  nDCG@10={res.get('nDCG@10', 0):.4f}  "
                        f"MRR@10={res.get('MRR@10', 0):.4f}  "
                        f"Recall@100={res.get('Recall@100', 0):.4f}  "
                        f"total={elapsed:.0f}ms  "
                        f"P50={res.get('latency', {}).get('p50_ms', 0):.1f}ms  "
                        f"P99={res.get('latency', {}).get('p99_ms', 0):.1f}ms"
                    )
                ds_results[sys_name] = res
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  [CRASH] {e}\n{tb[:500]}")
                ds_results[sys_name] = {
                    "error": str(e),
                    "traceback": tb[:2000],
                    "memory_mb": _process_memory_now_mb(),
                }
            # Free memory between systems
            gc.collect()

        per_dataset[ds_name] = ds_results

    # --- 4. Build & save final JSON ---
    configurations = {s.get("name", "?"): s for s in systems}
    final = build_final_json(per_dataset, configurations=configurations)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {RESULTS_PATH}")

    # --- 5. Print final table ---
    print_final_table(per_dataset, systems)
    return final


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real SOTA benchmark v2 -- multi-dataset BEIR")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Subset of datasets to run (default: all 5)",
    )
    parser.add_argument(
        "--systems", nargs="+", default=None,
        help="Subset of systems to run (default: all 7)",
    )
    parser.add_argument(
        "--skip-large", action="store_true",
        help=f"Skip datasets with > {LARGE_DATASET_DOC_THRESHOLD} docs",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: only run scifact (smallest dataset), all systems",
    )
    args = parser.parse_args()

    if args.smoke:
        args.datasets = ["scifact"]

    main(
        datasets_to_run=args.datasets,
        systems_to_run=args.systems,
        skip_large=args.skip_large,
    )

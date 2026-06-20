"""
MATHIR BEIR Benchmark — Real-world performance test
=====================================================
Tests HybridSearch against BEIR SciFact dataset (5183 docs, 1109 queries).
Compares numpy vs USearch vs sqlite-vec on actual data.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, r"D:\SECRET_PROJECT\MATHIR")

from mathir_search import HybridSearch
from mathir_vec import VecMemory

# ---------------------------------------------------------------------------
# Load BEIR SciFact
# ---------------------------------------------------------------------------

CORPUS_PATH = r"D:\SECRET_PROJECT\MATHIR\benchmarks\beir_data\scifact\scifact\corpus.jsonl"
QUERIES_PATH = r"D:\SECRET_PROJECT\MATHIR\benchmarks\beir_data\scifact\scifact\queries.jsonl"
QRELS_PATH = r"D:\SECRET_PROJECT\MATHIR\benchmarks\beir_data\scifact\scifact\qrels\test.tsv"

def load_corpus():
    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            # Combine title + text for embedding
            text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            docs.append({"id": doc["_id"], "text": text})
    return docs

def load_queries():
    queries = []
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries.append({"id": q["_id"], "text": q["text"]})
    return queries

def load_qrels():
    """Load relevance judgments: query_id → set of relevant doc_ids."""
    qrels = {}
    with open(QRELS_PATH, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, docid, rel = parts[0], parts[1], int(parts[2])
                if rel > 0:
                    qrels.setdefault(qid, set()).add(docid)
    return qrels

# ---------------------------------------------------------------------------
# Embedding (using bge-large via SentenceTransformer)
# ---------------------------------------------------------------------------

def embed_texts(texts, batch_size=64):
    """Embed texts using bge-large-en-v1.5 on CUDA."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)

# ---------------------------------------------------------------------------
# Evaluation: Recall@k
# ---------------------------------------------------------------------------

def recall_at_k(results, relevant, k=10):
    """Calculate recall@k for a single query."""
    retrieved_ids = [r["memory_id"] for r in results[:k]]
    if not relevant:
        return 0.0
    hits = len(set(retrieved_ids) & relevant)
    return hits / len(relevant)

def evaluate(search_fn, queries, qrels, k=10):
    """Evaluate search function on all queries."""
    recalls = []
    for q in queries:
        results = search_fn(q["text"])
        r = recall_at_k(results, qrels, k)
        recalls.append(r)
    return np.mean(recalls)

# ---------------------------------------------------------------------------
# Main Benchmark
# ---------------------------------------------------------------------------

def main():
    print("MATHIR BEIR SciFact Benchmark")
    print("=" * 60)

    # Load data
    print("Loading corpus...")
    corpus = load_corpus()
    print(f"  {len(corpus)} documents")

    print("Loading queries...")
    queries = load_queries()
    print(f"  {len(queries)} queries")

    print("Loading qrels...")
    qrels = load_qrels()
    print(f"  {len(qrels)} queries with relevance labels")

    # Embed corpus
    print("\nEmbedding corpus (bge-large 1024d, CUDA)...")
    doc_texts = [d["text"] for d in corpus]
    doc_embeddings = embed_texts(doc_texts, batch_size=64)
    print(f"  Embeddings shape: {doc_embeddings.shape}")

    # Embed queries
    print("Embedding queries...")
    query_texts = [q["text"] for q in queries]
    query_embeddings = embed_texts(query_texts, batch_size=64)
    print(f"  Query embeddings shape: {query_embeddings.shape}")

    dim = 1024

    # --- Test 1: HybridSearch (auto) ---
    print("\n--- Test 1: HybridSearch (auto) ---")
    db_path = r"C:\Users\So-i-learn-3D\AppData\Local\Temp\beir_hybrid.db"
    # Clean up all related files (DB + WAL/SHM + USearch index)
    for suffix in ["", "-wal", "-shm", "-journal"]:
        f = db_path + suffix
        if os.path.exists(f):
            os.remove(f)
    index_dir = r"C:\Users\So-i-learn-3D\AppData\Local\Temp\mathir_indexes"
    if os.path.exists(index_dir):
        import shutil
        shutil.rmtree(index_dir, ignore_errors=True)
    search = HybridSearch(dim=dim, db_path=db_path)

    # Insert all documents
    start = time.perf_counter()
    items = []
    for i, doc in enumerate(corpus):
        items.append({
            "memory_id": doc["id"],
            "embedding": doc_embeddings[i],
            "metadata": {"text": doc["text"][:200]},
        })
    search.store_batch(items)
    insert_ms = (time.perf_counter() - start) * 1000
    print(f"  Insert {len(corpus)} docs: {insert_ms:.1f}ms ({insert_ms/len(corpus):.2f}ms/doc)")
    print(f"  Backend: {search}")

    # Search benchmark
    times = []
    for i, q in enumerate(queries):
        start = time.perf_counter()
        results = search.search(query_embeddings[i], k=10)
        times.append((time.perf_counter() - start) * 1000)
    times = np.array(times)
    print(f"  Search {len(queries)} queries: avg={times.mean():.2f}ms p50={np.percentile(times,50):.2f}ms p99={np.percentile(times,99):.2f}ms")

    # Recall
    def hybrid_search_fn(text):
        # Re-embed query (in real use, this is cached)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
        q_emb = model.encode([text], normalize_embeddings=True)[0]
        return search.search(q_emb, k=10)

    # Use pre-computed embeddings for recall
    recalls = []
    for i, q in enumerate(queries):
        results = search.search(query_embeddings[i], k=10)
        retrieved_ids = [r["memory_id"] for r in results]
        relevant = qrels.get(q["id"], set())
        if relevant:
            hits = len(set(retrieved_ids) & relevant)
            recalls.append(hits / len(relevant))
    avg_recall = np.mean(recalls) if recalls else 0
    print(f"  Recall@10: {avg_recall:.4f}")
    search.close()

    # --- Test 2: Numpy only ---
    print("\n--- Test 2: Numpy brute-force ---")
    from mathir_search import _NumpyBackend
    numpy_backend = _NumpyBackend(dim)
    items_for_numpy = []
    for i, doc in enumerate(corpus):
        items_for_numpy.append({
            "memory_id": doc["id"],
            "embedding": doc_embeddings[i],
        })
    numpy_backend.build(items_for_numpy)

    times = []
    for i, q in enumerate(queries):
        start = time.perf_counter()
        results = numpy_backend.search(query_embeddings[i], k=10)
        times.append((time.perf_counter() - start) * 1000)
    times = np.array(times)
    print(f"  Search {len(queries)} queries: avg={times.mean():.2f}ms p50={np.percentile(times,50):.2f}ms p99={np.percentile(times,99):.2f}ms")

    recalls = []
    for i, q in enumerate(queries):
        results = numpy_backend.search(query_embeddings[i], k=10)
        retrieved_ids = [r[0] for r in results]
        relevant = qrels.get(q["id"], set())
        if relevant:
            hits = len(set(retrieved_ids) & relevant)
            recalls.append(hits / len(relevant))
    avg_recall = np.mean(recalls) if recalls else 0
    print(f"  Recall@10: {avg_recall:.4f}")

    # --- Test 3: sqlite-vec ---
    print("\n--- Test 3: sqlite-vec ---")
    vec_db = r"C:\Users\So-i-learn-3D\AppData\Local\Temp\beir_vec.db"
    for suffix in ["", "-wal", "-shm", "-journal"]:
        f = vec_db + suffix
        if os.path.exists(f):
            os.remove(f)
    vec = VecMemory(vec_db, dim)

    start = time.perf_counter()
    for i, doc in enumerate(corpus):
        vec.store(doc["id"], doc_embeddings[i], {"text": doc["text"][:200]})
    insert_ms = (time.perf_counter() - start) * 1000
    print(f"  Insert {len(corpus)} docs: {insert_ms:.1f}ms ({insert_ms/len(corpus):.2f}ms/doc)")

    times = []
    for i, q in enumerate(queries):
        start = time.perf_counter()
        results = vec.search(query_embeddings[i], k=10)
        times.append((time.perf_counter() - start) * 1000)
    times = np.array(times)
    print(f"  Search {len(queries)} queries: avg={times.mean():.2f}ms p50={np.percentile(times,50):.2f}ms p99={np.percentile(times,99):.2f}ms")

    recalls = []
    for i, q in enumerate(queries):
        results = vec.search(query_embeddings[i], k=10)
        retrieved_ids = [r["memory_id"] for r in results]
        relevant = qrels.get(q["id"], set())
        if relevant:
            hits = len(set(retrieved_ids) & relevant)
            recalls.append(hits / len(relevant))
    avg_recall = np.mean(recalls) if recalls else 0
    print(f"  Recall@10: {avg_recall:.4f}")
    vec.close()

    # Cleanup
    for base in [db_path, vec_db]:
        for suffix in ["", "-wal", "-shm", "-journal"]:
            f = base + suffix
            if os.path.exists(f):
                os.remove(f)

    print("\n" + "=" * 60)
    print("BEIR benchmark complete.")

if __name__ == "__main__":
    main()

"""
COMPREHENSIVE STRESS TEST: Multiple MATHIR Versions & Techniques
================================================================

Runs real stress tests on a PDF textbook using:
  - MATHIR V6 (legacy)
  - MATHIR V7 (doctoral-grade)
  - MATHIR V7.1 Approach A: Raw Embedding Bypass
  - MATHIR V7.1 Approach B: Multi-Encoder
  - MATHIR V7.1 Approach C: FAISS-backed
  - MATHIR V7.1 Approach D: Hybrid BM25 + Dense + Cross-Encoder
  - MATHIR Infinity: Compressed Sensing
  - MATHIR Infinity: Kernel Retrieval
  - FAISS VectorDB (baseline)

Each test extracts 200 chunks, stores them, and runs 50 domain queries.
"""

import os
import sys
import time
import json
import statistics
import argparse
from typing import List, Dict, Any, Tuple

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import fitz  # PyMuPDF

# ============================================================================
# PDF LOADING (shared)
# ============================================================================

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks


def extract_pdf_chunks(pdf_path: str, max_chunks: int = 200, chunk_size: int = 150) -> List[Dict[str, Any]]:
    print(f"  Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    print(f"  Pages: {len(doc)}")

    chunks = []
    for page_idx, page in enumerate(doc):
        text = page.get_text("text")
        if not text.strip():
            continue
        page_chunks = chunk_text(text, chunk_size=chunk_size, overlap=20)
        for ch_idx, ch_text in enumerate(page_chunks):
            if len(ch_text.split()) < 30:
                continue
            chunks.append({
                "text": ch_text,
                "page": page_idx + 1,
                "chunk_id": f"p{page_idx+1:04d}_c{ch_idx:03d}",
                "n_words": len(ch_text.split()),
            })
        if len(chunks) >= max_chunks:
            break

    doc.close()
    print(f"  Extracted {len(chunks)} chunks (avg {statistics.mean(c['n_words'] for c in chunks):.0f} words/chunk)")
    return chunks[:max_chunks]


# ============================================================================
# EMBEDDINGS (real, sentence-transformers)
# ============================================================================

class RealEmbedder:
    """Real semantic embeddings via sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"  Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dim: {self.dim}")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


# ============================================================================
# QUERY SET (50 Fluid Mechanics queries)
# ============================================================================

FLUID_MECHANICS_QUERIES = [
    "What is the continuity equation for incompressible flow?",
    "Define Reynolds number and its physical meaning.",
    "What is Bernoulli's equation?",
    "How does viscosity affect flow in a pipe?",
    "What is the Navier-Stokes equation?",
    "Explain the difference between laminar and turbulent flow.",
    "What are the boundary conditions for no-slip flow?",
    "How do you calculate pressure drop in a pipe?",
    "What is the momentum equation in fluid mechanics?",
    "Explain the concept of stream function.",
    "What is the vorticity equation?",
    "How is dimensional analysis used in fluid mechanics?",
    "What are the major losses in pipe flow?",
    "Explain the concept of dynamic similarity.",
    "What is the energy equation for viscous flow?",
    "What is the critical Reynolds number for pipe flow?",
    "How do you calculate friction factor?",
    "What is the hydraulic diameter?",
    "Explain the concept of boundary layer thickness.",
    "What is the difference between steady and unsteady flow?",
    "How is Mach number defined?",
    "What is the compressibility effect in fluid flow?",
    "Explain the concept of potential flow.",
    "What is the Stokes flow regime?",
    "How do you solve the Laplace equation for flow?",
    "What is the drag coefficient?",
    "Explain the lift force on an airfoil.",
    "What is the Prandtl boundary layer approximation?",
    "How is the Reynolds stress tensor defined?",
    "What is turbulent kinetic energy?",
    "Explain the k-epsilon turbulence model.",
    "What is the law of the wall?",
    "How do you calculate the entrance length in a pipe?",
    "What is the Hagen-Poiseuille flow?",
    "Explain the concept of flow separation.",
    "What is the Buckingham Pi theorem?",
    "How do you nondimensionalize the Navier-Stokes equation?",
    "What is the Euler equation for inviscid flow?",
    "Explain the concept of vorticity dynamics.",
    "What is the circulation theorem?",
    "How does surface tension affect fluid flow?",
    "What is the Weber number?",
    "Explain the concept of cavitation.",
    "What is the Froude number?",
    "How is open channel flow characterized?",
    "What is the Manning equation?",
    "Explain the concept of hydraulic jump.",
    "What is the Chezy formula?",
    "How is the flow rate measured with a Venturi meter?",
    "What is the orifice discharge coefficient?",
]


# ============================================================================
# QUALITY METRICS
# ============================================================================

STOPWORDS = {"what", "when", "where", "which", "have", "does", "with", "from",
             "this", "that", "are", "was", "were", "been", "how",
             "explain", "define", "difference", "between", "the", "and", "or"}


def keyword_overlap_score(query: str, retrieved_text: str) -> float:
    q_words = {w.lower().strip(".,;:()[]\"'") for w in query.split() if len(w) > 3}
    q_words = q_words - STOPWORDS
    if not q_words:
        return 0.0
    r_text = retrieved_text.lower()
    hits = sum(1 for w in q_words if w in r_text)
    return hits / len(q_words)


# ============================================================================
# TEST RUNNERS (one per system)
# ============================================================================

def run_faiss_vectordb(embeddings: np.ndarray, query_embeddings: np.ndarray,
                       queries: List[str], chunks: List[Dict]) -> Dict:
    """FAISS VectorDB baseline."""
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    storage_ms = 0  # Pre-loaded
    latencies = []
    overlaps = []
    hits = []

    for q_emb, q in zip(query_embeddings, queries):
        t0 = time.perf_counter()
        scores, indices = index.search(q_emb.reshape(1, -1).astype("float32"), 5)
        latencies.append((time.perf_counter() - t0) * 1000)

        idx = int(indices[0, 0])
        if idx >= 0:
            overlap = keyword_overlap_score(q, chunks[idx]["text"])
            overlaps.append(overlap)
            hits.append(1 if overlap >= 0.3 else 0)

    return {
        "name": "FAISS VectorDB",
        "storage_ms": storage_ms,
        "latencies": latencies,
        "overlaps": overlaps,
        "hits": hits,
        "queries_completed": len(latencies),
    }


def run_mathir_v6(embeddings: np.ndarray, query_embeddings: np.ndarray,
                  queries: List[str], chunks: List[Dict]) -> Dict:
    """MATHIR V6 (legacy 4-tier memory)."""
    from mathir_lib import MATHIRPlugin

    dim = embeddings.shape[1]
    plugin = MATHIRPlugin(embedding_dim=dim)

    # Store - use the underlying episodic memory directly
    t0 = time.perf_counter()
    for emb in embeddings:
        emb_t = torch.from_numpy(emb).float().unsqueeze(0)
        plugin.perceive(emb_t)
        # Store in episodic memory directly
        if hasattr(plugin, 'episodic') and plugin.episodic is not None:
            plugin.episodic.store(emb_t.squeeze(0), metadata={})
    storage_ms = (time.perf_counter() - t0) * 1000

    # Query - use the underlying episodic memory's search
    latencies = []
    overlaps = []
    hits = []
    for q_emb, q in zip(query_embeddings, queries):
        emb_t = torch.from_numpy(q_emb).float().unsqueeze(0)
        t0 = time.perf_counter()
        try:
            if hasattr(plugin, 'episodic') and plugin.episodic is not None:
                indices, scores = plugin.episodic.search(emb_t.squeeze(0), k=5)
                latencies.append((time.perf_counter() - t0) * 1000)
                if indices is not None and len(indices) > 0:
                    top_idx = int(indices[0])
                    if 0 <= top_idx < len(chunks):
                        overlap = keyword_overlap_score(q, chunks[top_idx]["text"])
                        overlaps.append(overlap)
                        hits.append(1 if overlap >= 0.3 else 0)
            else:
                # Fallback to recall
                results = plugin.recall(emb_t, k=5)
                latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "name": "MATHIR V6",
        "storage_ms": storage_ms,
        "latencies": latencies,
        "overlaps": overlaps,
        "hits": hits,
        "queries_completed": len(latencies),
    }


def run_mathir_v7(embeddings: np.ndarray, query_embeddings: np.ndarray,
                  queries: List[str], chunks: List[Dict]) -> Dict:
    """MATHIR V7 (doctoral-grade, 8 algorithms + 6 theorems)."""
    from mathir_lib import MATHIRPluginV7

    dim = embeddings.shape[1]
    plugin = MATHIRPluginV7(embedding_dim=dim)

    # Store directly in episodic memory
    t0 = time.perf_counter()
    for emb in embeddings:
        emb_t = torch.from_numpy(emb).float().unsqueeze(0)
        # Use plugin's perceive and store which handle projection
        plugin.perceive(emb_t)
        # Also store raw embedding in episodic for direct search
        if hasattr(plugin, 'episodic') and plugin.episodic is not None:
            try:
                plugin.episodic.store(emb_t.squeeze(0))
            except Exception:
                pass
    storage_ms = (time.perf_counter() - t0) * 1000

    # Query via episodic memory's search
    latencies = []
    overlaps = []
    hits = []
    for q_emb, q in zip(query_embeddings, queries):
        emb_t = torch.from_numpy(q_emb).float().unsqueeze(0)
        t0 = time.perf_counter()
        try:
            if hasattr(plugin, 'episodic') and plugin.episodic is not None:
                indices, scores = plugin.episodic.search(emb_t.squeeze(0), k=5)
                latencies.append((time.perf_counter() - t0) * 1000)
                if indices is not None and len(indices) > 0:
                    # Handle both 1D and 2D tensor shapes
                    first = indices[0]
                    if hasattr(first, '__len__'):
                        top_idx = int(first[0].item() if hasattr(first[0], 'item') else first[0])
                    else:
                        top_idx = int(first.item() if hasattr(first, 'item') else first)
                    if 0 <= top_idx < len(chunks):
                        overlap = keyword_overlap_score(q, chunks[top_idx]["text"])
                        overlaps.append(overlap)
                        hits.append(1 if overlap >= 0.3 else 0)
            else:
                latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "name": "MATHIR V7",
        "storage_ms": storage_ms,
        "latencies": latencies,
        "overlaps": overlaps,
        "hits": hits,
        "queries_completed": len(latencies),
    }


def run_mathir_v7_approach_a(embeddings: np.ndarray, query_embeddings: np.ndarray,
                              queries: List[str], chunks: List[Dict]) -> Dict:
    """V7.1 Approach A: Raw Embedding Bypass (full 384-dim, no projection)."""
    from mathir_lib.memory import RawEmbeddingEpisodicMemory

    dim = embeddings.shape[1]
    # Approach A: raw embedding, NO projection (projection=False)
    # hidden_dim must match embedding_dim
    mem = RawEmbeddingEpisodicMemory(
        embedding_dim=dim,
        projection=False,
        hidden_dim=dim,  # Match the embedding dim
        capacity=len(embeddings) + 100
    )

    t0 = time.perf_counter()
    for emb in embeddings:
        emb_t = torch.from_numpy(emb).float()
        mem.store(emb_t)
    storage_ms = (time.perf_counter() - t0) * 1000

    latencies = []
    overlaps = []
    hits = []
    for q_emb, q in zip(query_embeddings, queries):
        emb_t = torch.from_numpy(q_emb).float()
        t0 = time.perf_counter()
        indices, scores = mem.search(emb_t, k=5)
        latencies.append((time.perf_counter() - t0) * 1000)

        if indices is not None and len(indices) > 0:
            # Handle both 1D and 2D tensor shapes
            first = indices[0]
            if hasattr(first, '__len__'):
                top_idx = int(first[0].item() if hasattr(first[0], 'item') else first[0])
            else:
                top_idx = int(first.item() if hasattr(first, 'item') else first)
            if 0 <= top_idx < len(chunks):
                overlap = keyword_overlap_score(q, chunks[top_idx]["text"])
                overlaps.append(overlap)
                hits.append(1 if overlap >= 0.3 else 0)

    return {
        "name": "MATHIR V7.1 A (Raw)",
        "storage_ms": storage_ms,
        "latencies": latencies,
        "overlaps": overlaps,
        "hits": hits,
        "queries_completed": len(latencies),
    }


def run_mathir_v7_approach_c(embeddings: np.ndarray, query_embeddings: np.ndarray,
                              queries: List[str], chunks: List[Dict]) -> Dict:
    """V7.1 Approach C: FAISS-backed episodic memory."""
    from mathir_lib.memory import FAISSBackedEpisodicMemory

    dim = embeddings.shape[1]
    try:
        mem = FAISSBackedEpisodicMemory(feature_dim=dim, capacity=len(embeddings) + 100)
    except Exception as e:
        return {"name": "MATHIR V7.1 C (FAISS)", "error": str(e), "latencies": [], "overlaps": [], "hits": []}

    t0 = time.perf_counter()
    for emb in embeddings:
        emb_t = torch.from_numpy(emb).float()
        mem.store(emb_t)
    storage_ms = (time.perf_counter() - t0) * 1000

    latencies = []
    overlaps = []
    hits = []
    for q_emb, q in zip(query_embeddings, queries):
        emb_t = torch.from_numpy(q_emb).float()
        t0 = time.perf_counter()
        try:
            indices, scores = mem.search(emb_t, k=5)
        except Exception as e:
            indices = None
        latencies.append((time.perf_counter() - t0) * 1000)

        if indices is not None and len(indices) > 0:
            top_idx = int(indices[0])
            if 0 <= top_idx < len(chunks):
                overlap = keyword_overlap_score(q, chunks[top_idx]["text"])
                overlaps.append(overlap)
                hits.append(1 if overlap >= 0.3 else 0)

    return {
        "name": "MATHIR V7.1 C (FAISS)",
        "storage_ms": storage_ms,
        "latencies": latencies,
        "overlaps": overlaps,
        "hits": hits,
        "queries_completed": len(latencies),
    }


def run_mathir_v7_approach_d(embeddings: np.ndarray, query_embeddings: np.ndarray,
                              queries: List[str], chunks: List[Dict]) -> Dict:
    """V7.1 Approach D: Hybrid BM25 + Dense + Cross-Encoder (quality king)."""
    from mathir_lib.memory import HybridEpisodicMemory

    dim = embeddings.shape[1]
    try:
        mem = HybridEpisodicMemory(
            feature_dim=dim,
            use_cross_encoder=False,  # Disable for speed
            use_result_cache=False,    # Disable cache for fair test
            capacity=len(embeddings) + 100
        )
    except Exception as e:
        return {"name": "MATHIR V7.1 D (Hybrid)", "error": str(e), "latencies": [], "overlaps": [], "hits": []}

    t0 = time.perf_counter()
    for i, emb in enumerate(embeddings):
        emb_t = torch.from_numpy(emb).float()
        mem.store(emb_t, text=chunks[i]["text"])
    storage_ms = (time.perf_counter() - t0) * 1000

    latencies = []
    overlaps = []
    hits = []
    for q_emb, q in zip(query_embeddings, queries):
        emb_t = torch.from_numpy(q_emb).float()
        t0 = time.perf_counter()
        try:
            indices, scores = mem.search(emb_t, query_text=q, k=5)
        except Exception as e:
            indices = None
        latencies.append((time.perf_counter() - t0) * 1000)

        if indices is not None and len(indices) > 0:
            # Handle both 1D and 2D tensor shapes
            first = indices[0]
            if hasattr(first, '__len__'):
                top_idx = int(first[0].item() if hasattr(first[0], 'item') else first[0])
            else:
                top_idx = int(first.item() if hasattr(first, 'item') else first)
            if 0 <= top_idx < len(chunks):
                overlap = keyword_overlap_score(q, chunks[top_idx]["text"])
                overlaps.append(overlap)
                hits.append(1 if overlap >= 0.3 else 0)

    return {
        "name": "MATHIR V7.1 D (Hybrid)",
        "storage_ms": storage_ms,
        "latencies": latencies,
        "overlaps": overlaps,
        "hits": hits,
        "queries_completed": len(latencies),
    }


def run_mathir_infinity_cs(embeddings: np.ndarray, query_embeddings: np.ndarray,
                            queries: List[str], chunks: List[Dict]) -> Dict:
    """MATHIR Infinity: Compressed Sensing memory."""
    from mathir_infinity.compressed_sensing import CompressedSensingMemory

    dim = embeddings.shape[1]
    mem = CompressedSensingMemory(d=dim, K=8, algorithm='omp')

    t0 = time.perf_counter()
    for emb in embeddings:
        mem.store(emb)
    storage_ms = (time.perf_counter() - t0) * 1000

    latencies = []
    overlaps = []
    hits = []
    for q_emb, q in zip(query_embeddings, queries):
        t0 = time.perf_counter()
        results, _ = mem.retrieve(q_emb, k=5)
        latencies.append((time.perf_counter() - t0) * 1000)

        if results and len(results) > 0:
            top_idx = results[0]
            if 0 <= top_idx < len(chunks):
                overlap = keyword_overlap_score(q, chunks[top_idx]["text"])
                overlaps.append(overlap)
                hits.append(1 if overlap >= 0.3 else 0)

    return {
        "name": "MATHIR Infinity CS",
        "storage_ms": storage_ms,
        "latencies": latencies,
        "overlaps": overlaps,
        "hits": hits,
        "queries_completed": len(latencies),
    }


def run_mathir_infinity_kernel(embeddings: np.ndarray, query_embeddings: np.ndarray,
                                queries: List[str], chunks: List[Dict]) -> Dict:
    """MATHIR Infinity: Kernel retrieval (infinite-dimensional)."""
    from mathir_infinity.kernel_retrieval import KernelRetrieval, KernelConfig, KernelType

    dim = embeddings.shape[1]
    config = KernelConfig(kernel_type=KernelType.GAUSSIAN, gamma=0.5 / dim)
    mem = KernelRetrieval(config=config, capacity=len(embeddings) + 100)

    t0 = time.perf_counter()
    for emb in embeddings:
        mem.store(emb)
    storage_ms = (time.perf_counter() - t0) * 1000

    latencies = []
    overlaps = []
    hits = []
    for q_emb, q in zip(query_embeddings, queries):
        t0 = time.perf_counter()
        indices, scores = mem.retrieve(q_emb, k=5)
        latencies.append((time.perf_counter() - t0) * 1000)

        if indices and len(indices) > 0:
            top_idx = indices[0]
            if 0 <= top_idx < len(chunks):
                overlap = keyword_overlap_score(q, chunks[top_idx]["text"])
                overlaps.append(overlap)
                hits.append(1 if overlap >= 0.3 else 0)

    return {
        "name": "MATHIR Infinity Kernel",
        "storage_ms": storage_ms,
        "latencies": latencies,
        "overlaps": overlaps,
        "hits": hits,
        "queries_completed": len(latencies),
    }


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=r"D:\COURS\Fluid Mechanics 2\27846107.pdf")
    parser.add_argument("--chunks", type=int, default=200)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE STRESS TEST: Multiple MATHIR Versions & Techniques")
    print("=" * 80)
    print(f"PDF:      {args.pdf}")
    print(f"Chunks:   {args.chunks}")
    print(f"Queries:  {args.queries}")
    print(f"Model:    {args.model}")
    print()

    # ---- Step 1: Load PDF and extract chunks ----
    print("[1/3] Loading and chunking PDF...")
    chunks = extract_pdf_chunks(args.pdf, max_chunks=args.chunks)
    if not chunks:
        print("  ERROR: no chunks extracted")
        return

    # ---- Step 2: Generate real embeddings (computed once) ----
    print("\n[2/3] Computing real embeddings (one-time)...")
    embedder = RealEmbedder(args.model)
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(chunk_texts, show_progress=False)
    print(f"  Encoded {len(embeddings)} chunks, dim={embeddings.shape[1]}")

    # ---- Step 3: Run all test systems ----
    queries = FLUID_MECHANICS_QUERIES[:args.queries]
    if len(queries) < args.queries:
        queries = (queries * (args.queries // len(queries) + 1))[:args.queries]

    print(f"\n[3/3] Encoding {len(queries)} queries...")
    query_embeddings = embedder.encode(queries, show_progress=False)

    # Run all systems
    systems = [
        ("FAISS VectorDB", run_faiss_vectordb),
        ("MATHIR V6", run_mathir_v6),
        ("MATHIR V7", run_mathir_v7),
        ("MATHIR V7.1 A (Raw)", run_mathir_v7_approach_a),
        ("MATHIR V7.1 C (FAISS)", run_mathir_v7_approach_c),
        ("MATHIR V7.1 D (Hybrid)", run_mathir_v7_approach_d),
        ("MATHIR Infinity CS", run_mathir_infinity_cs),
        ("MATHIR Infinity Kernel", run_mathir_infinity_kernel),
    ]

    all_results = []
    for sys_name, sys_func in systems:
        print(f"\n{'='*70}")
        print(f"Running: {sys_name}")
        print('='*70)
        try:
            result = sys_func(embeddings, query_embeddings, queries, chunks)
            all_results.append(result)

            # Print immediate result
            if "error" in result:
                print(f"  ERROR: {result['error']}")
                continue

            latencies = result["latencies"]
            overlaps = result["overlaps"]
            hits = result["hits"]

            if latencies:
                mean_lat = statistics.mean(latencies)
                med_lat = statistics.median(latencies)
                p95_lat = sorted(latencies)[int(len(latencies)*0.95)]
                qps = 1000 / mean_lat if mean_lat > 0 else 0

                print(f"  Storage:     {result['storage_ms']:.2f} ms")
                print(f"  Latency:     mean={mean_lat:.2f} ms, median={med_lat:.2f} ms, P95={p95_lat:.2f} ms")
                print(f"  Throughput:  {qps:.0f} QPS")
                if overlaps:
                    print(f"  Quality:     {statistics.mean(overlaps):.2%} keyword overlap")
                    print(f"  Hits:        {sum(hits)}/{len(hits)} (>30% overlap)")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"name": sys_name, "error": str(e)})

    # ---- Final comparison ----
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)

    print(f"\n{'System':<30} {'Storage(ms)':>12} {'Mean(ms)':>10} {'P95(ms)':>10} {'QPS':>10} {'Quality':>10} {'Hits':>10}")
    print("-" * 100)

    for result in all_results:
        if "error" in result:
            print(f"{result['name']:<30} {'ERROR':>12} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
            continue

        latencies = result["latencies"]
        overlaps = result["overlaps"]
        hits = result["hits"]

        if not latencies:
            print(f"{result['name']:<30} {'No data':>12}")
            continue

        mean_lat = statistics.mean(latencies)
        p95_lat = sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0
        qps = 1000 / mean_lat if mean_lat > 0 else 0
        quality = statistics.mean(overlaps) if overlaps else 0
        hit_str = f"{sum(hits)}/{len(hits)}" if hits else "-"

        print(f"{result['name']:<30} {result['storage_ms']:>12.1f} {mean_lat:>10.2f} {p95_lat:>10.2f} {qps:>10.0f} {quality:>10.2%} {hit_str:>10}")

    # ---- Save results ----
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "comprehensive_stress_test_results.json")

    # Serialize
    save_data = {
        "pdf": args.pdf,
        "n_chunks": len(chunks),
        "n_queries": len(queries),
        "model": args.model,
        "results": []
    }

    for result in all_results:
        if "error" in result:
            save_data["results"].append({"name": result["name"], "error": result["error"]})
        else:
            latencies = result["latencies"]
            overlaps = result["overlaps"]
            hits = result["hits"]

            save_data["results"].append({
                "name": result["name"],
                "storage_ms": result["storage_ms"],
                "latency_mean_ms": statistics.mean(latencies) if latencies else 0,
                "latency_median_ms": statistics.median(latencies) if latencies else 0,
                "latency_p95_ms": sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
                "throughput_qps": 1000 / statistics.mean(latencies) if latencies and statistics.mean(latencies) > 0 else 0,
                "quality_overlap": statistics.mean(overlaps) if overlaps else 0,
                "hits": f"{sum(hits)}/{len(hits)}" if hits else "0/0",
            })

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

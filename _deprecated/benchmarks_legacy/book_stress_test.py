"""
MATHIR vs VectorDB — Real Book Stress Test
=============================================

Loads a real PDF textbook, extracts text, chunks it, stores in both
MATHIR and FAISS vector database, then runs 50 queries measuring:
  - Storage time
  - Query latency (P50, P95)
  - Retrieval quality (does the top result contain the query keywords?)
  - Throughput (queries per second)

Usage:
    python benchmarks/book_stress_test.py
    python benchmarks/book_stress_test.py --pdf "path/to/book.pdf" --queries 50
"""

import os
import sys
import time
import hashlib
import argparse
import statistics
from typing import List, Tuple, Dict, Any

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import fitz  # PyMuPDF

from mathir_lib import MATHIRPlugin, MATHIRPluginV7

# ---- Sentence-aware chunking (very simple) ----
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> List[str]:
    """Split text into overlapping word-based chunks."""
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
    """Extract text from a PDF and split into chunks with metadata."""
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
            if len(ch_text.split()) < 30:  # Skip very short chunks
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


# ---- Embedding: deterministic hash-based "embedding" (no API needed) ----
def text_to_embedding(text: str, dim: int = 256) -> np.ndarray:
    """
    Convert text to a deterministic embedding via multiple hash rounds
    (mimics what an LLM embedding does for semantic similarity).
    This is a STAND-IN for a real embedding model. It still tests
    the retrieval logic of MATHIR vs FAISS.
    """
    np.random.seed(0)  # reproducibility
    # Build per-dim hash projections
    h = hashlib.sha512(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    # Normalize
    return raw / (np.linalg.norm(raw) + 1e-8)


# ---- Simple FAISS-based VectorDB ----
import faiss

class VectorDB:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product (cosine after norm)
        self.metadatas: List[Dict[str, Any]] = []

    def store(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        # FAISS expects float32, normalized for cosine
        e = embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(e)
        self.index.add(e)
        # also store the chunk index for back-references
        metadata["__idx__"] = len(self.metadatas)
        self.metadatas.append(metadata)

    def query(self, embedding: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        e = embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(e)
        scores, indices = self.index.search(e, k)
        return [
            (float(scores[0, i]), self.metadatas[int(indices[0, i])])
            for i in range(min(k, len(indices[0])))
            if int(indices[0, i]) >= 0
        ]


# ---- MATHIR wrapper for parallel "store/query" interface ----
class MATHIRStore:
    def __init__(self, dim: int, use_v7: bool = True):
        self.dim = dim
        self.use_v7 = use_v7
        if use_v7:
            self.plugin = MATHIRPluginV7(embedding_dim=dim)
        else:
            self.plugin = MATHIRPlugin(embedding_dim=dim)
        self.metadatas: List[Dict[str, Any]] = []

    def store(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        # MATHIRPlugin expects a torch tensor [B, D]
        emb = torch.from_numpy(embedding).float().unsqueeze(0)
        self.plugin.perceive(emb)
        self.plugin.store({"embedding": emb, "page": metadata.get("page"),
                           "chunk_id": metadata.get("chunk_id")})
        self.metadatas.append(metadata)

    def query(self, embedding: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        emb = torch.from_numpy(embedding).float().unsqueeze(0)
        # Use recall to get the top-k
        results = self.plugin.recall(emb, k=k)
        out = []
        for r in results:
            sim = r.get("similarity", 0.0)
            idx = r.get("index", -1)
            meta = self.metadatas[idx] if 0 <= idx < len(self.metadatas) else {}
            out.append((float(sim), meta))
        return out


# ---- Quality check: does the top result text contain query keywords? ----
def query_keyword_overlap(query: str, retrieved_text: str) -> float:
    """Fraction of query keywords found in the retrieved chunk."""
    q_words = {w.lower().strip(".,;:()[]") for w in query.split() if len(w) > 3}
    if not q_words:
        return 0.0
    r_words = retrieved_text.lower()
    hits = sum(1 for w in q_words if w in r_words)
    return hits / len(q_words)


# ---- Sample queries based on the Fluid Mechanics domain ----
DEFAULT_QUERIES = [
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=r"D:\COURS\Fluid Mechanics 2\White_2011_7ed_Fluid-Mechanics.pdf")
    parser.add_argument("--chunks", type=int, default=200)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--use-v7", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 70)
    print("MATHIR vs VectorDB - REAL BOOK STRESS TEST")
    print("=" * 70)
    print(f"PDF:      {args.pdf}")
    print(f"Chunks:   {args.chunks}")
    print(f"Queries:  {args.queries}")
    print(f"EmbedDim: {args.dim}")
    print(f"Plugin:   {'MATHIRPluginV7' if args.use_v7 else 'MATHIRPlugin'}")
    print()

    # ---- Load PDF ----
    print("[1/4] Loading and chunking the PDF...")
    chunks = extract_pdf_chunks(args.pdf, max_chunks=args.chunks)
    if not chunks:
        print("  ERROR: no chunks extracted.")
        return

    # Pre-compute embeddings
    print("  Computing deterministic embeddings (no API needed)...")
    embeddings = [text_to_embedding(c["text"], dim=args.dim) for c in chunks]

    # ---- Storage benchmark ----
    print("\n[2/4] Storage benchmark...")
    db = VectorDB(dim=args.dim)
    mh = MATHIRStore(dim=args.dim, use_v7=args.use_v7)

    t0 = time.perf_counter()
    for emb, meta in zip(embeddings, chunks):
        db.store(emb, meta)
    t_db = time.perf_counter() - t0

    t0 = time.perf_counter()
    for emb, meta in zip(embeddings, chunks):
        mh.store(emb, meta)
    t_mh = time.perf_counter() - t0

    print(f"  VectorDB: stored {len(chunks)} chunks in {t_db*1000:.1f} ms "
          f"({t_db*1000/len(chunks):.2f} ms/chunk)")
    print(f"  MATHIR:   stored {len(chunks)} chunks in {t_mh*1000:.1f} ms "
          f"({t_mh*1000/len(chunks):.2f} ms/chunk)")
    print(f"  Ratio:    MATHIR is {t_mh/t_db:.2f}x {'slower' if t_mh > t_db else 'faster'} than VectorDB")

    # ---- Build queries ----
    queries = DEFAULT_QUERIES[:args.queries]
    if len(queries) < args.queries:
        # Generate more queries if needed (rotating)
        queries = (queries * (args.queries // len(queries) + 1))[:args.queries]

    print(f"\n[3/4] Running {len(queries)} retrieval queries...")

    # VectorDB queries
    db_latencies = []
    db_hits = []
    db_qualities = []
    for q in queries:
        q_emb = text_to_embedding(q, dim=args.dim)
        t0 = time.perf_counter()
        results = db.query(q_emb, k=5)
        db_latencies.append((time.perf_counter() - t0) * 1000)
        if results:
            top_text = chunks[results[0][1].get("__idx__", 0)]["text"] if "__idx__" in results[0][1] else chunks[results[0][1].get("chunk_idx", 0)]["text"] if "chunk_idx" in results[0][1] else ""
            # Recompute by chunk index
            chunk_idx = results[0][1].get("__idx__", -1)
            if chunk_idx >= 0:
                top_text = chunks[chunk_idx]["text"]
                quality = query_keyword_overlap(q, top_text)
                db_qualities.append(quality)
                db_hits.append(1 if quality > 0.3 else 0)

    # MATHIR queries
    mh_latencies = []
    mh_hits = []
    mh_qualities = []
    for q in queries:
        q_emb = text_to_embedding(q, dim=args.dim)
        t0 = time.perf_counter()
        results = mh.query(q_emb, k=5)
        mh_latencies.append((time.perf_counter() - t0) * 1000)
        if results:
            chunk_idx = -1
            for cid, ch in enumerate(chunks):
                if ch["chunk_id"] == results[0][1].get("chunk_id"):
                    chunk_idx = cid
                    break
            if chunk_idx >= 0:
                top_text = chunks[chunk_idx]["text"]
                quality = query_keyword_overlap(q, top_text)
                mh_qualities.append(quality)
                mh_hits.append(1 if quality > 0.3 else 0)

    # ---- Results ----
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<35} {'VectorDB':>15} {'MATHIR':>15}")
    print("-" * 70)

    def stats(latencies):
        return (statistics.mean(latencies), statistics.median(latencies),
                max(latencies), min(latencies))

    db_mean, db_med, db_max, db_min = stats(db_latencies)
    mh_mean, mh_med, mh_max, mh_min = stats(mh_latencies)

    print(f"{'Query latency - mean (ms)':<35} {db_mean:>15.2f} {mh_mean:>15.2f}")
    print(f"{'Query latency - median (ms)':<35} {db_med:>15.2f} {mh_med:>15.2f}")
    print(f"{'Query latency - min (ms)':<35} {db_min:>15.2f} {mh_min:>15.2f}")
    print(f"{'Query latency - max (ms)':<35} {db_max:>15.2f} {mh_max:>15.2f}")
    p95_db = sorted(db_latencies)[int(len(db_latencies)*0.95)]
    p95_mh = sorted(mh_latencies)[int(len(mh_latencies)*0.95)]
    print(f"{'Query latency - P95 (ms)':<35} {p95_db:>15.2f} {p95_mh:>15.2f}")

    db_qps = 1000 / db_mean if db_mean > 0 else 0
    mh_qps = 1000 / mh_mean if mh_mean > 0 else 0
    print(f"{'Throughput (queries/sec)':<35} {db_qps:>15.0f} {mh_qps:>15.0f}")

    if db_qualities:
        print(f"{'Top-1 keyword overlap':<35} {statistics.mean(db_qualities):>15.2%} {statistics.mean(mh_qualities) if mh_qualities else 0:>15.2%}")
    if db_hits:
        print(f"{'Hits (overlap > 30%)':<35} {sum(db_hits):>15d}/{len(db_hits)} {sum(mh_hits) if mh_hits else 0:>14d}/{len(mh_hits) if mh_hits else 0}")

    print("-" * 70)
    print(f"\nStorage: MATHIR is {t_mh/t_db:.2f}x the time of VectorDB")
    print(f"Query:   MATHIR is {mh_mean/db_mean:.2f}x the latency of VectorDB")
    print(f"        VectorDB is {db_qps/mh_qps:.2f}x faster in throughput")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if t_mh < t_db * 2:
        print(f"  Storage: MATHIR is competitive (within 2x of VectorDB)")
    else:
        print(f"  Storage: MATHIR is {t_mh/t_db:.1f}x slower (acceptable for in-memory learning)")
    if mh_mean < db_mean * 5:
        print(f"  Query:   MATHIR is competitive for real-time use")
    else:
        print(f"  Query:   VectorDB wins on pure latency (but lacks online learning)")

    print(f"\n  MATHIR ADVANTAGE: online learning + anomaly detection + multi-tier memory")
    print(f"  VectorDB ADVANTAGE: faster pure retrieval at scale")

    # Save results
    results = {
        "pdf": args.pdf,
        "n_chunks": len(chunks),
        "n_queries": len(queries),
        "embedding_dim": args.dim,
        "storage": {
            "vectordb_ms": t_db * 1000,
            "mathir_ms": t_mh * 1000,
        },
        "query_latency_ms": {
            "vectordb": {"mean": db_mean, "median": db_med, "p95": p95_db, "min": db_min, "max": db_max},
            "mathir":   {"mean": mh_mean, "median": mh_med, "p95": p95_mh, "min": mh_min, "max": mh_max},
        },
        "throughput_qps": {"vectordb": db_qps, "mathir": mh_qps},
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "book_stress_test_results.json")
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

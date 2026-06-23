"""
Approach D (Hybrid BM25+Cross-Encoder) vs Real Vector Database (FAISS)
========================================================================

The decisive stress test: 200 chunks from White's Fluid Mechanics,
50 domain-specific queries, real sentence-transformer embeddings.

Compares:
  - FAISS IndexFlatIP (raw 384-dim cosine) — the "real vector database"
  - MATHIR + Approach D (BM25 + Dense + Cross-Encoder Re-Rank)

Reports storage, query latency, throughput, and quality (top-1 keyword overlap).

Usage:
    python benchmarks/approach_d_vs_faiss.py
    python benchmarks/approach_d_vs_faiss.py --chunks 300 --queries 50
"""

import os
import sys
import time
import json
import argparse
import statistics
import warnings
from typing import List, Tuple, Dict, Any

warnings.filterwarnings("ignore")

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import fitz

# Force UTF-8 output (Windows console)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---- Real embedding model ----
class RealEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"  Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()
        print(f"  Embedding dim: {self.dim}")

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=32, show_progress_bar=show_progress,
            convert_to_numpy=True, normalize_embeddings=True,
        )


# ---- PDF chunking ----
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


def extract_pdf_chunks(pdf_path: str, max_chunks: int = 200) -> List[Dict[str, Any]]:
    print(f"  Loading PDF: {os.path.basename(pdf_path)}")
    doc = fitz.open(pdf_path)
    print(f"  Pages: {len(doc)}")
    chunks = []
    for page_idx, page in enumerate(doc):
        text = page.get_text("text")
        if not text.strip():
            continue
        for ch_idx, ch_text in enumerate(chunk_text(text, chunk_size=150, overlap=20)):
            if len(ch_text.split()) < 30:
                continue
            chunks.append({
                "text": ch_text, "page": page_idx + 1,
                "chunk_id": f"p{page_idx+1:04d}_c{ch_idx:03d}",
                "n_words": len(ch_text.split()),
            })
        if len(chunks) >= max_chunks:
            break
    doc.close()
    print(f"  Extracted {len(chunks)} chunks (avg {statistics.mean(c['n_words'] for c in chunks):.0f} words/chunk)")
    return chunks[:max_chunks]


# ---- FAISS Vector Database ----
import faiss

class FAISSVectorDB:
    """Real vector database using FAISS IndexFlatIP (exact, no compression)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas: List[Dict[str, Any]] = []

    def store_batch(self, embs: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        for e, m in zip(embs, metas):
            m["__idx__"] = len(self.metadatas)
            self.metadatas.append(m)
        self.index.add(embs.astype("float32"))

    def query(self, emb: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        e = emb.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(e, k)
        return [
            (float(scores[0, i]), self.metadatas[int(indices[0, i])])
            for i in range(min(k, len(indices[0])))
            if int(indices[0, i]) >= 0
        ]


# ---- Approach D: Hybrid BM25 + Dense + Cross-Encoder ----
class ApproachD:
    """The hybrid retriever: dense cosine + BM25 + RRF + cross-encoder re-rank."""

    def __init__(self, dim: int):
        from mathir_lib.memory.hybrid_episodic import HybridEpisodicMemory
        self.dim = dim
        self.memory = HybridEpisodicMemory(capacity=10000, feature_dim=dim, use_cross_encoder=True)
        self.metadatas: List[Dict[str, Any]] = []

    def store_batch(self, embs: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        for e, m in zip(embs, metas):
            t = torch.from_numpy(e).float().unsqueeze(0)
            self.memory.store(t, text=m["text"])
            self.metadatas.append(m)

    def query(self, emb: np.ndarray, k: int = 5, query_text: str = "") -> List[Tuple[float, Dict[str, Any]]]:
        t = torch.from_numpy(emb).float().unsqueeze(0)
        # Use search() with query_text so BM25 and cross-encoder activate
        indices, sims = self.memory.search(t, k=k, query_text=query_text)
        out = []
        for j in range(indices.size(1)):
            idx = int(indices[0, j].item())
            sim = float(sims[0, j].item())
            meta = self.metadatas[idx] if 0 <= idx < len(self.metadatas) else {}
            out.append((sim, meta))
        return out


# ---- Quality metric: keyword overlap ----
def keyword_overlap(query: str, retrieved_text: str) -> float:
    q_words = {w.lower().strip(".,;:()[]\"'") for w in query.split() if len(w) > 3}
    stopwords = {"what", "when", "where", "which", "have", "does", "with",
                 "from", "this", "that", "are", "was", "were", "been", "how",
                 "explain", "define", "difference", "between"}
    q_words = q_words - stopwords
    if not q_words:
        return 0.0
    r_text = retrieved_text.lower()
    hits = sum(1 for w in q_words if w in r_text)
    return hits / len(q_words)


def semantic_similarity_score(query: str, retrieved_text: str) -> float:
    """Loose semantic match: does the retrieved text discuss the query topic?"""
    q_lower = query.lower()
    r_lower = retrieved_text.lower()
    # Get key technical terms
    q_words = {w.lower().strip(".,;:()[]\"'") for w in query.split() if len(w) > 4}
    q_words = {w for w in q_words if w not in {"what", "when", "where", "which", "does", "explain", "define", "calculate", "compute"}}
    if not q_words:
        return 0.0
    # Check if at least 2 key terms appear or if any long technical term appears
    hits = sum(1 for w in q_words if w in r_lower)
    return min(1.0, hits / max(2, len(q_words) // 2))


# ---- Domain queries (Fluid Mechanics) ----
QUERIES = [
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
    args = parser.parse_args()

    print("=" * 70)
    print("APPROACH D (Hybrid BM25+CE) vs FAISS VECTOR DATABASE")
    print("Real Book Stress Test — 50 queries on White's Fluid Mechanics")
    print("=" * 70)
    print(f"PDF:      {os.path.basename(args.pdf)}")
    print(f"Chunks:   {args.chunks}")
    print(f"Queries:  {args.queries}")
    print()

    # ---- Load PDF + embed ----
    print("[1/4] Loading PDF and embedding chunks with real model...")
    chunks = extract_pdf_chunks(args.pdf, max_chunks=args.chunks)
    embedder = RealEmbedder()
    chunk_embs = embedder.encode([c["text"] for c in chunks], show_progress=False)
    print(f"  Encoded {len(chunk_embs)} chunks in {embedder.dim}-dim")

    queries = (QUERIES * (args.queries // len(QUERIES) + 1))[:args.queries]
    print(f"  Encoding {len(queries)} queries...")
    query_embs = embedder.encode(queries, show_progress=False)

    # ---- Build both systems ----
    print("\n[2/4] Building both systems...")
    faiss_db = FAISSVectorDB(dim=embedder.dim)
    approach_d = ApproachD(dim=embedder.dim)
    print(f"  FAISS VectorDB:        IndexFlatIP ({embedder.dim}-dim, exact cosine)")
    print(f"  Approach D (Hybrid):   Dense + BM25 + RRF + cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ---- Storage benchmark ----
    print("\n[3/4] Storage benchmark...")

    t0 = time.perf_counter()
    faiss_db.store_batch(chunk_embs, chunks)
    t_faiss = time.perf_counter() - t0

    t0 = time.perf_counter()
    approach_d.store_batch(chunk_embs, chunks)
    t_d = time.perf_counter() - t0

    print(f"  FAISS:    stored {len(chunks)} chunks in {t_faiss*1000:.1f} ms ({t_faiss*1000/len(chunks):.3f} ms/chunk)")
    print(f"  Approach D: stored {len(chunks)} chunks in {t_d*1000:.1f} ms ({t_d*1000/len(chunks):.3f} ms/chunk)")
    print(f"  Ratio:    Approach D is {t_d/max(t_faiss, 1e-6):.1f}x slower (one-time setup)")

    # ---- Query benchmark ----
    print(f"\n[4/4] Running {len(queries)} retrieval queries...")

    def run_query_loop(system, name, use_query_text=False):
        latencies = []
        overlaps = []
        sem_scores = []
        hits_30 = []
        hits_50 = []
        top1_pages = []
        for q_emb, q in zip(query_embs, queries):
            t0 = time.perf_counter()
            if use_query_text:
                results = system.query(q_emb, k=5, query_text=q)
            else:
                results = system.query(q_emb, k=5)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            if results:
                top_meta = results[0][1]
                if "__idx__" in top_meta:
                    idx = top_meta["__idx__"]
                    if 0 <= idx < len(chunks):
                        top_text = chunks[idx]["text"]
                        top1_pages.append(chunks[idx]["page"])
                elif "chunk_id" in top_meta:
                    for ci, c in enumerate(chunks):
                        if c["chunk_id"] == top_meta["chunk_id"]:
                            top_text = c["text"]
                            top1_pages.append(c["page"])
                            break
                    else:
                        top_text = ""
                else:
                    top_text = ""
                if top_text:
                    ov = keyword_overlap(q, top_text)
                    sem = semantic_similarity_score(q, top_text)
                    overlaps.append(ov)
                    sem_scores.append(sem)
                    hits_30.append(1 if ov >= 0.3 else 0)
                    hits_50.append(1 if ov >= 0.5 else 0)
        return {
            "latencies": latencies, "overlaps": overlaps, "sem_scores": sem_scores,
            "hits_30": hits_30, "hits_50": hits_50, "top1_pages": top1_pages,
        }

    print("  Running FAISS queries...")
    faiss_res = run_query_loop(faiss_db, "FAISS", use_query_text=False)
    print(f"    Done. Mean: {statistics.mean(faiss_res['latencies']):.2f} ms")

    print("  Running Approach D queries (with cross-encoder re-rank)...")
    d_res = run_query_loop(approach_d, "Approach D", use_query_text=True)
    print(f"    Done. Mean: {statistics.mean(d_res['latencies']):.2f} ms")

    # ---- Display results ----
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<40} {'FAISS':>15} {'Approach D':>15}")
    print("-" * 75)

    # Storage
    print(f"{'Storage total (ms)':<40} {t_faiss*1000:>15.1f} {t_d*1000:>15.1f}")
    print(f"{'Storage per chunk (ms)':<40} {t_faiss*1000/len(chunks):>15.3f} {t_d*1000/len(chunks):>15.3f}")

    # Latency
    f_mean, d_mean = statistics.mean(faiss_res['latencies']), statistics.mean(d_res['latencies'])
    f_med, d_med = statistics.median(faiss_res['latencies']), statistics.median(d_res['latencies'])
    f_p95 = sorted(faiss_res['latencies'])[int(len(faiss_res['latencies']) * 0.95)]
    d_p95 = sorted(d_res['latencies'])[int(len(d_res['latencies']) * 0.95)]
    f_min, d_min = min(faiss_res['latencies']), min(d_res['latencies'])
    f_max, d_max = max(faiss_res['latencies']), max(d_res['latencies'])
    print(f"{'Query latency - mean (ms)':<40} {f_mean:>15.2f} {d_mean:>15.2f}")
    print(f"{'Query latency - median (ms)':<40} {f_med:>15.2f} {d_med:>15.2f}")
    print(f"{'Query latency - P95 (ms)':<40} {f_p95:>15.2f} {d_p95:>15.2f}")
    print(f"{'Query latency - min (ms)':<40} {f_min:>15.2f} {d_min:>15.2f}")
    print(f"{'Query latency - max (ms)':<40} {f_max:>15.2f} {d_max:>15.2f}")

    # Throughput
    f_qps = 1000 / f_mean if f_mean > 0 else 0
    d_qps = 1000 / d_mean if d_mean > 0 else 0
    print(f"{'Throughput (queries/sec)':<40} {f_qps:>15.0f} {d_qps:>15.0f}")

    # Quality
    f_overlap = statistics.mean(faiss_res['overlaps']) if faiss_res['overlaps'] else 0
    d_overlap = statistics.mean(d_res['overlaps']) if d_res['overlaps'] else 0
    f_sem = statistics.mean(faiss_res['sem_scores']) if faiss_res['sem_scores'] else 0
    d_sem = statistics.mean(d_res['sem_scores']) if d_res['sem_scores'] else 0
    f_h30 = sum(faiss_res['hits_30'])
    d_h30 = sum(d_res['hits_30'])
    f_h50 = sum(faiss_res['hits_50'])
    d_h50 = sum(d_res['hits_50'])
    print(f"{'Top-1 keyword overlap':<40} {f_overlap*100:>14.1f}% {d_overlap*100:>14.1f}%")
    print(f"{'Top-1 semantic match':<40} {f_sem*100:>14.1f}% {d_sem*100:>14.1f}%")
    print(f"{'Relevant hits (overlap >= 30%)':<40} {f_h30:>14d}/{len(queries)} {d_h30:>14d}/{len(queries)}")
    print(f"{'Strong hits (overlap >= 50%)':<40} {f_h50:>14d}/{len(queries)} {d_h50:>14d}/{len(queries)}")
    print("-" * 75)

    # ---- Sample comparisons ----
    print("\n" + "=" * 70)
    print("SAMPLE QUERIES — Side by side")
    print("=" * 70)
    for i in [0, 5, 10, 20, 30]:
        if i >= len(queries):
            break
        q = queries[i]
        q_emb = query_embs[i]
        print(f"\nQ{i+1}: {q}")

        # FAISS
        f_res = faiss_db.query(q_emb, k=1)
        if f_res:
            idx = f_res[0][1].get("__idx__", -1)
            if 0 <= idx < len(chunks):
                text = chunks[idx]["text"][:160].replace("\n", " ").replace("\x0c", " ")
                print(f"  FAISS  (p.{chunks[idx]['page']}, sim={f_res[0][0]:.3f}): \"{text}...\"")

        # Approach D
        d_results = approach_d.query(q_emb, k=1, query_text=q)
        if d_results:
            for ci, c in enumerate(chunks):
                if c["chunk_id"] == d_results[0][1].get("chunk_id"):
                    text = c["text"][:160].replace("\n", " ").replace("\x0c", " ")
                    print(f"  App-D  (p.{c['page']}, sim={d_results[0][0]:.3f}): \"{text}...\"")
                    break

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("VERDICT — Approach D vs FAISS Vector Database")
    print("=" * 70)
    print(f"\n  QUALITY:    Approach D {'WINS' if d_overlap > f_overlap else 'LOSES'} "
          f"(FAISS={f_overlap*100:.1f}%, Approach D={d_overlap*100:.1f}%, "
          f"delta={(d_overlap - f_overlap)*100:+.1f}%)")
    print(f"  SEMANTIC:   Approach D {'WINS' if d_sem > f_sem else 'LOSES'} "
          f"(FAISS={f_sem*100:.1f}%, Approach D={d_sem*100:.1f}%)")
    print(f"  SPEED:      Approach D is {d_mean/f_mean:.1f}x SLOWER "
          f"({f_mean:.1f}ms vs {d_mean:.1f}ms per query)")
    print(f"  THROUGHPUT: FAISS is {f_qps/d_qps:.0f}x faster ({f_qps:.0f} vs {d_qps:.0f} QPS)")

    if d_overlap > f_overlap:
        print(f"\n  >> Approach D wins on quality by {(d_overlap - f_overlap)*100:.1f} percentage points")
        print(f"  >> Cost: {d_mean/f_mean:.0f}x slower per query (one-time re-rank cost)")
        print(f"  >> Use case: Batch processing, offline re-ranking, technical document Q&A")
    else:
        print(f"\n  >> FAISS wins overall (faster AND comparable quality)")

    # Practical implication
    print(f"\n  PRACTICAL:")
    print(f"    - 50 queries to FAISS:    {50 * f_mean:.0f} ms total")
    print(f"    - 50 queries to Approach D: {50 * d_mean:.0f} ms total ({50 * d_mean/1000:.1f} sec)")
    print(f"    - LLM call (typical):      ~2000 ms per call")
    print(f"    - Approach D overhead vs LLM: {d_mean/2000*100:.1f}% of LLM time")

    # Save
    out = {
        "pdf": args.pdf, "n_chunks": len(chunks), "n_queries": len(queries),
        "embedding_dim": embedder.dim,
        "storage_ms": {"faiss": t_faiss*1000, "approach_d": t_d*1000},
        "query_latency_ms": {
            "faiss": {"mean": f_mean, "median": f_med, "p95": f_p95, "min": f_min, "max": f_max},
            "approach_d": {"mean": d_mean, "median": d_med, "p95": d_p95, "min": d_min, "max": d_max},
        },
        "throughput_qps": {"faiss": f_qps, "approach_d": d_qps},
        "quality": {
            "faiss": {"overlap_mean": f_overlap, "semantic_mean": f_sem,
                      "hits_30pct": f_h30, "hits_50pct": f_h50},
            "approach_d": {"overlap_mean": d_overlap, "semantic_mean": d_sem,
                          "hits_30pct": d_h30, "hits_50pct": d_h50},
        },
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "approach_d_vs_faiss_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()

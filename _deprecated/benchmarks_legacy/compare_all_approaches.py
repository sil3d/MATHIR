"""
MATHIR — Master Comparison Test
=================================

Compares ALL retrieval approaches side by side on the same real-book test:
  - Baseline: FAISS VectorDB (raw 384-dim cosine)
  - Baseline: MATHIR V7 (default, with 64-dim projection)
  - Approach A: MATHIR with raw embedding bypass
  - Approach B: MATHIR with multi-encoder ensemble
  - Approach C: MATHIR with FAISS-backed index

Measures: storage time, query latency, throughput, quality (keyword overlap).

Usage:
    python benchmarks/compare_all_approaches.py --chunks 200 --queries 50
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

# We'll import the approaches once they're built
def safe_import(name, path):
    """Import a module only if it exists."""
    import importlib.util
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"  Failed to import {name}: {e}")
        return None


# ---------- Embedder ----------
class RealEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=32, show_progress_bar=show_progress,
            convert_to_numpy=True, normalize_embeddings=True,
        )


# ---------- Chunking ----------
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
    print(f"  Extracted {len(chunks)} chunks")
    return chunks[:max_chunks]


# ---------- FAISS VectorDB ----------
import faiss

class FAISSVectorDB:
    name = "FAISS VectorDB (raw 384-dim)"

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas: List[Dict[str, Any]] = []

    def store(self, emb: np.ndarray, meta: Dict[str, Any]) -> None:
        meta["__idx__"] = len(self.metadatas)
        self.metadatas.append(meta)
        self.index.add(emb.astype("float32").reshape(1, -1))

    def store_batch(self, embs: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        for e, m in zip(embs, metas):
            self.store(e, m)

    def query(self, emb: np.ndarray, k: int = 5):
        e = emb.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(e, k)
        return [
            (float(scores[0, i]), self.metadatas[int(indices[0, i])])
            for i in range(min(k, len(indices[0])))
            if int(indices[0, i]) >= 0
        ]


# ---------- MATHIR V7 (default) ----------
class MATHIRV7:
    name = "MATHIR V7 (default, 64-dim projection)"

    def __init__(self, dim: int):
        from mathir_lib import MATHIRPluginV7
        self.plugin = MATHIRPluginV7(embedding_dim=dim)
        self.metadatas: List[Dict[str, Any]] = []

    def store(self, emb: np.ndarray, meta: Dict[str, Any]) -> None:
        t = torch.from_numpy(emb).float().unsqueeze(0)
        self.plugin.perceive(t)
        self.plugin.store({"embedding": t, "page": meta.get("page"),
                           "chunk_id": meta.get("chunk_id")})
        self.metadatas.append(meta)

    def store_batch(self, embs: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        for e, m in zip(embs, metas):
            self.store(e, m)

    def query(self, emb: np.ndarray, k: int = 5):
        t = torch.from_numpy(emb).float().unsqueeze(0)
        results = self.plugin.recall(t, k=k)
        out = []
        for r in results:
            sim = r.get("similarity", 0.0)
            idx = r.get("index", -1)
            meta = self.metadatas[idx] if 0 <= idx < len(self.metadatas) else {}
            out.append((float(sim), meta))
        return out


# ---------- Quality ----------
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


# ---------- Queries ----------
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


def evaluate_system(name: str, system, queries, query_embeddings, chunks, top_k=5) -> Dict[str, Any]:
    """Run a system against queries and return metrics."""
    print(f"\n  Evaluating: {name}")

    # Use embeddings stored in chunks
    embs = np.array([c["__emb"] for c in chunks])
    metas = [{k: v for k, v in c.items() if k != "__emb"} for c in chunks]
    t0 = time.perf_counter()
    system.store_batch(embs, metas)
    t_store = time.perf_counter() - t0

    # Query
    latencies = []
    overlaps = []
    hits = []
    for q_emb, q in zip(query_embeddings, queries):
        t0 = time.perf_counter()
        # Pass query_text to systems that support it (hybrid)
        if hasattr(system, 'query') and 'query_text' in system.query.__code__.co_varnames:
            results = system.query(q_emb, k=top_k, query_text=q)
        else:
            results = system.query(q_emb, k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        if results:
            top_meta = results[0][1]
            if "__idx__" in top_meta:
                idx = top_meta["__idx__"]
            elif "chunk_id" in top_meta:
                # Find by chunk_id
                idx = -1
                for ci, c in enumerate(chunks):
                    if c["chunk_id"] == top_meta["chunk_id"]:
                        idx = ci
                        break
            else:
                idx = -1
            if 0 <= idx < len(chunks):
                overlap = keyword_overlap(q, chunks[idx]["text"])
                overlaps.append(overlap)
                hits.append(1 if overlap >= 0.3 else 0)

    return {
        "name": name,
        "storage_ms": t_store * 1000,
        "storage_per_chunk_ms": (t_store * 1000) / len(chunks),
        "query_mean_ms": statistics.mean(latencies),
        "query_median_ms": statistics.median(latencies),
        "query_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "query_min_ms": min(latencies),
        "query_max_ms": max(latencies),
        "throughput_qps": 1000 / statistics.mean(latencies),
        "overlap_mean": statistics.mean(overlaps) if overlaps else 0,
        "hits": sum(hits),
        "n_queries": len(queries),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=r"D:\COURS\Fluid Mechanics 2\White_2011_7ed_Fluid-Mechanics.pdf")
    parser.add_argument("--chunks", type=int, default=200)
    parser.add_argument("--queries", type=int, default=50)
    args = parser.parse_args()

    print("=" * 70)
    print("MATHIR — MASTER COMPARISON: ALL APPROACHES")
    print("=" * 70)

    # Load PDF
    print("\n[1/4] Loading PDF and embedding...")
    chunks = extract_pdf_chunks(args.pdf, max_chunks=args.chunks)
    embedder = RealEmbedder()
    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = embedder.encode(chunk_texts, show_progress=False)
    for i, c in enumerate(chunks):
        c["__emb"] = chunk_embs[i]
    print(f"  Embedding dim: {embedder.dim}, chunks: {len(chunks)}")

    # Queries
    queries = (DEFAULT_QUERIES * (args.queries // len(DEFAULT_QUERIES) + 1))[:args.queries]
    print(f"  Encoding {len(queries)} queries...")
    query_embs = embedder.encode(queries, show_progress=False)

    # ---- Run all systems ----
    systems = []

    print("\n[2/4] Building all systems...")
    systems.append(FAISSVectorDB(dim=embedder.dim))
    systems.append(MATHIRV7(dim=embedder.dim))

    # Try to import the new approaches
    for approach_name, mod_path, class_name in [
        ("MATHIR + Raw Embedding (A)", "mathir_lib/memory/raw_episodic.py", "RawEmbeddingEpisodicMemory"),
        ("MATHIR + Multi-Encoder (B)", "mathir_lib/memory/ensemble_episodic.py", "EnsembleEpisodicMemory"),
        ("MATHIR + FAISS-backed (C)", "mathir_lib/memory/faiss_episodic.py", "FAISSBackedEpisodicMemory"),
        ("MATHIR + Hybrid BM25+CE (D)", "mathir_lib/memory/hybrid_episodic.py", "HybridEpisodicMemory"),
    ]:
        full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), mod_path)
        if not os.path.exists(full_path):
            print(f"  SKIP: {approach_name} (file not found)")
            continue
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"approach_{class_name}", full_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = getattr(mod, class_name)
            # Try the constructor with the right argument name
            try:
                instance = cls(capacity=args.chunks, feature_dim=embedder.dim)
            except TypeError:
                try:
                    instance = cls(capacity=args.chunks, embedding_dim=embedder.dim)
                except TypeError:
                    import inspect
                    sig = inspect.signature(cls.__init__)
                    kwargs = {}
                    for p_name, p in sig.parameters.items():
                        if "dim" in p_name.lower() and p_name != "self":
                            kwargs[p_name] = embedder.dim
                        elif p_name == "capacity":
                            kwargs[p_name] = args.chunks
                    instance = cls(**kwargs)
            instance.name = approach_name

            # Adapter to expose store_batch/query interface
            class Adapter:
                def __init__(self, inner, name):
                    self.inner = inner
                    self.name = name
                    self.metadatas = []
                def store(self, emb, meta):
                    # If this is the hybrid, pass text too
                    if hasattr(self.inner, 'store') and 'text' in self.inner.store.__code__.co_varnames:
                        self.inner.store(torch.from_numpy(emb).float().unsqueeze(0), text=meta.get("text", ""))
                    else:
                        self.inner.store(torch.from_numpy(emb).float().unsqueeze(0))
                    self.metadatas.append(meta)
                def store_batch(self, embs, metas):
                    for e, m in zip(embs, metas):
                        self.store(e, m)
                def query(self, emb, k=5, query_text=None):
                    t = torch.from_numpy(emb).float().unsqueeze(0)
                    # Use search() if available (returns indices, similarities)
                    if hasattr(self.inner, 'search'):
                        # Some search methods accept query_text for hybrid
                        try:
                            if query_text and 'query_text' in self.inner.search.__code__.co_varnames:
                                indices, sims = self.inner.search(t, k=k, query_text=query_text)
                            else:
                                indices, sims = self.inner.search(t, k=k)
                        except TypeError:
                            indices, sims = self.inner.search(t, k=k)
                        out = []
                        for j in range(indices.size(1) if indices.dim() > 1 else 1):
                            idx = int(indices[0, j].item()) if indices.dim() > 1 else int(indices[j].item())
                            sim = float(sims[0, j].item()) if sims.dim() > 1 else float(sims[j].item())
                            meta = self.metadatas[idx] if 0 <= idx < len(self.metadatas) else {}
                            out.append((sim, meta))
                        return out
                    else:
                        res = self.inner.retrieve(t, k=k)
                        return [(0.0, {}) for _ in range(k)]

            systems.append(Adapter(instance, approach_name))
            print(f"  Loaded: {approach_name}")
        except Exception as e:
            print(f"  FAILED: {approach_name} - {e}")

    print(f"\n[3/4] Evaluating {len(systems)} systems on {len(queries)} queries...")
    results = []
    for sys in systems:
        try:
            r = evaluate_system(sys.name, sys, queries, query_embs, chunks)
            results.append(r)
        except Exception as e:
            print(f"  ERROR evaluating {sys.name}: {e}")
            import traceback
            traceback.print_exc()

    # ---- Display ----
    print("\n" + "=" * 70)
    print("MASTER COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'System':<42} {'QPS':>8} {'Over':>7} {'Hits':>7} {'Store':>8} {'Q-med':>7}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<42} {r['throughput_qps']:>8.0f} "
              f"{r['overlap_mean']*100:>6.1f}% {r['hits']:>4d}/{r['n_queries']:<2d} "
              f"{r['storage_per_chunk_ms']:>6.2f}ms {r['query_median_ms']:>6.2f}ms")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("VERDICT — Which approach wins?")
    print("=" * 70)
    if results:
        best_quality = max(results, key=lambda r: r["overlap_mean"])
        best_speed = max(results, key=lambda r: r["throughput_qps"])
        print(f"  Best QUALITY:  {best_quality['name']} ({best_quality['overlap_mean']*100:.1f}% overlap)")
        print(f"  Best SPEED:    {best_speed['name']} ({best_speed['throughput_qps']:.0f} QPS)")

        # Check if any new approach beats baseline VectorDB
        baseline = next((r for r in results if "FAISS" in r["name"]), None)
        v7_default = next((r for r in results if "V7" in r["name"]), None)
        if baseline and v7_default:
            for r in results:
                if r["name"] not in [baseline["name"], v7_default["name"]]:
                    quality_delta = (r["overlap_mean"] - v7_default["overlap_mean"]) * 100
                    print(f"  vs V7 default:  {r['name']:<35} quality delta = {quality_delta:+.1f}%")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "compare_all_approaches_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()

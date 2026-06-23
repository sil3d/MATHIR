"""
mathir_optimized.OptimizedMATHIR
================================

Production-grade hybrid retriever designed for BEIR-scale evaluation.

Pipeline
--------
    Query ──┬─► Dense (BGE) ────► FAISS Top-K_d ─┐
            │                                     ├─► RRF fusion ─┬─► Cross-Encoder ─► Top-K
            └─► BM25 (rank_bm25) ──► Top-K_b ─────┘                │
                                                                   └─► (CE disabled) ─► Top-K

Why this design
---------------
* **Dense** (BGE-base-en-v1.5) catches semantic neighbours.
* **BM25** catches exact technical terms that dense misses.
* **RRF** (Cormack et al., 2009) is parameter-free and robust to
  score-scale mismatch — the de-facto standard for hybrid retrieval.
* **Cross-encoder** re-ranks the fused top-N for the highest precision.

The class is **configurable**: any of the 3 branches can be disabled.
The benchmark exercises all 3 configurations:
    - dense-only  (no BM25, no CE)
    - + BM25      (dense + BM25 + RRF, no CE)
    - + CE        (dense + BM25 + RRF + CE rerank)

API
---
    >>> retriever = OptimizedMATHIR()  # uses BGE-base by default
    >>> retriever.index(doc_ids, doc_texts)
    >>> doc_ids, scores = retriever.search("query text", k=10,
    ...                                      query_text="query text")
    >>> stats = retriever.get_stats()

This file is the "new file" referenced in the BEIR v2 benchmark brief.
"""

from __future__ import annotations

import os
import re
import sys
import time
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Defaults
# ============================================================================
DEFAULT_EMBEDDER = "BAAI/bge-base-en-v1.5"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# TinyBERT = ~10x faster than MiniLM-L-6 with ~1-2pp nDCG drop.
TINYBERT_CROSS_ENCODER = "cross-encoder/ms-marco-TinyBERT-L-2-v2"


# ============================================================================
# Tokenizer (cheap, BM25-friendly)
# ============================================================================
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanum tokenizer, keeps hyphenated terms whole."""
    return _TOKEN_RE.findall(text.lower())


# ============================================================================
# OptimizedMATHIR
# ============================================================================
class OptimizedMATHIR:
    """
    Hybrid retriever: Dense (FAISS) + BM25 + RRF + Cross-Encoder.

    See module docstring for the architecture. All three retrieval
    branches are optional (use_bm25, use_cross_encoder) so the class
    can be benchmarked in three configurations.
    """

    def __init__(
        self,
        embedder_name: str = DEFAULT_EMBEDDER,
        use_bm25: bool = True,
        use_cross_encoder: bool = True,
        cross_encoder_name: str = DEFAULT_CROSS_ENCODER,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 3.0,
        dense_top_k: int = 100,
        bm25_top_k: int = 100,
        ce_top_n: int = 50,
        device: str = "cpu",
        # ----- injection points for tests / custom models -----
        embedder=None,
        cross_encoder=None,
    ):
        self.embedder_name = embedder_name
        self.use_bm25 = bool(use_bm25)
        self.use_cross_encoder = bool(use_cross_encoder)
        self.cross_encoder_name = cross_encoder_name
        self.rrf_k = int(rrf_k)
        self.bm25_weight = float(bm25_weight)
        self.dense_weight = float(dense_weight)
        self.dense_top_k = int(dense_top_k)
        self.bm25_top_k = int(bm25_top_k)
        self.ce_top_n = int(ce_top_n)
        self.device = device

        # ----- injected (test mode) or lazy-loaded (prod mode) -----
        self._embedder = embedder
        self._cross_encoder = cross_encoder
        self._cross_encoder_failed = False
        self._cross_encoder_backend = "injected" if cross_encoder is not None else "none"

        # ----- internal index state -----
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []
        self._doc_embeddings: Optional[np.ndarray] = None
        self._faiss_index = None
        self._bm25 = None
        self._bm25_tokens: List[List[str]] = []
        self._indexed: bool = False

        # ----- per-stage latency accumulators (ms) -----
        self._stats: Dict[str, float] = {
            "index_time_ms": 0.0,
            "encode_time_ms": 0.0,
            "search_time_ms": 0.0,
            "rerank_time_ms": 0.0,
            "queries": 0,
            "latencies_ms": [],  # full per-query latencies
        }
        # Caches for the cross-encoder (LRU on (q, d) → score)
        self._ce_cache: "OrderedDict[Tuple[str, str], float]" = OrderedDict()
        self._ce_cache_size: int = 10_000

    # ------------------------------------------------------------------
    # Lazy loaders (private)
    # ------------------------------------------------------------------
    def _load_embedder(self):
        """Load the dense embedder if not injected."""
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer  # type: ignore
        try:
            self._embedder = SentenceTransformer(self.embedder_name, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"[OptimizedMATHIR] Could not load embedder "
                f"'{self.embedder_name}': {exc}"
            ) from exc

    def _load_cross_encoder(self) -> bool:
        """Load the cross-encoder. Returns True on success, False on fallback."""
        if self._cross_encoder is not None:
            return True
        if self._cross_encoder_failed:
            return False
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._cross_encoder = CrossEncoder(self.cross_encoder_name, device=self.device)
            self._cross_encoder_backend = "pytorch"
            return True
        except Exception as exc:
            warnings.warn(
                f"[OptimizedMATHIR] Could not load cross-encoder "
                f"'{self.cross_encoder_name}': {exc}. "
                f"Falling back to RRF-only hybrid retrieval.",
                RuntimeWarning,
            )
            self._cross_encoder_failed = True
            self._cross_encoder_backend = "failed"
            return False

    # ------------------------------------------------------------------
    # Public API: index + search
    # ------------------------------------------------------------------
    def index(self, doc_ids, doc_texts):
        """Encode and index documents. doc_ids and doc_texts are parallel lists."""
        import faiss  # type: ignore

        if len(doc_ids) != len(doc_texts):
            raise ValueError(
                f"doc_ids and doc_texts must have the same length "
                f"(got {len(doc_ids)} vs {len(doc_texts)})"
            )
        t0 = time.perf_counter()

        self._doc_ids = [str(d) for d in doc_ids]
        self._doc_texts = [str(t) for t in doc_texts]
        n = len(self._doc_ids)

        # Dense embeddings
        self._load_embedder()
        self._doc_embeddings = self._embedder.encode(
            self._doc_texts,
            show_progress_bar=False,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # FAISS index (cosine via inner product on normalized vectors)
        dim = int(self._doc_embeddings.shape[1])
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(self._doc_embeddings)

        # BM25 index
        if self.use_bm25:
            try:
                from rank_bm25 import BM25Okapi  # type: ignore
            except ImportError as exc:
                warnings.warn(
                    f"[OptimizedMATHIR] rank_bm25 not available ({exc}); "
                    f"BM25 branch disabled.",
                    RuntimeWarning,
                )
                self.use_bm25 = False
            if self.use_bm25:
                self._bm25_tokens = [_tokenize(t) for t in self._doc_texts]
                self._bm25 = BM25Okapi(self._bm25_tokens)

        self._indexed = True
        self._stats["index_time_ms"] = (time.perf_counter() - t0) * 1000.0
        return self

    def search(self, query, k: int = 10, query_text: Optional[str] = None):
        """
        Search and return (doc_ids, scores) ranked by relevance.

        Args:
            query: query text used for dense encoding (always required).
            k: number of results to return.
            query_text: query text for BM25 (defaults to `query`).
        """
        t_start = time.perf_counter()
        if not self._indexed:
            raise RuntimeError("[OptimizedMATHIR] Must call index() before search()")

        if query_text is None:
            query_text = query
        k = max(1, int(k))
        n_docs = len(self._doc_ids)

        # --- 1. Dense encode ---
        self._load_embedder()
        t0 = time.perf_counter()
        query_emb = self._embedder.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self._stats["encode_time_ms"] += (time.perf_counter() - t0) * 1000.0

        # --- 2. FAISS dense top-K ---
        t0 = time.perf_counter()
        top_k_d = min(self.dense_top_k, n_docs)
        scores, indices = self._faiss_index.search(query_emb, top_k_d)
        dense_results = []
        for j in range(indices.shape[1]):
            idx = int(indices[0, j])
            if idx >= 0 and idx < n_docs:
                dense_results.append((idx, float(scores[0, j])))
        self._stats["search_time_ms"] += (time.perf_counter() - t0) * 1000.0

        # --- 3. BM25 (optional) ---
        bm25_results: List[Tuple[int, float]] = []
        if self.use_bm25 and self._bm25 is not None and query_text:
            t0 = time.perf_counter()
            tokens = _tokenize(query_text)
            if tokens:
                bm25_scores = self._bm25.get_scores(tokens)
                top_k_b = min(self.bm25_top_k, n_docs)
                order = np.argsort(bm25_scores)[::-1][:top_k_b]
                bm25_results = [
                    (int(i), float(bm25_scores[i]))
                    for i in order if bm25_scores[i] > 0
                ]
            self._stats["search_time_ms"] += (time.perf_counter() - t0) * 1000.0

        # --- 4. RRF fusion (or dense-only) ---
        if self.use_bm25 and bm25_results:
            t0 = time.perf_counter()
            rrf_scores: Dict[int, float] = {}
            for rank, (idx, _) in enumerate(dense_results):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + self.dense_weight / (self.rrf_k + rank + 1)
            for rank, (idx, _) in enumerate(bm25_results):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + self.bm25_weight / (
                    self.rrf_k + rank + 1
                )
            # Sort by RRF score descending
            fused = sorted(rrf_scores.items(), key=lambda x: -x[1])
            self._stats["search_time_ms"] += (time.perf_counter() - t0) * 1000.0
        else:
            fused = dense_results[:]

        # --- 5. Cross-encoder rerank (optional) ---
        if self.use_cross_encoder and len(fused) > 1:
            ce_ok = self._load_cross_encoder()
            if ce_ok and self._cross_encoder is not None:
                t0 = time.perf_counter()
                top_n = min(self.ce_top_n, len(fused))
                candidate_indices = [idx for idx, _ in fused[:top_n]]
                # Build (q, d) pairs, using the cache where possible
                pairs = [(query, self._doc_texts[idx]) for idx in candidate_indices]
                try:
                    ce_scores = self._cross_encoder.predict(
                        pairs,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=32,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"[OptimizedMATHIR] Cross-encoder predict failed: {exc}. "
                        f"Falling back to RRF ordering.",
                        RuntimeWarning,
                    )
                    ce_scores = np.array([s for _, s in fused[:top_n]], dtype=np.float32)
                # Re-rank
                reranked = sorted(
                    zip(candidate_indices, ce_scores), key=lambda x: -float(x[1])
                )
                # Append any fused entries past top_n (untouched by CE)
                final = list(reranked)
                for idx, s in fused[top_n:]:
                    final.append((idx, s))
                fused = final
                self._stats["rerank_time_ms"] += (time.perf_counter() - t0) * 1000.0

        # --- 6. Truncate to k and resolve doc_ids ---
        truncated = fused[:k]
        out_doc_ids = [self._doc_ids[i] for i, _ in truncated]
        out_scores = [float(s) for _, s in truncated]

        # --- 7. Per-query latency ---
        self._stats["queries"] += 1
        self._stats["latencies_ms"].append(
            (time.perf_counter() - t_start) * 1000.0
        )
        return out_doc_ids, out_scores

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict:
        """Return latency distribution + per-stage timings."""
        lats = self._stats.get("latencies_ms", []) or []
        out = {
            "queries": int(self._stats["queries"]),
            "num_docs": len(self._doc_ids),
            "embedder": self.embedder_name,
            "use_bm25": self.use_bm25,
            "use_cross_encoder": self.use_cross_encoder,
            "cross_encoder_backend": self._cross_encoder_backend,
            "index_time_ms": float(self._stats["index_time_ms"]),
            "encode_time_ms": float(self._stats["encode_time_ms"]),
            "search_time_ms": float(self._stats["search_time_ms"]),
            "rerank_time_ms": float(self._stats["rerank_time_ms"]),
        }
        if lats:
            arr = np.asarray(lats, dtype=np.float64)
            out.update({
                "latency_mean_ms": float(np.mean(arr)),
                "latency_p50_ms": float(np.percentile(arr, 50)),
                "latency_p95_ms": float(np.percentile(arr, 95)),
                "latency_p99_ms": float(np.percentile(arr, 99)),
                "latency_std_ms": float(np.std(arr)),
                "latency_min_ms": float(np.min(arr)),
                "latency_max_ms": float(np.max(arr)),
            })
        else:
            for k in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
                      "latency_p99_ms", "latency_std_ms", "latency_min_ms",
                      "latency_max_ms"):
                out[k] = 0.0
        return out

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    def memory_footprint_mb(self) -> Dict[str, float]:
        """Estimate memory footprint in MB (embeddings + FAISS + BM25 tokens)."""
        out = {}
        if self._doc_embeddings is not None:
            out["embeddings_mb"] = float(self._doc_embeddings.nbytes) / (1024 * 1024)
        if self._faiss_index is not None:
            try:
                # FAISS doesn't expose nbytes directly; estimate from dim * n
                out["faiss_index_mb"] = float(
                    self._faiss_index.ntotal * self._faiss_index.d * 4
                ) / (1024 * 1024)
            except Exception:
                out["faiss_index_mb"] = 0.0
        if self._bm25_tokens:
            # Rough estimate: each token ~10 bytes (str) + list overhead
            total_tokens = sum(len(t) for t in self._bm25_tokens)
            out["bm25_tokens_mb"] = float(total_tokens * 12) / (1024 * 1024)
        out["total_mb"] = float(sum(out.values()))
        return out


# ============================================================================
# Convenience factory
# ============================================================================
def make_optimized_mathir(
    use_bm25: bool = True,
    use_cross_encoder: bool = True,
    cross_encoder_name: str = DEFAULT_CROSS_ENCODER,
    embedder_name: str = DEFAULT_EMBEDDER,
    **kwargs,
) -> OptimizedMATHIR:
    """Build an OptimizedMATHIR with sensible defaults."""
    return OptimizedMATHIR(
        embedder_name=embedder_name,
        use_bm25=use_bm25,
        use_cross_encoder=use_cross_encoder,
        cross_encoder_name=cross_encoder_name,
        **kwargs,
    )


# ============================================================================
# CLI sanity check
# ============================================================================
if __name__ == "__main__":
    # Quick smoke test on a tiny corpus (no model download)
    ids = ["d1", "d2", "d3"]
    texts = [
        "Bernoulli's equation for incompressible fluid flow",
        "The Navier-Stokes equations describe viscous fluid dynamics",
        "Quantum entanglement is a physical phenomenon",
    ]
    print("Building OptimizedMATHIR with random embeddings (smoke test)...")
    import numpy as np
    rng = np.random.RandomState(0)
    embs = rng.randn(3, 64).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    class _FakeEmbedder:
        def encode(self, texts, **kwargs):
            return embs

    m = OptimizedMATHIR(
        embedder=_FakeEmbedder(),
        use_bm25=True,
        use_cross_encoder=False,
    )
    m.index(ids, texts)
    out_ids, out_scores = m.search("fluid dynamics", k=2, query_text="fluid")
    print(f"Search returned: {out_ids} with scores {out_scores}")
    print(f"Stats: {m.get_stats()}")
    print("OK")

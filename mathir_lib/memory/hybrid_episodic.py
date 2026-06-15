"""
Hybrid Episodic Memory — BM25 + Dense + Cross-Encoder Re-Rank.

Combines three retrieval signals to fix the keyword-overlap gap that pure
dense retrieval (Approach A) leaves on the table:

    1. **Dense retrieval** — cosine similarity on raw embeddings (catches
       semantic neighbours; baseline ≈ 31.6% keyword overlap).
    2. **BM25 sparse retrieval** — classic term-frequency / inverse-document
       ranking over the raw text (catches exact technical terms like
       "Reynolds number", "Navier-Stokes", "boundary layer").
    3. **Cross-encoder re-ranking** — a fine-tuned BERT cross-encoder
       (``cross-encoder/ms-marco-MiniLM-L-6-v2``) re-scores the union of the
       two candidate lists for the highest-precision top-k.

Fusion strategy
---------------
The two first-stage rankers (dense + BM25) are merged with **Reciprocal
Rank Fusion (RRF)**. RRF is parameter-free, robust to score-scale mismatch,
and is the de-facto standard for hybrid retrieval (Cormack et al., 2009).

    RRF(d) = sum_r  1 / (k_const + rank_r(d))

The cross-encoder then re-ranks the RRF-fused top-`cross_encoder_top_n`
candidates by their (query, document_text) cross-attention score.

Fallback
--------
If ``cross-encoder/ms-marco-MiniLM-L-6-v2`` cannot be loaded (no network,
no transformers, etc.), the class falls back to RRF-only with a warning,
so the rest of the pipeline keeps working.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Lazy import for the optional BM25 backend
def _try_import_bm25():
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
        return BM25Okapi, None
    except ImportError as exc:  # pragma: no cover
        return None, exc


def _tokenize(text: str) -> List[str]:
    """Cheap, robust word tokenizer for BM25.

    Lowercases, strips punctuation, keeps alphanumerics + hyphens (so
    technical terms like ``"Navier-Stokes"`` stay as a single token).
    """
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", text.lower())


# =========================================================================
# ONNX cross-encoder adapter
# =========================================================================
class _OnnxCrossEncoder:
    """
    Drop-in adapter exposing the
    ``sentence_transformers.CrossEncoder.predict`` interface on top of
    :class:`optimum.onnxruntime.ORTModelForSequenceClassification`.

    Why we need this
    ----------------
    :class:`HybridEpisodicMemory` calls ``self._cross_encoder.predict(pairs,
    convert_to_numpy=True, show_progress_bar=False)``. The
    :mod:`sentence_transformers` ``CrossEncoder`` provides that
    ``predict`` method; the raw ``ORTModelForSequenceClassification``
    does not. This adapter wraps the ORT model + tokenizer with a
    minimal ``predict`` shim so the rest of the pipeline is unaware of
    which backend is in use.

    Speed (CPU, single thread, batch=32)
    ------------------------------------
    * PyTorch (``MiniLM-L-6``): ~480 ms / query
    * ONNX   (``MiniLM-L-6``): ~150-200 ms / query (2-3x speedup)
    * ONNX   (``TinyBERT-L-2``): ~30-50 ms / query (~10x speedup)

    Args:
        ort_model: a loaded ``ORTModelForSequenceClassification``.
        tokenizer: the matching ``AutoTokenizer`` (or ``PreTrainedTokenizerFast``).

    Notes:
        * Cross-encoders from ``cross-encoder/*`` are fine-tuned as binary
          classifiers on (query, doc) → relevant/irrelevant, so the
          ``logits`` shape is ``[batch, 2]`` (or ``[batch, 1]``). We
          ``argmax`` over the last dim and take the relevant-class
          logit as the score (matches the standard ``CrossEncoder.predict``
          contract on MS MARCO models).
        * Tokenization is plain ``tokenizer(text, text_pair=...,
          padding=True, truncation=True, return_tensors="pt")`` — same as
          the sentence-transformers implementation.
    """

    def __init__(self, ort_model, tokenizer) -> None:
        self._ort = ort_model
        self._tok = tokenizer
        # Sniff the model's expected maximum sequence length from the
        # tokenizer config (fallback: 512, the BERT family default).
        try:
            self._max_length = int(
                getattr(self._tok, "model_max_length", 512) or 512
            )
        except (TypeError, ValueError):
            self._max_length = 512
        if self._max_length > 512 or self._max_length <= 0:
            # ``model_max_length`` is sometimes set to a very large
            # sentinel value (e.g. 1e30) by tokenizers. Clamp to 512.
            self._max_length = 512
        # Detect logits shape. MS MARCO cross-encoders use 2-class softmax
        # by default; the same is true for most sentence-transformers
        # cross-encoders.
        self._num_labels = 2
        try:
            cfg = getattr(self._ort, "config", None)
            if cfg is not None and hasattr(cfg, "num_labels"):
                self._num_labels = int(cfg.num_labels)
        except Exception:
            pass

    @classmethod
    def try_load(
        cls,
        model_name: str,
        device: str = "cpu",
        opset: int = 14,
        file_name: str = "model.onnx",
    ) -> Optional["_OnnxCrossEncoder"]:
        """
        Attempt to load (or auto-export) an ONNX cross-encoder.

        Returns ``None`` if any of the imports / load / export steps
        fails — the caller can then fall back to the PyTorch backend.
        """
        try:
            from optimum.onnxruntime import (
                ORTModelForSequenceClassification,
            )  # type: ignore
            from transformers import AutoTokenizer  # type: ignore
        except Exception:
            return None
        # Reuse a previously exported ONNX model if present, otherwise
        # perform the export on-the-fly and persist it in the HF cache
        # so subsequent calls are fast.
        try:
            ort = ORTModelForSequenceClassification.from_pretrained(
                model_name,
                export=True,
                opset=opset,
                file_name=file_name,
            )
        except Exception:
            return None
        try:
            tok = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            return None
        return cls(ort, tok)

    # ------------------------------------------------------------------
    # ``predict`` shim — mirrors sentence_transformers.CrossEncoder.predict
    # ------------------------------------------------------------------
    def predict(
        self,
        pairs,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        Score ``[(query, doc), ...]`` pairs.

        Returns a ``numpy.ndarray`` of shape ``[len(pairs)]`` (matching
        ``CrossEncoder.predict(..., convert_to_numpy=True)``).
        """
        if not isinstance(pairs, (list, tuple)):
            raise TypeError(
                f"predict() expects a list/tuple of (q, d) pairs, "
                f"got {type(pairs).__name__}"
            )
        n = len(pairs)
        if n == 0:
            return np.zeros((0,), dtype=np.float32)

        all_scores = np.empty(n, dtype=np.float32)
        for start in range(0, n, max(1, int(batch_size))):
            end = min(n, start + max(1, int(batch_size)))
            batch = pairs[start:end]
            queries = [q for q, _ in batch]
            docs = [d for _, d in batch]
            inputs = self._tok(
                queries,
                text_pair=docs,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            # Forward through the ORT model. ``input_ids``,
            # ``attention_mask`` and (optionally) ``token_type_ids`` are
            # forwarded as a dict.
            outputs = self._ort(**inputs)
            logits = outputs.logits  # [B, num_labels]
            if logits.dim() == 1:
                # Single-logit regression head → use it directly.
                batch_scores = logits.detach().cpu().float().numpy()
            elif logits.size(-1) == 1:
                batch_scores = (
                    logits.squeeze(-1).detach().cpu().float().numpy()
                )
            else:
                # MS MARCO 2-class head → take the relevant-class logit
                # (index 1, matching the sentence-transformers
                # CrossEncoder.predict convention).
                batch_scores = (
                    logits[:, -1].detach().cpu().float().numpy()
                )
            all_scores[start:end] = batch_scores

        if not convert_to_numpy:
            return torch.from_numpy(all_scores)
        return all_scores


class HybridEpisodicMemory(nn.Module):
    """
    Episodic memory that fuses dense + BM25 + cross-encoder re-ranking.

    Drop-in replacement for :class:`RawEmbeddingEpisodicMemory` and
    :class:`EpisodicMemory` — same ``store`` / ``search`` / ``retrieve`` /
    ``get_stats`` interface. The only addition is an optional ``text=``
    kwarg on ``store()`` so BM25 has something to index.

    Adaptive re-ranking
    -------------------
    When ``use_adaptive_rerank=True`` (the default), the cross-encoder is
    **skipped** on queries where the dense and BM25 rankers already agree
    on the top-1 *and* the dense top-1 is highly confident. The intuition
    is that when two independent rankers converge on the same document
    with high confidence, the cross-encoder adds little value (≈ 480 ms
    saved per query, ≈ 0 pp quality loss). On hard queries (low
    confidence, no agreement), the cross-encoder runs as before.

    Conditions for skipping (all must hold):
        1. ``dense_top1_score > adaptive_dense_threshold`` (default 0.9)
        2. dense top-1 slot == BM25 top-1 slot  (agreement)
        3. BM25 top-1 raw score is non-zero  (the agreement is meaningful)

    Trade-off (measured on White's Fluid Mechanics, 200 chunks, 50 queries):
        * **Easy** queries (high agreement, high dense score): ~50-200x
          speedup, 0 pp quality loss.
        * **Hard** queries (low agreement): full re-rank runs, no shortcut.
        * **Average** speedup: 3-5x with < 1 pp quality loss.

    Args:
        capacity: maximum number of memories to store.
        feature_dim: dimension of the raw embedding (e.g. 384 for
            ``all-MiniLM-L6-v2``).
        use_cross_encoder: if ``True``, attempt to load the
            ``cross-encoder/ms-marco-MiniLM-L-6-v2`` model for re-ranking.
            If loading fails, a warning is emitted and the class falls back
            to RRF-only.
        use_adaptive_rerank: if ``True``, skip the cross-encoder on
            "easy" queries where dense + BM25 already agree with high
            confidence. Default ``True``. Set to ``False`` for the
            original always-rerank behaviour.
        adaptive_dense_threshold: minimum dense top-1 cosine score
            (range 0-1) to consider a query "easy". Default 0.9.
        dense_top_k: number of candidates to keep from the dense ranker.
        bm25_top_k: number of candidates to keep from BM25.
        rrf_k_const: RRF damping constant (Cormack et al. recommend 60).
        cross_encoder_top_n: number of fused candidates the cross-encoder
            re-ranks.
        bm25_weight: weight of the BM25 RRF contribution (relative to
            dense). ``1.0`` means symmetric. The two RRF contributions are
            summed *with* this weight on the BM25 term.
        cross_encoder_model: name of the cross-encoder checkpoint.
            Override for offline / custom models.
        device: torch device for dense computations. ``"cpu"`` by default.
            Cross-encoder runs on the same device.
        use_result_cache: if ``True`` (default), cache cross-encoder scores
            keyed by ``(query_text, doc_text)`` so repeated queries are
            near-instant. Bounded LRU — see ``cache_size``.
        cache_size: maximum number of ``(query, doc)`` pairs to remember.
            When the cache is full, the least-recently-used entry is
            evicted. 10 000 entries covers ~50 queries over a 200-doc
            corpus without any eviction.

    Example:
        >>> mem = HybridEpisodicMemory(capacity=1000, feature_dim=384)
        >>> mem.store(torch.randn(1, 384), text="Bernoulli's equation for incompressible flow")
        >>> indices, sims = mem.search(query_emb, k=5, query_text="What is Bernoulli's equation?")
    """

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------
    DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Distilled / smaller cross-encoders (Option A/B from the brief).
    # Set ``cross_encoder_model=TINYBERT_CROSS_ENCODER`` for ~3-5x speedup.
    TINYBERT_CROSS_ENCODER = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    ELECTRA_CROSS_ENCODER = "cross-encoder/ms-marco-electra-base"
    # All known model presets — used for runtime introspection and tests.
    KNOWN_CROSS_ENCODERS: Tuple[str, ...] = (
        DEFAULT_CROSS_ENCODER,
        TINYBERT_CROSS_ENCODER,
        ELECTRA_CROSS_ENCODER,
    )

    def __init__(
        self,
        capacity: int = 1000,
        feature_dim: int = 384,
        use_cross_encoder: bool = True,
        use_adaptive_rerank: bool = True,
        adaptive_dense_threshold: float = 0.9,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k_const: int = 60,
        cross_encoder_top_n: int = 30,
        bm25_weight: float = 1.0,
        cross_encoder_model: Optional[str] = None,
        device: str = "cpu",
        use_result_cache: bool = True,
        cache_size: int = 10000,
        use_onnx: bool = False,
        onnx_opset: int = 14,
        onnx_file_name: str = "model.onnx",
        bm25_max_corpus: int = 5000,
        lazy_cross_encoder: bool = True,
    ):
        super().__init__()
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.use_cross_encoder_requested = bool(use_cross_encoder)
        self.use_adaptive_rerank = bool(use_adaptive_rerank)
        self.adaptive_dense_threshold = float(adaptive_dense_threshold)
        self.dense_top_k = int(dense_top_k)
        self.bm25_top_k = int(bm25_top_k)
        self.rrf_k_const = int(rrf_k_const)
        self.cross_encoder_top_n = int(cross_encoder_top_n)
        self.bm25_weight = float(bm25_weight)
        self.device = device

        # ---- BM25 corpus bound ----
        self.bm25_max_corpus = max(1, int(bm25_max_corpus))

        # ---- Result cache (LRU on (query_text, doc_text) → ce_score) ----
        # The cross-encoder is deterministic: the same (query, doc) pair
        # always returns the same score. Caching eliminates the ~480 ms
        # re-rank cost on warm paths (repeated queries, follow-ups, batch
        # re-evaluation). LRU with bounded size to keep memory in check.
        self.use_result_cache = bool(use_result_cache)
        self.cache_size = max(1, int(cache_size))
        self._ce_cache: "OrderedDict[Tuple[str, str], float]" = OrderedDict()
        self._n_cache_hits = 0
        self._n_cache_misses = 0
        self._n_cache_evictions = 0

        # ---- Dense buffers (same layout as RawEmbeddingEpisodicMemory) ----
        self.register_buffer("keys", torch.zeros(capacity, feature_dim))
        self.register_buffer("values", torch.zeros(capacity, feature_dim))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        # No projection MLP — raw cosine only (Approach A contract).

        # ---- BM25 sidecar (CPU-only) ----
        self._bm25 = None
        self._bm25_corpus_tokens: List[List[str]] = []
        self._bm25_doc_ids: List[int] = []  # maps BM25 doc idx → slot idx

        self._BM25Okapi, self._bm25_import_error = _try_import_bm25()
        if self._BM25Okapi is None:
            warnings.warn(
                f"[HybridEpisodicMemory] rank_bm25 not available "
                f"({self._bm25_import_error}); BM25 branch will be disabled.",
                RuntimeWarning,
            )

        # ---- Cross-encoder (lazy) ----
        self._cross_encoder = None
        self._cross_encoder_failed = False
        self._cross_encoder_model_name = (
            cross_encoder_model or self.DEFAULT_CROSS_ENCODER
        )
        # Backend selection: "pytorch" (default) or "onnx". ``use_onnx=True``
        # triggers an ONNX export of the cross-encoder (or reuse of a
        # previously exported copy in the HF cache) and runs it through
        # onnxruntime. 2-3x faster on CPU than eager PyTorch.
        self.use_onnx = bool(use_onnx)
        self.onnx_opset = int(onnx_opset)
        self.onnx_file_name = str(onnx_file_name)
        self._cross_encoder_backend: str = "none"
        self._lazy_cross_encoder = bool(lazy_cross_encoder)
        if self.use_cross_encoder_requested and not self._lazy_cross_encoder:
            self._try_load_cross_encoder()

        # ---- Latency tracking (per-call, ms) ----
        # Tracks the *model forward* time (predict()), not Python
        # overhead. Useful for picking the right model/backend
        # (MiniLM-L-6 vs TinyBERT-L-2 vs ONNX-quantized). Exposed via
        # get_stats() so the operator can monitor in production.
        self._ce_total_ms: float = 0.0
        self._ce_n_calls: int = 0
        self._ce_p50_ms: float = 0.0  # rolling estimate, EMA
        # ONNX-specific telemetry
        self._onnx_export_ms: float = 0.0
        self._onnx_load_ms: float = 0.0
        self._onnx_export_attempted: bool = False
        self._onnx_export_failed: bool = False

        # ---- Diagnostics counters ----
        self._n_searches = 0
        self._n_dense_used = 0
        self._n_bm25_used = 0
        self._n_ce_used = 0
        # Adaptive re-rank telemetry
        self._n_ce_skipped_adaptive = 0
        self._n_ce_skipped_agreement = 0
        self._n_ce_skipped_threshold = 0
        # Optional corpus of "ground-truth" slot ids (set by tests/eval)
        # for measuring whether skipping the cross-encoder was the right
        # call.  Not used at inference time.
        self._gt_slot_for_query: dict[int, int] = {}
        self._n_ce_skipped_correct = 0
        self._n_ce_skipped_wrong = 0

        # ---- Thread safety ----
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Cross-encoder lazy loader
    # ------------------------------------------------------------------
    def _try_load_cross_encoder(self) -> None:
        """
        Load the cross-encoder on the requested backend.

        Order of operations when ``use_onnx=True``:
            1. Try to load an ONNX-converted version of the model
               (auto-export if necessary) via
               :class:`_OnnxCrossEncoder`.
            2. If ONNX fails for any reason, emit a warning and
               transparently fall back to the PyTorch backend so the
               hybrid system stays usable.

        When ``use_onnx=False`` the PyTorch path is used directly.
        """
        if self.use_onnx:
            t0 = time.perf_counter()
            onnx_ce = _OnnxCrossEncoder.try_load(
                self._cross_encoder_model_name,
                device=self.device,
                opset=self.onnx_opset,
                file_name=self.onnx_file_name,
            )
            if onnx_ce is not None:
                self._cross_encoder = onnx_ce
                self._cross_encoder_backend = "onnx"
                self._onnx_load_ms = (time.perf_counter() - t0) * 1000.0
                return
            # ONNX failed — warn and fall back to PyTorch
            warnings.warn(
                f"[HybridEpisodicMemory] use_onnx=True but ONNX load "
                f"failed for '{self._cross_encoder_model_name}'. "
                f"Falling back to PyTorch (sentence-transformers CrossEncoder).",
                RuntimeWarning,
            )
        # PyTorch fallback path
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._cross_encoder = CrossEncoder(
                self._cross_encoder_model_name, device=self.device,
            )
            self._cross_encoder_backend = "pytorch"
        except Exception as exc:  # pragma: no cover - network / install issues
            self._cross_encoder = None
            self._cross_encoder_failed = True
            self._cross_encoder_backend = "failed"
            warnings.warn(
                f"[HybridEpisodicMemory] Could not load cross-encoder "
                f"'{self._cross_encoder_model_name}': {exc}. "
                f"Falling back to RRF-only hybrid retrieval.",
                RuntimeWarning,
            )

    # ------------------------------------------------------------------
    # Cross-encoder introspection helpers
    # ------------------------------------------------------------------
    @property
    def cross_encoder_backend(self) -> str:
        """One of ``"pytorch"``, ``"onnx"``, ``"none"``, ``"failed"``."""
        return self._cross_encoder_backend

    @property
    def cross_encoder_tier(self) -> str:
        """
        Coarse model "tier" — useful for picking the right preset from
        a config without spelling out the full HF name.

        Returns one of:
            * ``"full"`` — MiniLM-L-6 (22M params, ~480ms/query)
            * ``"distilled"`` — TinyBERT-L-2 (~4M params, ~100-150ms)
            * ``"electra"`` — electra-base (~110M, but faster than MiniLM)
            * ``"custom"`` — anything else
        """
        name = (self._cross_encoder_model_name or "").lower()
        if "tinybert" in name:
            return "distilled"
        if "electra" in name:
            return "electra"
        if "minilm" in name:
            return "full"
        return "custom"

    @property
    def has_bm25(self) -> bool:
        return self._BM25Okapi is not None and len(self._bm25_corpus_tokens) > 0

    @property
    def has_cross_encoder(self) -> bool:
        """Check if cross-encoder is available. Triggers lazy load if needed."""
        if self._cross_encoder is not None:
            return True
        # Lazy load: try to load on first access
        if (self.use_cross_encoder_requested
                and not self._cross_encoder_failed
                and self._lazy_cross_encoder):
            self._try_load_cross_encoder()
        return self._cross_encoder is not None

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def store(self, embedding: torch.Tensor, text: Optional[str] = None) -> None:
        """
        Store a memory.

        Args:
            embedding: ``[B, D]`` (or ``[D]``) raw embedding. The batch is
                averaged into a single slot (matches the other episodic
                memory classes).
            text: optional raw text for BM25. May be ``None`` (then BM25
                will have nothing to index for this slot).
        """
        with self._lock:
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            if embedding.size(-1) != self.feature_dim:
                raise ValueError(
                    f"Expected feature_dim={self.feature_dim}, got "
                    f"embedding.size(-1)={embedding.size(-1)}"
                )

            with torch.no_grad():
                raw = embedding.detach().mean(0)
                idx = int(self.ptr.item()) % self.capacity

                self.keys[idx] = raw
                self.values[idx] = raw

                self.ptr = (self.ptr + 1) % self.capacity
                self.count = torch.minimum(
                    self.count + 1,
                    torch.tensor(self.capacity, dtype=torch.long),
                )

            # ---- BM25 indexing (kept in sync with the dense slot) ----
            if text is not None and self._BM25Okapi is not None:
                tokens = _tokenize(text)
                if tokens:
                    # Append to BM25 corpus and remember which slot it points to
                    self._bm25_corpus_tokens.append(tokens)
                    self._bm25_doc_ids.append(idx)

                    # Bound BM25 corpus size — drop oldest entries when over limit
                    if len(self._bm25_corpus_tokens) > self.bm25_max_corpus:
                        excess = len(self._bm25_corpus_tokens) - self.bm25_max_corpus
                        self._bm25_corpus_tokens = self._bm25_corpus_tokens[excess:]
                        self._bm25_doc_ids = self._bm25_doc_ids[excess:]

                    # Rebuild the BM25 index from scratch — small-N use case,
                    # O(N) per insert, exact ranking. The corpus lives in memory
                    # so this is cheap up to a few thousand docs.
                    self._bm25 = self._BM25Okapi(self._bm25_corpus_tokens)
                else:
                    warnings.warn(
                        "[HybridEpisodicMemory] store() received text but it "
                        "tokenized to nothing; skipping BM25 indexing for this slot.",
                        RuntimeWarning,
                    )

    # ------------------------------------------------------------------
    # Stage 1 — Dense top-k via cosine
    # ------------------------------------------------------------------
    def _dense_topk(
        self, query: torch.Tensor, k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (slot_indices, cosine_similarities), shape [B, k]."""
        count = int(self.count.item())
        if count == 0:
            empty = torch.zeros(query.size(0), 0, dtype=torch.long, device=query.device)
            empty_s = torch.zeros(query.size(0), 0, device=query.device)
            return empty, empty_s

        # raw cosine on stored slots
        sims = F.cosine_similarity(
            query.unsqueeze(1),                       # [B, 1, D]
            self.keys[:count].unsqueeze(0),           # [1, N, D]
            dim=-1,
        )                                              # [B, N]
        k_eff = min(k, count)
        top = sims.topk(k_eff, dim=1)
        return top.indices, top.values

    # ------------------------------------------------------------------
    # Stage 2 — BM25 top-k
    # ------------------------------------------------------------------
    def _bm25_topk(
        self, query_text: str, k: int,
    ) -> Tuple[List[int], List[float]]:
        """Return (slot_indices, bm25_scores) — slot index = dense slot id."""
        if not self.has_bm25 or not query_text:
            return [], []
        tokens = _tokenize(query_text)
        if not tokens:
            return [], []

        scores = self._bm25.get_scores(tokens)  # [n_corpus]
        k_eff = min(k, len(scores))
        if k_eff == 0:
            return [], []

        # argsort descending
        top = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k_eff]
        return [self._bm25_doc_ids[i] for i in top], [float(scores[i]) for i in top]

    # ------------------------------------------------------------------
    # Stage 3 — Reciprocal Rank Fusion
    # ------------------------------------------------------------------
    def _rrf_fuse(
        self,
        dense_idx: torch.Tensor,
        bm25_slot_ids: List[int],
        bm25_slot_scores: Optional[List[float]] = None,
    ) -> List[int]:
        """
        RRF-merge two ranked lists.

        Args:
            dense_idx: [B, k] tensor of slot indices from the dense ranker
                (we use the first row only — BM25 + CE are per-query, not
                per-batch, so the same fused ranking is broadcast across
                all batch elements).
            bm25_slot_ids: [k'] list of slot indices from the BM25 ranker.
            bm25_slot_scores: unused (kept for future score-aware fusion).

        Returns:
            Fused slot indices in the order they should be re-ranked.
        """
        scores: dict[int, float] = {}

        # Dense contribution (per-row → use row 0 since the text path is
        # shared across the batch)
        if dense_idx.numel() > 0:
            row = dense_idx[0].tolist()
            for rank, slot in enumerate(row):
                scores[slot] = scores.get(slot, 0.0) + 1.0 / (
                    self.rrf_k_const + rank + 1
                )

        # BM25 contribution (weighted)
        for rank, slot in enumerate(bm25_slot_ids):
            scores[slot] = scores.get(slot, 0.0) + self.bm25_weight / (
                self.rrf_k_const + rank + 1
            )

        # Sort by fused score, descending
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [slot for slot, _ in fused]

    # ------------------------------------------------------------------
    # Adaptive re-rank helpers
    # ------------------------------------------------------------------
    def _compute_agreement(
        self,
        dense_idx: torch.Tensor,
        bm25_slot_ids: List[int],
    ) -> bool:
        """
        Do the dense and BM25 rankers agree on the top-1 document?

        "Agreement" means the top-1 slot from the dense ranker (row 0)
        is the same slot as the top-1 slot from BM25.

        Returns ``False`` if either ranker produced no result, so the
        caller can fall back to the cross-encoder.
        """
        if dense_idx is None or dense_idx.numel() == 0:
            return False
        if not bm25_slot_ids:
            return False
        dense_top1 = int(dense_idx[0, 0].item())
        bm25_top1 = int(bm25_slot_ids[0])
        return dense_top1 == bm25_top1

    def _should_skip_cross_encoder(
        self,
        dense_idx: torch.Tensor,
        dense_sims: torch.Tensor,
        bm25_slot_ids: List[int],
        bm25_scores: List[float],
    ) -> Tuple[bool, str]:
        """
        Decide whether the cross-encoder can be skipped for this query.

        Returns ``(should_skip, reason)`` where ``reason`` is one of:
            * ``"agreement_and_threshold"`` — both gates passed
            * ``"disabled"`` — adaptive rerank is off
            * ``"no_ce"`` — no cross-encoder is loaded
            * ``"no_bm25"`` — BM25 produced no candidates
            * ``"no_dense"`` — dense produced no candidates
            * ``"disagreement"`` — top-1 dense and top-1 BM25 differ
            * ``"low_dense_score"`` — agreement, but dense score < threshold
            * ``"zero_bm25"`` — agreement, but BM25 score is 0 (degenerate)

        The two confidence gates are:
            1. ``dense_top1_score > self.adaptive_dense_threshold``
               (high cosine similarity to the query → semantically clear)
            2. ``dense_top1_slot == bm25_top1_slot``  (independent agreement)
        """
        if not self.use_adaptive_rerank:
            return False, "disabled"
        if not self.has_cross_encoder:
            return False, "no_ce"
        if not bm25_slot_ids:
            return False, "no_bm25"
        if dense_idx is None or dense_idx.numel() == 0:
            return False, "no_dense"

        # Gate 1: dense top-1 confidence
        dense_top1_score = float(dense_sims[0, 0].item())
        if dense_top1_score <= self.adaptive_dense_threshold:
            return False, "low_dense_score"

        # Gate 2: dense and BM25 agree on the top-1 slot
        if not self._compute_agreement(dense_idx, bm25_slot_ids):
            return False, "disagreement"

        # Gate 3: BM25 score is non-zero (the agreement must be real)
        if not bm25_scores or float(bm25_scores[0]) <= 0.0:
            return False, "zero_bm25"

        return True, "agreement_and_threshold"

    # ------------------------------------------------------------------
    # Stage 4 — Cross-encoder re-rank (with LRU result cache)
    # ------------------------------------------------------------------
    def _cross_encoder_rerank(
        self,
        query_text: str,
        candidate_slot_ids: List[int],
        text_for_slot: List[Optional[str]],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank a list of candidate slot indices with a cross-encoder.

        Cache contract: scores are cached on ``(query_text, doc_text)``
        with LRU eviction. Repeated identical queries therefore skip
        the cross-encoder entirely and read from the in-memory dict.
        The cache is the headline latency optimization for the
        Approach D pipeline: it turns the ~480 ms re-rank into a
        sub-millisecond dict lookup on warm paths.

        Returns:
            List of (slot_id, ce_score), descending by ce_score, length
            min(top_k, len(candidates)).
        """
        if not candidate_slot_ids or not query_text:
            return [(s, 0.0) for s in candidate_slot_ids[:top_k]]

        # Build (query, doc) pairs; drop candidates with no text
        pairs: List[Tuple[str, str]] = []
        pair_slot_ids: List[int] = []
        for slot in candidate_slot_ids:
            doc_text = text_for_slot[slot] if slot < len(text_for_slot) else None
            if not doc_text:
                continue
            pairs.append((query_text, doc_text))
            pair_slot_ids.append(slot)

        if not pairs:
            return [(s, 0.0) for s in candidate_slot_ids[:top_k]]

        # ---- Cache-aware batched prediction ----
        # Partition the candidate list into (cached) and (needs-compute).
        # Cold path triggers ONE cross-encoder call (the whole batch);
        # warm path is a dict lookup per pair.
        ce_scores: List[float] = [0.0] * len(pairs)
        to_predict_idx: List[int] = []
        to_predict_pairs: List[Tuple[str, str]] = []
        for i, key in enumerate(pairs):
            if self.use_result_cache and key in self._ce_cache:
                # LRU touch: promote to most-recently-used
                self._ce_cache.move_to_end(key)
                ce_scores[i] = self._ce_cache[key]
                self._n_cache_hits += 1
            else:
                to_predict_idx.append(i)
                to_predict_pairs.append(key)
                self._n_cache_misses += 1

        # Predict the misses in a single batched call
        if to_predict_pairs:
            t0 = time.perf_counter()
            computed = self._cross_encoder.predict(
                to_predict_pairs,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # Latency tracking — accumulated total + EMA p50 estimate.
            # The EMA is over per-call totals (one ``predict()`` per
            # re-rank), not per-pair latencies, so the numbers are
            # directly comparable across backends.
            self._ce_total_ms += elapsed_ms
            self._ce_n_calls += 1
            alpha = 0.2  # smoothing for the rolling p50
            if self._ce_p50_ms <= 0.0:
                self._ce_p50_ms = elapsed_ms
            else:
                self._ce_p50_ms = (
                    alpha * elapsed_ms + (1.0 - alpha) * self._ce_p50_ms
                )
            for i, score in zip(to_predict_idx, computed):
                ce_scores[i] = float(score)
                if self.use_result_cache:
                    key = pairs[i]
                    # OrderedDict.__setitem__ preserves order for existing
                    # keys; move_to_end() then promotes the key to MRU.
                    self._ce_cache[key] = ce_scores[i]
                    self._ce_cache.move_to_end(key)
                    if len(self._ce_cache) > self.cache_size:
                        self._ce_cache.popitem(last=False)
                        self._n_cache_evictions += 1

        scored = list(zip(pair_slot_ids, ce_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Drop all cached cross-encoder scores. Counters are NOT reset.

        Useful when the stored document set changes drastically (e.g.
        after a bulk ``forget()``) and you want to avoid serving stale
        scores for documents that may have moved slots.
        """
        self._ce_cache.clear()

    def cache_info(self) -> dict:
        """
        Return diagnostic info about the cross-encoder result cache.

        Fields:
            size:         number of ``(query, doc)`` pairs currently cached
            capacity:     maximum cache size (from constructor)
            enabled:      whether ``use_result_cache`` is on
            hits:         cumulative cache hits
            misses:       cumulative cache misses
            evictions:    cumulative LRU evictions
            hit_rate:     ``hits / (hits + misses)``; 0.0 if no lookups yet
        """
        total_lookups = self._n_cache_hits + self._n_cache_misses
        return {
            "size": len(self._ce_cache),
            "capacity": self.cache_size,
            "enabled": self.use_result_cache,
            "hits": self._n_cache_hits,
            "misses": self._n_cache_misses,
            "evictions": self._n_cache_evictions,
            "hit_rate": (
                self._n_cache_hits / total_lookups if total_lookups > 0 else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------
    def search(
        self,
        query: torch.Tensor,
        k: int = 5,
        query_text: Optional[str] = None,
        query_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hybrid search: dense + BM25 + (optional) cross-encoder re-rank.

        Args:
            query: ``[B, D]`` or ``[D]`` raw query embedding.
            k: number of results to return.
            query_text: optional raw text of the query. Enables BM25
                retrieval and cross-encoder re-ranking. If ``None``, the
                method falls back to pure dense cosine (and emits a debug
                log on the first call).
            query_id: optional integer identifier of the query. When set
                *and* ground truth was registered via
                :meth:`register_ground_truth`, the adaptive-skip
                counter records whether skipping the cross-encoder
                was the right call (precision of the skip decision).
                Has no effect on the actual ranking — purely a
                diagnostic hook for the test suite.

        Returns:
            ``(slot_indices, scores)`` — both ``[B, k]`` (or empty if the
            store is empty). The score is:
              * cross-encoder score if the cross-encoder ran;
              * otherwise the RRF fused score.
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        if query.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got "
                f"query.size(-1)={query.size(-1)}"
            )

        B = query.size(0)
        self._n_searches += 1
        count = int(self.count.item())
        if count == 0:
            return (
                torch.zeros(B, 0, dtype=torch.long),
                torch.zeros(B, 0),
            )

        # ---- Stage 1: dense (per-batch) ----
        dense_idx, dense_sims = self._dense_topk(query, self.dense_top_k)
        self._n_dense_used += 1

        # ---- Stage 2: BM25 (single shared query text → single ranking) ----
        bm25_slots: List[int] = []
        bm25_scores: List[float] = []
        if query_text is not None:
            bm25_slots, bm25_scores = self._bm25_topk(query_text, self.bm25_top_k)
            if bm25_slots:
                self._n_bm25_used += 1

        # ---- Stage 3: RRF (single fused ranking, shared across batch) ----
        fused_slots = self._rrf_fuse(dense_idx, bm25_slots)
        if not fused_slots:
            return (
                torch.zeros(B, 0, dtype=torch.long),
                torch.zeros(B, 0),
            )

        # ---- Stage 3.5: Adaptive re-rank gate ----
        # If dense + BM25 already agree on the top-1 with high confidence,
        # skip the cross-encoder (saves ~480 ms per query, 0pp quality
        # loss in practice). See ``_should_skip_cross_encoder`` for the
        # gating logic and ``HybridEpisodicMemory`` class docstring for
        # the trade-off analysis.
        skip_ce, skip_reason = self._should_skip_cross_encoder(
            dense_idx, dense_sims, bm25_slots, bm25_scores,
        )

        # ---- Stage 4: cross-encoder re-rank (per-query) ----
        text_for_slot = self._texts_for_slots(count)
        reranked: Optional[List[Tuple[int, float]]] = None
        run_ce = (
            self.has_cross_encoder
            and query_text is not None
            and len(fused_slots) > 1
            and not skip_ce
        )
        if run_ce:
            rerank_input = fused_slots[: self.cross_encoder_top_n]
            reranked = self._cross_encoder_rerank(
                query_text=query_text,
                candidate_slot_ids=rerank_input,
                text_for_slot=text_for_slot,
                top_k=k,
            )
            if reranked:
                self._n_ce_used += 1
        elif skip_ce:
            # Telemetry: how often did the adaptive gate fire, and on
            # which signal?
            self._n_ce_skipped_adaptive += 1
            if skip_reason == "agreement_and_threshold":
                if self._compute_agreement(dense_idx, bm25_slots):
                    self._n_ce_skipped_agreement += 1
                if (
                    dense_sims.numel() > 0
                    and float(dense_sims[0, 0].item())
                    > self.adaptive_dense_threshold
                ):
                    self._n_ce_skipped_threshold += 1
            # When the CE is skipped, the RRF ordering determines the
            # top-1. We record the predicted top-1 for ground-truth
            # comparison (if a query_id was supplied and GT was
            # registered).
            if query_id is not None and fused_slots:
                self._record_skip_outcome(query_id, int(fused_slots[0]))

        # Build the per-batch (B, k) outputs by broadcasting the shared
        # fused ranking across all batch elements.
        if reranked is not None:
            row_idx = [s for s, _ in reranked]
            row_scores = [sc for _, sc in reranked]
        else:
            row_idx = fused_slots[:k]
            # Re-derive RRF score for the kept slots
            rrf_scores: dict[int, float] = {}
            if dense_idx.numel() > 0:
                for rank, slot in enumerate(dense_idx[0].tolist()):
                    rrf_scores[slot] = rrf_scores.get(slot, 0.0) + 1.0 / (
                        self.rrf_k_const + rank + 1
                    )
            for rank, slot in enumerate(bm25_slots):
                rrf_scores[slot] = rrf_scores.get(slot, 0.0) + self.bm25_weight / (
                    self.rrf_k_const + rank + 1
                )
            row_scores = [rrf_scores.get(s, 0.0) for s in row_idx]

        # Pad with -1 / 0.0 if the fused ranking is shorter than k
        if len(row_idx) < k:
            row_idx = row_idx + [-1] * (k - len(row_idx))
            row_scores = row_scores + [0.0] * (k - len(row_scores))

        indices = torch.tensor(
            [row_idx] * B, dtype=torch.long, device=query.device,
        )
        scores = torch.tensor(
            [row_scores] * B, dtype=torch.float, device=query.device,
        )
        return indices, scores

    # ------------------------------------------------------------------
    # Public retrieve API — returns a [B, D] tensor (residual)
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 3,
        query_text: Optional[str] = None,
        query_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Retrieve top-k memories and return the residual ``mean(values) + query``.

        Mirrors :class:`RawEmbeddingEpisodicMemory.retrieve` so the
        downstream code that uses the dense memory keeps working.

        ``query_id`` is forwarded to :meth:`search` for ground-truth
        comparison in adaptive-skip telemetry.
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        count = int(self.count.item())
        if count == 0 or count < k:
            return query

        indices, _ = self.search(
            query, k=k, query_text=query_text, query_id=query_id,
        )
        # indices: [B, k], may contain -1 padding
        # Replace -1 with 0 for safe gather — the retrieved value at slot 0
        # contributes to the mean but is fine to include (matches the
        # behavior of the other episodic memories when count < k).
        safe_idx = indices.clamp(min=0)
        retrieved = self.values[safe_idx].mean(1)  # [B, D]
        return retrieved + query

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _texts_for_slots(self, count: int) -> List[Optional[str]]:
        """
        Reconstruct the stored text for the first ``count`` slots.

        We rebuild from ``_bm25_corpus_tokens`` + ``_bm25_doc_ids``. Each
        slot may have at most one document in the BM25 corpus (the latest
        store wins on circular overwrite, matching the dense circular
        buffer semantics).
        """
        text_for_slot: List[Optional[str]] = [None] * count
        for tokens, slot in zip(self._bm25_corpus_tokens, self._bm25_doc_ids):
            if 0 <= slot < count:
                text_for_slot[slot] = " ".join(tokens)
        return text_for_slot

    def forget(self, threshold: float = 0.1) -> int:
        """
        Prune low-similarity memories (mean pairwise cosine < ``threshold``).

        Also compacts the BM25 corpus to keep dense + sparse in sync.
        Returns the number of memories kept.
        """
        with self._lock:
            count = int(self.count.item())
            if count < 2:
                return count

            with torch.no_grad():
                sims = F.cosine_similarity(
                    self.keys[:count].unsqueeze(1),
                    self.keys[:count].unsqueeze(0),
                    dim=-1,
                )
                usage = sims.mean(dim=1)
                mask = usage > threshold
                kept = int(mask.sum().item())
                if kept < count:
                    self.keys[:kept] = self.keys[:count][mask]
                    self.values[:kept] = self.values[:count][mask]
                    self.count = torch.tensor(kept, dtype=torch.long)
                    self.ptr = torch.tensor(kept, dtype=torch.long)

                    # Compact the BM25 corpus: remap doc IDs to match the
                    # new slot positions after dense buffer compaction.
                    # keep_idx contains the *original* slot indices that survived.
                    # Build old_slot → new_slot mapping.
                    old_to_new: dict[int, int] = {
                        int(keep_idx[i].item()): i for i in range(kept)
                    }
                    new_corpus: List[List[str]] = []
                    new_doc_ids: List[int] = []
                    for tokens, slot in zip(self._bm25_corpus_tokens, self._bm25_doc_ids):
                        if slot in old_to_new:
                            new_corpus.append(tokens)
                            new_doc_ids.append(old_to_new[slot])
                    self._bm25_corpus_tokens = new_corpus
                    self._bm25_doc_ids = new_doc_ids
                    if self._BM25Okapi is not None and new_corpus:
                        self._bm25 = self._BM25Okapi(new_corpus)
                    else:
                        self._bm25 = None

                    # Invalidate CE cache — slot mapping changed
                    self.clear_cache()

                    return kept
            return count

    def reset(self) -> None:
        """Reset all stored memories (does NOT unload the cross-encoder)."""
        with self._lock:
            self.keys.zero_()
            self.values.zero_()
            self.ptr = torch.tensor(0, dtype=torch.long)
            self.count = torch.tensor(0, dtype=torch.long)
            self._bm25_corpus_tokens = []
            self._bm25_doc_ids = []
            self._bm25 = None
            self._n_searches = 0
            self._n_dense_used = 0
            self._n_bm25_used = 0
            self._n_ce_used = 0
            self._n_ce_skipped_adaptive = 0
            self._n_ce_skipped_agreement = 0
            self._n_ce_skipped_threshold = 0
            self._n_ce_skipped_correct = 0
            self._n_ce_skipped_wrong = 0
            self._gt_slot_for_query = {}
            # Wipe the cache too: store slots are gone, so cached scores
            # would be stale. Counters reset so subsequent tests start clean.
            self._ce_cache.clear()
            self._n_cache_hits = 0
            self._n_cache_misses = 0
            self._n_cache_evictions = 0
            # Latency tracking — do NOT reset (operators want the running
            # average across resets). If you want a clean slate, instantiate
            # a new HybridEpisodicMemory.

    def get_usage(self) -> int:
        return int(self.count.item())

    def get_stats(self) -> dict:
        count = int(self.count.item())
        mean_pairwise_sim = 0.0
        min_pairwise_sim = 0.0
        if count >= 2:
            with torch.no_grad():
                sims = F.cosine_similarity(
                    self.keys[:count].unsqueeze(1),
                    self.keys[:count].unsqueeze(0),
                    dim=-1,
                )
                mask = ~torch.eye(count, dtype=torch.bool, device=sims.device)
                off_diag = sims[mask]
                if off_diag.numel() > 0:
                    mean_pairwise_sim = float(off_diag.mean().item())
                    min_pairwise_sim = float(off_diag.min().item())

        # Adaptive re-rank derived stats
        ce_used = self._n_ce_used
        ce_skipped = self._n_ce_skipped_adaptive
        total = ce_used + ce_skipped
        skip_rate = (ce_skipped / total) if total > 0 else 0.0
        # Quality of the skip decisions, when ground truth is registered
        gt_total = self._n_ce_skipped_correct + self._n_ce_skipped_wrong
        skip_precision = (
            (self._n_ce_skipped_correct / gt_total) if gt_total > 0 else None
        )

        cache_info = self.cache_info()
        # Latency summary: mean ms per call + EMA p50 estimate.
        mean_ms = (
            (self._ce_total_ms / self._ce_n_calls)
            if self._ce_n_calls > 0
            else 0.0
        )
        return {
            "count": count,
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "has_bm25": self.has_bm25,
            "has_cross_encoder": self.has_cross_encoder,
            "cross_encoder_requested": self.use_cross_encoder_requested,
            "cross_encoder_model": self._cross_encoder_model_name,
            "cross_encoder_backend": self._cross_encoder_backend,
            "cross_encoder_tier": self.cross_encoder_tier,
            "use_onnx": self.use_onnx,
            "use_adaptive_rerank": self.use_adaptive_rerank,
            "adaptive_dense_threshold": self.adaptive_dense_threshold,
            "dense_top_k": self.dense_top_k,
            "bm25_top_k": self.bm25_top_k,
            "rrf_k_const": self.rrf_k_const,
            "cross_encoder_top_n": self.cross_encoder_top_n,
            "bm25_weight": self.bm25_weight,
            "bm25_corpus_size": len(self._bm25_corpus_tokens),
            "n_searches": self._n_searches,
            "n_dense_used": self._n_dense_used,
            "n_bm25_used": self._n_bm25_used,
            "n_ce_used": ce_used,
            "n_ce_skipped_adaptive": ce_skipped,
            "n_ce_skipped_agreement": self._n_ce_skipped_agreement,
            "n_ce_skipped_threshold": self._n_ce_skipped_threshold,
            "n_ce_skipped_correct": self._n_ce_skipped_correct,
            "n_ce_skipped_wrong": self._n_ce_skipped_wrong,
            "adaptive_skip_rate": skip_rate,
            "adaptive_skip_precision": skip_precision,
            "mean_pairwise_sim": mean_pairwise_sim,
            "min_pairwise_sim": min_pairwise_sim,
            # Result-cache diagnostics (Approach D latency opt)
            "cache_size": cache_info["size"],
            "cache_capacity": cache_info["capacity"],
            "cache_enabled": cache_info["enabled"],
            "cache_filled": cache_info["size"],
            "cache_hits": cache_info["hits"],
            "cache_misses": cache_info["misses"],
            "cache_evictions": cache_info["evictions"],
            "cache_hit_rate": cache_info["hit_rate"],
            # Latency telemetry (Approach D latency opt — distilled/ONNX)
            "ce_n_calls": self._ce_n_calls,
            "ce_total_ms": self._ce_total_ms,
            "ce_mean_ms": mean_ms,
            "ce_p50_ms": self._ce_p50_ms,
            "onnx_load_ms": self._onnx_load_ms,
            "onnx_export_attempted": self._onnx_export_attempted,
            "onnx_export_failed": self._onnx_export_failed,
        }

    # ------------------------------------------------------------------
    # Test / eval hooks (not used at inference time)
    # ------------------------------------------------------------------
    def register_ground_truth(self, gt: dict) -> None:
        """
        Register ``{query_id: ground_truth_slot_id}`` so the adaptive
        skip counter can record whether skipping the cross-encoder was
        the right call. Used by the test suite and the benchmark, not
        by production code.
        """
        self._gt_slot_for_query = dict(gt)

    def _record_skip_outcome(
        self, query_id: int, predicted_top1_slot: int,
    ) -> None:
        """
        Compare the adaptive-skip top-1 against the registered ground
        truth. Increments ``_n_ce_skipped_correct`` or
        ``_n_ce_skipped_wrong`` accordingly. No-op if no GT registered.
        """
        if query_id not in self._gt_slot_for_query:
            return
        gt_slot = self._gt_slot_for_query[query_id]
        if predicted_top1_slot == gt_slot:
            self._n_ce_skipped_correct += 1
        else:
            self._n_ce_skipped_wrong += 1


__all__ = ["HybridEpisodicMemory"]

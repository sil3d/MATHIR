"""
MATHIR Drop-in — Main ``MATHIRMemory`` class.

A self-contained 4-tier memory plugin that mirrors the API of the full
``mathir_lib`` package but ships as a single file. The 5 tiers are:

    1. Working      — circular buffer of the last N embeddings
    2. Episodic     — key-value store with cosine-similarity recall
    3. Semantic     — online k-means prototypes (concept clustering)
    4. Immune       — anomaly detector (distance to nearest "normal")

Plus a small KL-constrained router that blends the four contexts.

The class is intentionally framework-light: only ``torch`` and the
in-tree ``store`` / ``config`` / ``exceptions`` modules. There is no
hidden import path to the wider ``mathir_lib`` package — copy this
folder anywhere and it works.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    DEFAULT_CONFIG,
    configure,
    get_default_config,
    validate_config,
)
from .exceptions import DimensionMismatchError, StorageError
from .store import SQLiteStore


# ---------------------------------------------------------------------------
# Lightweight hybrid device manager for the drop-in package.
# This mirrors mathir_lib.hybrid_device.HybridDeviceManager but keeps the
# drop-in self-contained (no dependency on mathir_lib).
# ---------------------------------------------------------------------------

class _HybridDeviceManager:
    """Minimal device manager for the drop-in package.

    Handles CPU↔GPU transfers at tier boundaries. When ``device_map`` is
    ``None`` or empty, acts as a no-op passthrough.
    """

    def __init__(self, device_map=None, fallback="cpu"):
        self.device_map = dict(device_map) if device_map else {}
        self._fallback = torch.device(fallback)
        self._transfer_count = 0
        self._transfer_bytes = 0
        self._component_transfers = {}

    def get_device(self, component):
        return torch.device(self.device_map.get(component, self._fallback))

    def to_device(self, tensor, component):
        target = self.get_device(component)
        if tensor.device == target:
            return tensor
        self._transfer_count += 1
        self._transfer_bytes += tensor.nelement() * tensor.element_size()
        self._component_transfers[component] = (
            self._component_transfers.get(component, 0) + 1
        )
        return tensor.detach().to(target)

    def get_stats(self):
        return {
            "transfer_count": self._transfer_count,
            "transfer_bytes": self._transfer_bytes,
            "component_transfers": dict(self._component_transfers),
            "device_map": dict(self.device_map),
        }

    def is_hybrid(self):
        return bool(self.device_map)

# UniversalBridge is optional: the file exists in the package but if
# the import fails for any reason we fall back to the vanilla paths
# in MATHIRMemory.  This keeps ``import mathir_dropin`` working even
# on a slimmed-down deployment that only has memory.py + store.py.
try:  # pragma: no cover - import-time only
    from .universal_bridge import UniversalBridge
except Exception:  # pragma: no cover
    UniversalBridge = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Self-contained device detection (no dependency on mathir_lib)
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Auto-detect best available device.

    Returns ``"cuda:0"`` if a CUDA GPU is available with at least 1 GB
    free VRAM, otherwise ``"cpu"``.  This is intentionally duplicated
    from ``mathir_lib.device_utils`` so the drop-in package stays
    dependency-free.
    """
    if torch.cuda.is_available():
        try:
            free_mem = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
            if free_mem >= 1.0:
                return "cuda:0"
        except Exception:
            return "cuda:0"
    return "cpu"


def _auto_device_map() -> Dict[str, str]:
    """Generate a tier→device map for single-device execution.

    If GPU is available, all components go on GPU.  Otherwise returns
    an empty dict (CPU-only, no special mapping needed).
    """
    device = _detect_device()
    if device == "cpu":
        return {}  # Everything on CPU — no special mapping needed.
    return {
        "working": "cuda:0",
        "episodic": "cuda:0",
        "semantic": "cuda:0",
        "immune": "cuda:0",
        "router": "cuda:0",
        "input_proj": "cuda:0",
        "output_proj": "cuda:0",
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    """Generate a sortable-ish memory id like ``mem_a1b2c3d4``."""
    return "mem_" + uuid.uuid4().hex[:8]


def _infer_modality(metadata: Optional[Dict[str, Any]]) -> str:
    """Best-effort modality guess from user metadata.

    Explicit ``metadata["modality"]`` always wins. Otherwise we look at
    common content-type keys. Default is ``"text"`` — the drop-in is
    modality-agnostic and only uses the string for filtering, not for
    any pre-processing.
    """
    if not metadata:
        return "text"
    if "modality" in metadata and isinstance(metadata["modality"], str):
        return metadata["modality"]
    if "content_type" in metadata and isinstance(metadata["content_type"], str):
        return metadata["content_type"]
    if "image" in metadata or "image_url" in metadata or "pil_image" in metadata:
        return "image"
    if "audio" in metadata or "audio_url" in metadata or "waveform" in metadata:
        return "audio"
    if "video" in metadata or "video_url" in metadata:
        return "video"
    return "text"


# ===========================================================================
# Tier 1 — Working memory (circular buffer with attention retrieval)
# ===========================================================================

class _WorkingMemory(nn.Module):
    """Last N embeddings, attention-retrieved.

    Capacity: 64 by default. Update: every store(). Retrieval:
    multi-head attention over the buffer, residual-added to the query.
    """

    def __init__(self, capacity: int, dim: int, num_heads: int = 4):
        super().__init__()
        self.capacity = capacity
        self.dim = dim
        self.register_buffer("buffer", torch.zeros(capacity, dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))
        self.attention = nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )

    def store(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            flat = x.detach().reshape(-1, self.dim)
            n = flat.size(0)
            ptr = int(self.ptr.item())
            if n >= self.capacity:
                self.buffer.copy_(flat[-self.capacity:])
                self.ptr.fill_(0)
                self.count.fill_(self.capacity)
                return
            end = ptr + n
            if end <= self.capacity:
                self.buffer[ptr:end] = flat
            else:
                first = self.capacity - ptr
                self.buffer[ptr:] = flat[:first]
                self.buffer[: n - first] = flat[first:]
            self.ptr.fill_((ptr + n) % self.capacity)
            self.count.fill_(min(self.count.item() + n, self.capacity))

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        c = int(self.count.item())
        if c == 0:
            return torch.zeros_like(query)
        ctx = self.buffer[:c].unsqueeze(0).expand(query.size(0), -1, -1)
        out, _ = self.attention(query.unsqueeze(1), ctx, ctx)
        return out.squeeze(1)

    def reset(self) -> None:
        self.buffer.zero_()
        self.ptr.fill_(0)
        self.count.fill_(0)

    @property
    def usage(self) -> int:
        return int(self.count.item())


# ===========================================================================
# Tier 2 — Episodic memory (key-value cosine store)
# ===========================================================================

class _EpisodicMemory(nn.Module):
    """Top-k cosine similarity over a circular key-value store.

    Capacity: 1000 by default. Embeddings are stored verbatim in the
    *internal* projected space; raw embeddings are stored in SQLite for
    the persist-and-restore flow.
    """

    def __init__(self, capacity: int, dim: int, key_dim: int = 64):
        super().__init__()
        self.capacity = capacity
        self.dim = dim
        self.key_dim = key_dim
        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, dim))
        self.register_buffer("stability", torch.ones(capacity))
        self.register_buffer("recall_count", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))
        self.encoder = nn.Linear(dim, key_dim)

    def store(self, x: torch.Tensor) -> int:
        """Store ``x`` (mean-pooled if batched). Returns the slot index."""
        with torch.no_grad():
            pooled = x.detach().mean(0)
            key = self.encoder(pooled)
            idx = int(self.ptr.item()) % self.capacity
            self.keys[idx] = key
            self.values[idx] = pooled
            self.stability[idx] = 1.0
            self.recall_count[idx] = 0
            self.ptr.fill_((idx + 1) % self.capacity)
            self.count.fill_(min(self.count.item() + 1, self.capacity))
            return idx

    def search(self, query: torch.Tensor, k: int = 5) -> List[Dict[str, Any]]:
        """Return ``[{index, similarity, value, stability, recall_count}, ...]``."""
        c = int(self.count.item())
        if c == 0:
            return []
        with torch.no_grad():
            key = self.encoder(query)
            sims = F.cosine_similarity(
                key.unsqueeze(1), self.keys[:c].unsqueeze(0), dim=-1
            )
            k_eff = min(k, c)
            top = sims.topk(k_eff, dim=1)
            results: List[Dict[str, Any]] = []
            for j in range(k_eff):
                idx = int(top.indices[0, j].item())
                results.append({
                    "index": idx,
                    "similarity": float(top.values[0, j].item()),
                    "value": self.values[idx].cpu(),
                    "stability": float(self.stability[idx].item()),
                    "recall_count": int(self.recall_count[idx].item()),
                })
            return results

    def retrieve(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Mean of top-k values, used internally by ``perceive()``."""
        c = int(self.count.item())
        if c < k:
            return torch.zeros_like(query)
        with torch.no_grad():
            key = self.encoder(query)
            sims = F.cosine_similarity(
                key.unsqueeze(1), self.keys[:c].unsqueeze(0), dim=-1
            )
            top = sims.topk(min(k, c), dim=1).indices
            return self.values[top].mean(1)

    def bump_recall(self, indices: List[int]) -> None:
        """Ebbinghaus-style stability boost on access."""
        with torch.no_grad():
            for idx in indices:
                if 0 <= idx < self.capacity:
                    self.recall_count[idx] += 1
                    # Stability grows with each recall, capped at 1.0 to avoid
                    # unbounded float blow-up over long-lived processes.
                    self.stability[idx] = min(self.stability[idx] * 1.5, 1.0)

    def forget(self, threshold: float) -> List[int]:
        """Prune memories with low average intra-set similarity.

        Returns the list of *original* slot indices (in ``[0, count)``)
        that survived the prune, in their new compacted order. "Low
        similarity" means below ``threshold`` against the centroid of
        the buffer — this is a crude utility heuristic; the user can
        override with custom logic.

        The caller uses the returned indices to keep any external
        slot→id mapping (and the SQLite store) in sync. An empty buffer
        or a no-op prune returns the identity mapping ``list(range(c))``.
        """
        c = int(self.count.item())
        if c < 2:
            return list(range(c))
        with torch.no_grad():
            sims = F.cosine_similarity(
                self.keys[:c].unsqueeze(1),
                self.keys[:c].unsqueeze(0),
                dim=-1,
            )
            utility = sims.mean(dim=1)
            keep_mask = utility > threshold
            keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
            n_kept = int(keep_idx.numel())
            if n_kept == c:
                return list(range(c))
            if n_kept == 0:
                # Keep at least one to avoid empty store.
                keep_idx = torch.tensor([0], dtype=torch.long)
                n_kept = 1
            self.keys[:n_kept] = self.keys[:c][keep_idx]
            self.values[:n_kept] = self.values[:c][keep_idx]
            self.stability[:n_kept] = self.stability[:c][keep_idx]
            self.recall_count[:n_kept] = self.recall_count[:c][keep_idx]
            self.count.fill_(n_kept)
            self.ptr.fill_(n_kept % self.capacity)
            return [int(i) for i in keep_idx.tolist()]

    def reset(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.stability.fill_(1.0)
        self.recall_count.zero_()
        self.ptr.fill_(0)
        self.count.fill_(0)

    @property
    def usage(self) -> int:
        return int(self.count.item())


# ===========================================================================
# Tier 3 — Semantic memory (online k-means prototypes)
# ===========================================================================

class _SemanticMemory(nn.Module):
    """K prototypes updated by online k-means, retrieved by argmax.

    Capacity: 256 prototypes. Update: incremental with rate 0.01.
    """

    def __init__(self, num_prototypes: int, dim: int, proj_dim: int = 64,
                 update_rate: float = 0.01):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.dim = dim
        self.proj_dim = proj_dim
        self.update_rate = update_rate
        # Kaiming-ish init so cosine similarities start low.
        proto = torch.randn(num_prototypes, proj_dim) * 0.1
        self.register_buffer("prototypes", proto)
        self.register_buffer("usage", torch.zeros(num_prototypes))
        self.down = nn.Linear(dim, proj_dim)
        self.up = nn.Linear(proj_dim, dim)

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            proj = self.down(query)
            sims = F.cosine_similarity(
                proj.unsqueeze(1), self.prototypes.unsqueeze(0), dim=-1
            )
            idx = sims.argmax(dim=1)
        return self.up(self.prototypes[idx])

    def update(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            proj = self.down(x)
            sims = F.cosine_similarity(
                proj.unsqueeze(1), self.prototypes.unsqueeze(0), dim=-1
            )
            idx = sims.argmax(dim=1)
            for i in range(x.size(0)):
                p = self.prototypes[idx[i]]
                self.prototypes[idx[i]] = (1 - self.update_rate) * p + self.update_rate * proj[i].detach()
                self.usage[idx[i]] += 1

    def reset(self) -> None:
        self.prototypes.copy_(torch.randn(self.num_prototypes, self.proj_dim) * 0.1)
        self.usage.zero_()

    def stats(self) -> Dict[str, Any]:
        used = int((self.usage > 0).sum().item())
        return {
            "num_prototypes": self.num_prototypes,
            "used_prototypes": used,
            "avg_usage": float(self.usage.mean().item()),
            "max_usage": float(self.usage.max().item()),
        }


# ===========================================================================
# Tier 4 — Immune memory (anomaly detector)
# ===========================================================================

class _ImmuneMemory(nn.Module):
    """Distance to nearest "normal" sample.

    Capacity: 100 reference vectors. Returns the *score* (lower is
    more normal). ``threshold`` controls what counts as anomalous.
    """

    def __init__(self, capacity: int, dim: int, threshold: float = 2.0):
        super().__init__()
        self.capacity = capacity
        self.dim = dim
        self.threshold = threshold
        self.register_buffer("bank", torch.zeros(capacity, dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))

    def store(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            pooled = x.detach().mean(0)
            idx = int(self.ptr.item()) % self.capacity
            self.bank[idx] = pooled
            self.ptr.fill_((idx + 1) % self.capacity)
            self.count.fill_(min(self.count.item() + 1, self.capacity))

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous score = min L2 distance to bank. Shape: ``[B]``."""
        c = int(self.count.item())
        if c < 5:
            return torch.zeros(x.size(0), device=x.device)
        with torch.no_grad():
            d = torch.cdist(x, self.bank[:c])
            return d.min(dim=1).values

    def recognize(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` if anomalous, zeros otherwise (shaped ``[B, D]``)."""
        s = self.anomaly_score(x)
        mask = (s > self.threshold).float().unsqueeze(-1)
        return mask * x

    def reset(self) -> None:
        self.bank.zero_()
        self.ptr.fill_(0)
        self.count.fill_(0)

    @property
    def usage(self) -> int:
        return int(self.count.item())


# ===========================================================================
# KL-constrained router
# ===========================================================================

class _KLRouter(nn.Module):
    """4-way softmax router with a KL term to a uniform prior.

    The KL penalty prevents the router from collapsing onto a single
    tier, which would defeat the purpose of having four specialized
    stores. Coefficient is adapted online (not in the drop-in version
    for simplicity).
    """

    def __init__(self, dim: int, num_tiers: int = 4, hidden: int = 128,
                 kl_coefficient: float = 0.01):
        super().__init__()
        self.num_tiers = num_tiers
        self.kl_coefficient = kl_coefficient
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, num_tiers)
        )
        uniform = torch.ones(num_tiers) / num_tiers
        self.register_buffer("uniform_prior", uniform)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.net(x)
        weights = F.softmax(logits, dim=-1)
        kl = F.kl_div(
            F.log_softmax(logits, dim=-1),
            self.uniform_prior.unsqueeze(0).expand_as(weights),
            reduction="batchmean",
        )
        return {"weights": weights, "kl_loss": self.kl_coefficient * kl}


# ===========================================================================
# Main class
# ===========================================================================

class MATHIRMemory(nn.Module):
    """Drop-in MATHIR memory plugin.

    Parameters
    ----------
    embedding_dim:
        The dimensionality of incoming embeddings. This is the *only*
        required argument and the most common source of errors — see
        :class:`DimensionMismatchError`.
    config:
        Optional dict produced by :func:`configure`. Defaults are used
        for any missing key.
    db_path:
        Path to a SQLite database file. ``None`` keeps everything in
        memory (fastest, no persistence). Pass ``":memory:"`` for an
        in-memory SQLite store (useful for tests that need FTS5
        without a real file).

    Example
    -------
    >>> import torch
    >>> from mathir_dropin import MATHIRMemory
    >>> mem = MATHIRMemory(embedding_dim=384, db_path="agent.db")
    >>> mid = mem.store(torch.randn(1, 384), {"text": "hello"})
    >>> hits = mem.recall(torch.randn(1, 384), k=3)
    """

    def __init__(
        self,
        embedding_dim: int,
        config: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        device_map: Any = "auto",
    ):
        super().__init__()

        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be a positive int, got {embedding_dim!r}"
            )
        self.embedding_dim = embedding_dim

        # ---- device auto-detection --------------------------------------
        # When ``device_map="auto"``, we detect the best device and move
        # the *entire* module to it.  This avoids cross-device tensor/
        # parameter mismatches that a mixed device map would cause.
        _target_device: Optional[torch.device] = None
        if device_map == "auto":
            _detected = _detect_device()
            if _detected != "cpu":
                _target_device = torch.device(_detected)
            device_map = None  # No mixed-device map; module-level .to() below.
        # ``None`` means single-device passthrough (existing behaviour).

        # ---- config -----------------------------------------------------
        cfg = configure(config)
        cfg["memory"]["embedding_dim"] = embedding_dim
        if db_path is None:
            # Caller asked explicitly for no file → force RAM-only storage.
            # Without this, ``storage.type`` would stay "sqlite" and
            # we'd silently start writing to the default ``mathir.db``
            # in the CWD, which both violates the caller's intent and
            # can leak rows between test runs.
            cfg["storage"]["type"] = "memory"
        else:
            cfg["storage"]["db_path"] = db_path
        validate_config(cfg)
        self.config = cfg
        self._cfg_mem = cfg["memory"]
        self._cfg_storage = cfg["storage"]
        self._cfg_router = cfg["router"]

        # ---- hybrid device manager -----------------------------------
        self.device_manager = _HybridDeviceManager(
            device_map,
            fallback=str(_target_device) if _target_device is not None else "cpu",
        )

        # ---- projections ----------------------------------------------
        internal = self._cfg_mem["internal_dim"]
        self.input_proj = nn.Linear(embedding_dim, internal)
        self.output_proj = nn.Linear(internal, embedding_dim)
        self.layer_norm = nn.LayerNorm(internal) if cfg["perception"]["use_layer_norm"] else nn.Identity()

        # ---- tiers -----------------------------------------------------
        self.working = _WorkingMemory(
            capacity=self._cfg_mem["working_capacity"],
            dim=internal,
        )
        self.episodic = _EpisodicMemory(
            capacity=self._cfg_mem["episodic_capacity"],
            dim=internal,
        )
        self.semantic = _SemanticMemory(
            num_prototypes=self._cfg_mem["semantic_prototypes"],
            dim=internal,
        )
        self.immune = _ImmuneMemory(
            capacity=self._cfg_mem["immunological_capacity"],
            dim=internal,
            threshold=self._cfg_mem["anomaly_threshold"],
        )

        # ---- router ----------------------------------------------------
        if self._cfg_router["type"] == "kl_constrained":
            self.router = _KLRouter(
                dim=internal,
                num_tiers=4,
                hidden=self._cfg_router["hidden_dim"],
                kl_coefficient=self._cfg_router["kl_coefficient"],
            )
        else:
            # Uniform router: equal weights, no learning.
            self.router = _UniformRouter(num_tiers=4)

        # ---- storage ---------------------------------------------------
        self._store: Optional[SQLiteStore] = None
        if self._cfg_storage["type"] == "sqlite":
            self._store = SQLiteStore(self._cfg_storage["db_path"])

        # ---- bookkeeping -----------------------------------------------
        self._tier_for_id: Dict[str, str] = {}  # memory_id -> tier name
        # Parallel to the episodic circular buffer: slot index -> the
        # memory_id stored there (or None for an empty/overwritten slot).
        # This is what lets forget() / save() keep SQLite in sync, since
        # the episodic tier itself only holds anonymous vectors.
        self._episodic_ids: List[Optional[str]] = [None] * self.episodic.capacity
        # Rows stored while auto_save is OFF wait here until save() flushes
        # them. Each entry is the full kwargs dict for SQLiteStore.insert.
        self._pending_writes: List[Dict[str, Any]] = []
        self._auto_save = bool(self._cfg_storage.get("auto_save", True))
        self._created_at = time.time()
        # A single RLock covers every in-memory mutation. PyTorch's
        # buffer arithmetic is not atomic across threads (read-modify-
        # write of ``self.ptr`` etc.), so concurrent ``store()`` calls
        # would otherwise lose writes under contention.
        self._op_lock = threading.RLock()
        # Provider tracking for embeddings
        self.provider = provider
        self.model = model
        # Lazily-built UniversalBridge instance for cross-provider and
        # cross-lingual recall.  We initialise the field as None and
        # build it on first use of :meth:`universal_recall` so the
        # constructor stays cheap (and the bridge is not constructed
        # for users who never call the new API).
        self._bridge: Optional["UniversalBridge"] = None

        # ---- move entire module to target device if auto-detected GPU ---
        if _target_device is not None:
            self.to(_target_device)

    # ------------------------------------------------------------------
    # Embedding dimension check
    # ------------------------------------------------------------------

    def _check_dim(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.size(-1) != self.embedding_dim:
            raise DimensionMismatchError(
                expected=self.embedding_dim,
                got=int(embedding.size(-1)),
                where="input embedding",
            )
        return embedding

    # ------------------------------------------------------------------
    # Core API: perceive / store / recall / forget / get_stats
    # ------------------------------------------------------------------

    def perceive(
        self,
        embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process an embedding and return the enhanced context.

        The returned dict is **the same shape regardless of modality** —
        that's the whole point of MATHIR. The four tiers contribute
        their context, the router blends them, and the result is
        projected back to ``embedding_dim`` for downstream use.

        Returns
        -------
        dict with keys:
            enhanced_embedding : torch.Tensor  [B, embedding_dim]
            modality           : str
            router_weights     : torch.Tensor  [B, 4]
            anomaly_score      : torch.Tensor  [B]
            memory_id          : str or None   (set if store=True)
        """
        x_in = self._check_dim(embedding)
        modality = _infer_modality(metadata)

        # Transfer input to input_proj device if hybrid
        x_in_dev = self.device_manager.to_device(x_in, "input_proj")
        x = self.input_proj(x_in_dev)

        # All tier reads/writes happen under the same lock that store()
        # uses. Without it, the working buffer's circular pointer could
        # be corrupted by a store() racing against this perceive()'s
        # final working.store(x) below — silently losing writes.
        with self._op_lock:
            # Retrieve from each tier. Working context comes from
            # *before* the current input is added (otherwise we'd always
            # trivially retrieve the input itself with similarity 1.0).
            w_ctx = self.working.retrieve(
                self.device_manager.to_device(x, "working"),
            )
            e_ctx = self.episodic.retrieve(
                self.device_manager.to_device(x, "episodic"), k=3,
            )
            s_ctx = self.semantic.retrieve(
                self.device_manager.to_device(x, "semantic"),
            )
            i_ctx = self.immune.recognize(
                self.device_manager.to_device(x, "immune"),
            )

            # Transfer all contexts to router device for fusion
            x_router = self.device_manager.to_device(x, "router")
            w_ctx = self.device_manager.to_device(w_ctx, "router")
            e_ctx = self.device_manager.to_device(e_ctx, "router")
            s_ctx = self.device_manager.to_device(s_ctx, "router")
            i_ctx = self.device_manager.to_device(i_ctx, "router")

            # Route and blend.
            routed = self.router(x_router)
            weights = routed["weights"]  # [B, 4]
            w = weights.unsqueeze(-1)     # [B, 4, 1] for broadcasting
            ctx_stack = torch.stack([w_ctx, e_ctx, s_ctx, i_ctx], dim=1)  # [B, 4, D]
            fused = (w * ctx_stack).sum(dim=1)  # [B, D]

            # Residual + norm.
            if self.config["perception"]["use_residual"]:
                fused = fused + x_router
            fused = self.layer_norm(fused)

            enhanced = self.output_proj(
                self.device_manager.to_device(fused, "output_proj"),
            )

            # Anomaly score in the original projected space (uses the
            # immune tier's continuous score, not the thresholded mask).
            anomaly = self.immune.anomaly_score(
                self.device_manager.to_device(x, "immune"),
            )

            # Also write to the working buffer so future calls see it.
            self.working.store(self.device_manager.to_device(x, "working"))

        return {
            "enhanced_embedding": enhanced,
            "modality": modality,
            "router_weights": weights,
            "anomaly_score": anomaly,
            "memory_id": None,
        }

    def store(
        self,
        embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        tier: str = "episodic",
        persist: Optional[bool] = None,
        extra_providers: Optional[Dict[str, tuple]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Persist an embedding to memory.

        Parameters
        ----------
        embedding:
            ``[B, embedding_dim]`` or ``[embedding_dim]``.
        metadata:
            Free-form dict. ``metadata["text"]`` (or ``"modality_text"``)
            is what FTS5 will search against; other entries are stored
            as JSON.
        tier:
            Which tier owns the write. Default ``"episodic"`` (long-term
            experiences). The other tiers update themselves internally
            regardless.
        persist:
            Override the config-level ``auto_save``. ``True`` forces
            a write to SQLite, ``False`` skips it.
        extra_providers:
            Optional dict of additional provider embeddings to store.
            Key is provider name (e.g., "openai", "cohere"), value is a
            tuple of (embedding_tensor, model_name). This allows storing
            embeddings for multiple providers at insert time so you never
            need to re-embed on provider switch.

        Returns
        -------
        memory_id : str
            Unique id usable with :meth:`recall` / :meth:`forget`.
        """
        x_in = self._check_dim(embedding)
        modality = _infer_modality(metadata)

        # Transfer input to input_proj device if hybrid
        x_in_dev = self.device_manager.to_device(x_in, "input_proj")
        x = self.input_proj(x_in_dev)

        memory_id = _new_id()
        modality_text = ""
        if metadata:
            for k in ("text", "modality_text", "content", "transcript", "caption"):
                v = metadata.get(k)
                if isinstance(v, str):
                    modality_text = v
                    break
        if metadata is None:
            metadata = {}
        metadata = {**metadata, "modality": modality}

        # Update tiers under a single lock so concurrent stores don't
        # race on the circular-buffer ``ptr`` counters.
        with self._op_lock:
            self.working.store(self.device_manager.to_device(x, "working"))
            slot = self.episodic.store(self.device_manager.to_device(x, "episodic"))
            self.semantic.update(self.device_manager.to_device(x, "semantic"))
            self.immune.store(self.device_manager.to_device(x, "immune"))
            self._tier_for_id[memory_id] = tier
            # The slot the episodic buffer just wrote to now belongs to
            # this id; whatever id used to live there has been evicted
            # from the in-memory tier (its SQLite row, if any, remains
            # the source of truth and is intentionally untouched).
            if 0 <= slot < len(self._episodic_ids):
                self._episodic_ids[slot] = memory_id

        # The row we would persist (used now if auto_save, or buffered
        # for a later save()).
        row_provider = provider if provider is not None else (self.provider or "unknown")
        row_model = model if model is not None else (self.model or "unknown")
        row = {
            "memory_id": memory_id,
            "embedding": x_in.detach().cpu().reshape(-1),
            "metadata": metadata,
            "modality": modality,
            "modality_text": modality_text,
            "tier": tier,
            "provider": row_provider,
            "model": row_model,
        }

        # SQLite persistence (already serialised internally).
        do_persist = self._auto_save if persist is None else bool(persist)
        if self._store is not None:
            if do_persist:
                try:
                    self._store.insert(**row)
                    # Store extra provider embeddings if provided
                    if extra_providers:
                        for prov_name, (prov_emb, prov_model) in extra_providers.items():
                            self._store.insert_embedding(
                                memory_id=memory_id,
                                provider=prov_name,
                                model=prov_model,
                                embedding=prov_emb,
                            )
                except StorageError as e:
                    # Persistence is best-effort: a write failure must not
                    # lose the in-memory state. We re-raise only if the
                    # user explicitly asked for persistence.
                    if persist:
                        raise
                    # Otherwise keep it pending so a later save() can retry.
                    with self._op_lock:
                        self._pending_writes.append(row)
                    metadata["_persist_error"] = str(e)
            else:
                # auto_save off (or persist=False): hold the row until the
                # caller explicitly flushes with save().
                with self._op_lock:
                    self._pending_writes.append(row)

        return memory_id

    def recall(
        self,
        query_embedding: torch.Tensor,
        k: int = 5,
        modality: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find the ``k`` most similar memories.

        Returns a list of dicts sorted by descending similarity. Each
        dict has::

            {
              "memory_id":   str,
              "similarity":  float,    # cosine in [-1, 1]
              "metadata":    dict,
              "embedding":   torch.Tensor,   # original 1-D embedding
              "tier":        str,
              "modality":    str,
            }

        Parameters
        ----------
        provider:
            Optional provider name (e.g., "openai", "cohere"). When specified,
            the search uses that provider's stored embeddings for cosine
            similarity, falling back to primary embeddings if a memory
            doesn't have an embedding for that provider.
        model:
            Optional model name for the provider. If specified with provider,
            used to filter results to that model. Generally left None for
            broad similarity search.

        Strategy:
            1. Always check the live in-memory store (covers anything
               stored in this session).
            2. If SQLite is enabled, also search the persisted rows
               and merge by memory_id. Persisted results include
               everything from previous sessions.
        """
        q = self._check_dim(query_embedding)
        if k <= 0:
            return []

        # The in-memory episodic tier and the SQLite store can hold the
        # same data; if we hit both we'd return duplicates (the in-memory
        # side uses synthetic "live_<idx>" ids that never collide with
        # SQLite's "mem_xxx" ids). When SQLite is configured we treat
        # it as the source of truth and skip the in-memory scan; the
        # in-memory search is only the fallback for ``db_path=None``.
        if self._store is not None:
            try:
                if provider:
                    # Use multi-embedding search for provider-specific embeddings
                    sql_results = self._store.search_by_embedding_multi(
                        query_embedding=q.detach().cpu().reshape(-1),
                        provider=provider,
                        k=k,
                        modality=modality,
                    )
                else:
                    sql_results = self._store.search_by_embedding(
                        query_embedding=q.detach().cpu().reshape(-1),
                        k=k,
                        modality=modality,
                    )
            except StorageError:
                sql_results = []
            # Bump recall counters and return.
            for r in sql_results:
                try:
                    self._store.bump_recall(r["memory_id"])
                except StorageError:
                    pass
            # The SQLite path already filtered by modality, but apply
            # again defensively in case the storage layer's filter is
            # ever weakened.
            if modality is not None:
                sql_results = [r for r in sql_results if r.get("modality") == modality]
            return sql_results[:k]

        # --- In-memory only path (db_path=None) ---
        q_proj = self.input_proj(self.device_manager.to_device(q, "input_proj"))
        live_hits = self.episodic.search(
            self.device_manager.to_device(q_proj, "episodic"), k=k,
        )
        live_results: List[Dict[str, Any]] = []
        for h in live_hits:
            live_results.append({
                "memory_id":  f"live_{h['index']}",
                "similarity": h["similarity"],
                "metadata":   {"_live": True},
                "embedding":  self.device_manager.to_device(h["value"], "cpu").cpu(),
                "tier":       "episodic",
                "modality":   "text",
                "stability":  h["stability"],
                "recall_count": h["recall_count"],
            })
        for r in live_results:
            idx_str = str(r["memory_id"]).replace("live_", "")
            try:
                self.episodic.bump_recall([int(idx_str)])
            except (ValueError, IndexError):
                pass
        return live_results[:k]

    def recall_text(
        self,
        query_text: str,
        k: int = 5,
        modality: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """BM25 text search via SQLite FTS5.

        Returns the same dict shape as :meth:`recall` so callers can
        mix embedding and text retrieval transparently. ``similarity``
        here is the BM25 cost negated (higher = more relevant) so
        downstream sort code keeps working.
        """
        if not self._store:
            return []
        try:
            rows = self._store.search_by_text(query_text, k=k, modality=modality)
        except StorageError:
            return []
        # Normalize BM25 cost to a similarity-like score in [0, 1].
        # Lower bm25() = more relevant, so negate + min-max scale.
        if rows:
            costs = [abs(float(r.get("rank", 0.0)) or 0.0) for r in rows]
            lo, hi = min(costs), max(costs)
            span = max(hi - lo, 1e-9)
            for r, c in zip(rows, costs):
                r["similarity"] = 1.0 - (c - lo) / span
        return rows

    # ------------------------------------------------------------------
    # Universal cross-provider / cross-lingual recall
    # ------------------------------------------------------------------

    def available_providers(self) -> List[str]:
        """List of distinct provider names stored in ``memory_embeddings``.

        Returns an empty list if the SQLite store is disabled (in-memory
        mode) or no provider embeddings have been written yet.
        """
        if self._store is None:
            return []
        try:
            return self._store.list_providers()
        except StorageError:
            return []

    def universal_recall(
        self,
        query: str,
        query_embedding: Optional[torch.Tensor] = None,
        k: int = 5,
        provider: Optional[str] = None,
        modality: Optional[str] = None,
        cross_lingual: bool = True,
        use_recall_boost: bool = True,
    ) -> List[Dict[str, Any]]:
        """Universal cross-provider, cross-lingual, conversational-safe recall.

        This is the new high-level entry point that fixes the three
        shortcomings of :meth:`recall` / :meth:`recall_text`:

        1. **Conversational queries** — :meth:`recall_text` returns
           zero results on questions like ``"What do you know about
           python closures?"`` because FTS5's porter+unicode61
           tokenizer drops "what", "do", "you", "about" and the
           remaining tokens don't always match.  Here we **expand the
           query** (raw / lowercased / stopword-stripped / stemmed /
           transliterated / single-token) and run FTS5 for each
           variant.

        2. **Cross-provider fallback** — if the requested provider
           has no stored embeddings, we walk a fallback chain:
           ``requested -> primary -> other providers with matching
           dim -> text search (via expand_query) -> primary
           embeddings as a last resort``.

        3. **Cross-lingual matching** — Jaccard similarity over
           character n-grams between the query and each candidate's
           ``modality_text``.  A French query ``"clotures en python"``
           can find the English memory ``"python closures"``.

        4. **Recall-count boost** — frequently-recalled memories get
           a small logarithmic score boost so they are more likely
           to surface again (``1 + α · log(1 + recall_count)`` with
           ``α = 0.15``).

        Parameters
        ----------
        query:
            The user's natural-language query.  Used for text search
            and (if ``cross_lingual``) for trigram re-ranking.
        query_embedding:
            Optional embedding of the query.  When provided, used
            for the embedding channel; when absent, the hybrid ranker
            relies on text + cross-lingual + recall_count only.
        k:
            Maximum number of results to return.
        provider:
            Preferred provider for embedding lookup.  If unavailable,
            the fallback chain is used.
        modality:
            Optional modality filter passed through to FTS5 / cosine.
        cross_lingual:
            Toggle the character-n-gram re-ranking channel.
        use_recall_boost:
            Toggle the recall-count boost in the final score.

        Returns
        -------
        List of dicts sorted by descending ``final_score``.  Each
        dict has the standard :meth:`recall` keys plus::

            final_score     : float
            text_score      : float
            emb_score       : float
            xl_score        : float
            recall_boost    : float
            source          : str  ('text' | 'embedding' | 'hybrid')
        """
        if UniversalBridge is None:
            # Bridge module failed to import — degrade to the union of
            # the existing methods, which is the original behaviour.
            warnings_extra: List[Dict[str, Any]] = []
            if query:
                warnings_extra = self.recall_text(query, k=k, modality=modality)
            if query_embedding is not None:
                emb_hits = self.recall(query_embedding, k=k, modality=modality, provider=provider)
                for h in emb_hits:
                    h["final_score"] = h.get("similarity", 0.0)
                    h["source"] = "embedding"
                warnings_extra = warnings_extra + [
                    h for h in emb_hits
                    if h["memory_id"] not in {w["memory_id"] for w in warnings_extra}
                ]
            return warnings_extra[:k]

        if self._bridge is None:
            self._bridge = UniversalBridge()

        # 1. Text channel: expand the query and merge FTS5 results.
        text_results: List[Dict[str, Any]] = []
        seen_text_ids: set = set()
        if query and self._store is not None:
            variants = self._bridge.expand_query(query)
            for variant in variants:
                if not variant:
                    continue
                try:
                    rows = self._store.search_by_text(variant, k=k, modality=modality)
                except StorageError:
                    continue
                for r in rows:
                    if r["memory_id"] in seen_text_ids:
                        continue
                    seen_text_ids.add(r["memory_id"])
                    text_results.append(r)
                # Stop once we have enough candidates; expand only if
                # we still have room.  This bounds the worst case at
                # ``expansion_variants * k`` rows.
                if len(text_results) >= k * 2:
                    break

        # 2. Embedding channel: walk the provider fallback chain.
        emb_results: List[Dict[str, Any]] = []
        if query_embedding is not None and self._store is not None:
            try:
                q_vec = self._check_dim(query_embedding).detach().cpu().reshape(-1)
            except DimensionMismatchError:
                q_vec = None
            if q_vec is not None:
                chain = self._bridge.provider_fallback_chain(
                    requested=provider,
                    available=self.available_providers(),
                    primary=self.provider or "primary",
                )
                for prov in chain:
                    try:
                        rows = self._store.search_by_embedding_multi(
                            query_embedding=q_vec,
                            provider=prov,
                            k=k,
                            modality=modality,
                        )
                    except StorageError:
                        rows = []
                    if rows:
                        emb_results = rows
                        break

        # 3. Cross-lingual text similarity fallback: if the text and
        #    embedding channels are both empty, scan *all* memories
        #    with the Jaccard scorer.  Bounded by store size; the
        #    n-gram method is O(N · |text|).
        if not text_results and not emb_results and query and self._store is not None:
            try:
                all_ids = self._store.all_ids()
            except StorageError:
                all_ids = []
            for mid in all_ids:
                try:
                    row = self._store.get(mid)
                except StorageError:
                    continue
                if row is None:
                    continue
                txt = row.get("modality_text") or ""
                if not txt:
                    md = row.get("metadata") or {}
                    for k_ in ("text", "modality_text", "content", "transcript", "caption"):
                        v = md.get(k_)
                        if isinstance(v, str) and v:
                            txt = v
                            break
                if not txt:
                    continue
                sim = self._bridge.text_similarity(query, txt)
                if sim > 0.0:
                    text_results.append({
                        "memory_id": mid,
                        "similarity": sim,
                        "modality": row.get("modality", "text"),
                        "metadata": row.get("metadata", {}) or {},
                        "modality_text": txt,
                        "timestamp": row.get("timestamp", 0.0),
                        "tier": row.get("tier", "episodic"),
                        "stability": row.get("stability", 1.0),
                        "recall_count": row.get("recall_count", 0),
                    })

        # 4. Hybrid ranking.
        ranked = self._bridge.hybrid_recall(
            query=query or "",
            embedding=query_embedding,
            k=k,
            provider=provider,
            text_candidates=text_results or None,
            embedding_candidates=emb_results or None,
            cross_lingual=cross_lingual,
        )
        if not use_recall_boost:
            for entry in ranked:
                entry["final_score"] -= entry.get("recall_boost", 0.0)
                entry["recall_boost"] = 0.0
            ranked.sort(key=lambda e: e["final_score"], reverse=True)

        # 5. Source labelling + re-emit in the standard recall shape.
        text_ids = {r["memory_id"] for r in text_results}
        emb_ids = {r["memory_id"] for r in emb_results}
        for entry in ranked:
            in_text = entry["memory_id"] in text_ids
            in_emb = entry["memory_id"] in emb_ids
            if in_text and in_emb:
                entry["source"] = "hybrid"
            elif in_text:
                entry["source"] = "text"
            else:
                entry["source"] = "embedding"
            # Ensure the "similarity" key the rest of MATHIR expects is
            # present, mapped from the final score so downstream sort
            # code keeps working.
            entry.setdefault("similarity", float(entry["final_score"]))

        # 6. Bump recall counters for the top-k results so the boost
        #    is self-reinforcing (Ebbinghaus spaced repetition).
        if self._store is not None:
            for entry in ranked:
                try:
                    self._store.bump_recall(entry["memory_id"])
                except StorageError:
                    pass

        return ranked

    def forget(self, threshold: float = 0.1) -> int:
        """Prune low-utility episodic memories.

        Returns the number of memories dropped. ``threshold`` is the
        minimum average cosine similarity to the rest of the buffer;
        anything below is considered noise and removed.
        """
        with self._op_lock:
            c = self.episodic.usage
            # Ids currently live in the episodic buffer, before pruning.
            active_ids = {
                self._episodic_ids[i]
                for i in range(c)
                if self._episodic_ids[i] is not None
            }
            keep_idx = self.episodic.forget(threshold)
            # Rebuild the slot→id map to match the compacted buffer.
            survivors = [self._episodic_ids[i] for i in keep_idx]
            new_ids: List[Optional[str]] = [None] * len(self._episodic_ids)
            new_ids[: len(survivors)] = survivors
            self._episodic_ids = new_ids
            surviving_ids = {mid for mid in survivors if mid is not None}
            dropped = c - len(keep_idx)
            # Ids that were live and got pruned must also leave SQLite so
            # recall() (which trusts SQLite) stops returning them. Rows
            # already evicted by buffer wraparound are NOT in active_ids,
            # so their SQLite entries are correctly left untouched.
            pruned_ids = active_ids - surviving_ids

        if self._store is not None and pruned_ids:
            for mid in pruned_ids:
                try:
                    self._store.delete(mid)
                except StorageError:
                    pass
        # Also drop any not-yet-flushed pending rows for pruned ids.
        if pruned_ids and self._pending_writes:
            with self._op_lock:
                self._pending_writes = [
                    r for r in self._pending_writes
                    if r["memory_id"] not in pruned_ids
                ]
        return dropped

    def get_stats(self) -> Dict[str, Any]:
        """Snapshot of every tier + the router + storage.

        Returned structure::

            {
              "embedding_dim":   int,
              "internal_dim":    int,
              "created_at":      float,
              "tier_working":    {"usage": int, "capacity": int},
              "tier_episodic":   {"usage": int, "capacity": int, ...},
              "tier_semantic":   {...},
              "tier_immune":     {...},
              "router":          {"type": str, "kl_coefficient": float},
              "storage":         {"type": str, "db_path": str|None, "row_count": int},
            }
        """
        stats: Dict[str, Any] = {
            "embedding_dim": self.embedding_dim,
            "internal_dim": int(self._cfg_mem["internal_dim"]),
            "created_at": self._created_at,
            "tier_working": {
                "usage": self.working.usage,
                "capacity": self.working.capacity,
            },
            "tier_episodic": {
                "usage": self.episodic.usage,
                "capacity": self.episodic.capacity,
            },
            "tier_semantic": self.semantic.stats(),
            "tier_immune": {
                "usage": self.immune.usage,
                "capacity": self.immune.capacity,
                "threshold": self.immune.threshold,
            },
            "router": {
                "type": self._cfg_router["type"],
                "kl_coefficient": self._cfg_router["kl_coefficient"],
            },
            "storage": {
                "type": self._cfg_storage["type"],
                "db_path": self._cfg_storage.get("db_path"),
                "row_count": (
                    self._store.count() if self._store is not None else 0
                ),
            },
            "hybrid_device": self.device_manager.get_stats(),
        }
        return stats

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Force-persist in-memory episodic rows to SQLite.

        The default config auto-saves on every ``store()``; this method
        is for callers that disabled ``auto_save`` and want to flush
        at a controlled checkpoint.
        """
        if self._store is None:
            return
        # Flush everything that was buffered while auto_save was off.
        # Hold _op_lock for the *entire* swap + flush so a concurrent
        # save() cannot race the failed-row re-queue, and a concurrent
        # store() cannot append to _pending_writes mid-flush. Releasing
        # the lock between the swap and the loop previously let another
        # save() observe (and double-insert) the same pending rows.
        with self._op_lock:
            pending = self._pending_writes
            self._pending_writes = []
            for row in pending:
                try:
                    self._store.insert(**row)
                except StorageError:
                    # Keep it pending so a future save() can retry.
                    self._pending_writes.append(row)

    def load(self) -> None:
        """Hydrate the in-memory tiers from SQLite.

        Persisted embeddings are projected through ``input_proj`` so
        they sit in the same internal space as live writes. Existing
        in-memory state is *not* cleared — the user can call
        :meth:`reset` first if they want a clean reload.
        """
        if self._store is None:
            return
        # Embeddings are always persisted on CPU (see ``store()`` which
        # does ``x_in.detach().cpu().reshape(-1)``) but ``input_proj``
        # may live on CUDA when ``device_map="auto"`` placed the whole
        # module on a detected GPU.  Without this transfer the linear
        # projection would raise ``RuntimeError: Expected all tensors
        # to be on the same device, but found at least two devices,
        # cuda:0 and cpu``.  We use the device of the first parameter
        # of ``input_proj`` as the source of truth (rather than the
        # module-level ``device`` attribute) so the fix also covers
        # any ``to(device)`` calls made after construction.
        proj_device = next(self.input_proj.parameters()).device
        rows = self._store.all_ids()
        for mid in rows:
            row = self._store.get(mid)
            if row is None or row["embedding"] is None:
                continue
            emb = torch.as_tensor(row["embedding"], dtype=torch.float32).unsqueeze(0)
            # Move the loaded tensor onto the projection device.  For
            # the common CPU-only path this is a cheap no-op (same
            # device); on CUDA it is the only way to avoid the
            # cross-device crash described above.
            if emb.device != proj_device:
                emb = emb.to(proj_device)
            try:
                self._check_dim(emb)
            except DimensionMismatchError:
                # Persisted rows from a different embedding_dim are
                # silently skipped — they cannot be projected back.
                continue
            x = self.input_proj(emb)
            with torch.no_grad():
                self.working.store(x)
                slot = self.episodic.store(x)
                self.semantic.update(x)
                self.immune.store(x)
            if 0 <= slot < len(self._episodic_ids):
                self._episodic_ids[slot] = mid
            self._tier_for_id[mid] = row.get("tier", "episodic")

    def reset(self) -> None:
        """Wipe every in-memory tier AND the SQLite database.

        This is a FULL reset: use with care. After reset, the memory
        is empty as if just created.

        If you want a SOFT reset (only in-memory, keep DB), use the
        private ``_reset_in_memory()`` method.
        """
        self._reset_in_memory()
        if self._store is not None:
            try:
                self._store.drop_all()
            except Exception as e:
                raise StorageError(f"Failed to clear SQLite: {e}") from e

    def _reset_in_memory(self) -> None:
        """Wipe every in-memory tier. Does not touch SQLite."""
        self.working.reset()
        self.episodic.reset()
        self.semantic.reset()
        self.immune.reset()
        self._tier_for_id.clear()
        self._episodic_ids = [None] * self.episodic.capacity
        self._pending_writes = []

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by ID.

        Args:
            memory_id: the memory_id returned by ``store()``.

        Returns:
            True if the memory was found and deleted, False otherwise.

        Example:
            >>> mid = memory.store(emb, metadata={"text": "hello"})
            >>> memory.delete(mid)
            True
        """
        # Track which tier holds this memory_id (we set this in store())
        tier_for_id = getattr(self, "_tier_for_id", None)
        if tier_for_id is None:
            self._tier_for_id = {}
            tier_for_id = self._tier_for_id

        tier = tier_for_id.pop(memory_id, None)

        # Drop the slot→id mapping and any unflushed pending write.
        with self._op_lock:
            self._episodic_ids = [
                None if mid == memory_id else mid for mid in self._episodic_ids
            ]
            self._pending_writes = [
                r for r in self._pending_writes if r["memory_id"] != memory_id
            ]

        # Remove from SQLite if persistence is enabled.
        deleted_in_db = False
        if self._store is not None:
            try:
                deleted_in_db = self._store.delete(memory_id)
            except Exception as e:
                raise StorageError(f"Failed to delete from SQLite: {e}") from e

        # Successful if the row was found in EITHER store. ``tier`` is
        # the value popped from the in-memory map above, so it is
        # non-None exactly when the id was tracked in memory.
        return deleted_in_db or (tier is not None)


# Uniform router (no learned parameters; used when router.type="uniform")
class _UniformRouter(nn.Module):
    """Trivial router: equal weights for every tier."""

    def __init__(self, num_tiers: int = 4):
        super().__init__()
        self.num_tiers = num_tiers

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # noqa: ARG002
        w = torch.ones(x.size(0), self.num_tiers, device=x.device) / self.num_tiers
        return {"weights": w, "kl_loss": torch.tensor(0.0, device=x.device)}


# Module-level convenience aliases

def save(memory: MATHIRMemory) -> None:
    """Top-level ``save`` helper: ``from mathir_dropin import save``."""
    memory.save()


def load(memory: MATHIRMemory) -> None:
    """Top-level ``load`` helper: ``from mathir_dropin import load``."""
    memory.load()


__all__ = ["MATHIRMemory", "save", "load"]

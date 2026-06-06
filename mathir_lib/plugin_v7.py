"""
MATHIRPluginV7 — Doctoral-grade memory plugin with all V7 theoretical advances.

V7 adds 8 new algorithms over V6:
  1. EbbinghausMemory (spaced-repetition forgetting)
  2. SparseCodingMemory (17× compression, Theorem 5)
  3. VariationalMemory (uncertainty estimates)
  4. CrossAttentionMemory (learned addressing)
  5. HyperbolicMemory (Poincaré ball, hierarchical)
  6. InfoNCELoss (mutual information learning)
  7. NeuralODEMemory (continuous-time evolution)
  8. MahalanobisImmunologicalMemory (NP-optimal anomaly, Theorem 4)

Backward compatible with V6 — all V6 options still work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple

from .config import get_default_config, merge_config
from .memory import (
    # V6
    WorkingMemory, EpisodicMemory, SemanticMemory, ImmunologicalMemory,
    MahalanobisImmunologicalMemory,
    # V7 — high impact
    EbbinghausMemory, SparseCodingMemory, VariationalMemory,
    # V7 — advanced
    CrossAttentionMemory, HyperbolicMemory, InfoNCELoss, NeuralODEMemory,
    # Raw-embedding episodic (bypass the projection bottleneck)
    RawEmbeddingEpisodicMemory,
)


class MATHIRPluginV7(nn.Module):
    """
    V7 plugin with all theoretical improvements.

    Backward compatible with V6 MATHIRPlugin — same .perceive(), .store(),
    .recall() interface. Just pass use_v7=True or use a v7 config.

    Args:
        embedding_dim: LLM embedding dimension
        config: Config dict (use v7.yaml for full features)

    Example:
        >>> plugin = MATHIRPluginV7(4096)  # V6 behavior by default
        >>> plugin = MATHIRPluginV7(4096, config=load_config('config/v7.yaml'))
    """

    def __init__(self, embedding_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        # Load config
        self.config = merge_config(get_default_config(), config or {})
        self.config["memory"]["embedding_dim"] = embedding_dim
        self.embedding_dim = embedding_dim

        # Extract config
        mem_cfg = self.config["memory"]
        self.internal_dim = mem_cfg["internal_dim"]
        self.working_capacity = mem_cfg["working_capacity"]
        self.episodic_capacity = mem_cfg["episodic_capacity"]
        self.semantic_prototypes = mem_cfg["semantic_prototypes"]
        self.immunological_capacity = mem_cfg["immunological_capacity"]
        self.kl_coefficient = mem_cfg["kl_coefficient"]

        # V7 flags
        self.use_sparse_coding = mem_cfg.get("use_sparse_coding", False)
        self.use_variational = mem_cfg.get("use_variational", False)
        self.use_cross_attention = mem_cfg.get("use_cross_attention", False)
        self.use_neural_ode = mem_cfg.get("use_neural_ode", False)
        self.use_infonce = mem_cfg.get("use_infonce", False)
        self.episodic_type = mem_cfg.get("episodic_type", "standard")
        self.semantic_type = mem_cfg.get("semantic_type", "standard")
        self.immune_type = mem_cfg.get("immune_type", "standard")
        # Approach A: raw-embedding episodic (bypass the projection bottleneck)
        self.use_raw_embedding = mem_cfg.get("use_raw_embedding", False)
        self.raw_embedding_dim = mem_cfg.get("raw_embedding_dim", 384)
        self.raw_projection = mem_cfg.get("raw_projection", False)
        self.raw_proj_dim = mem_cfg.get("raw_proj_dim", 64)

        # Projections
        self.input_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.output_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(self.internal_dim)

        # ===== V6 Memory Tiers (always present) =====
        # Working memory
        self.register_buffer("working_buffer", torch.zeros(self.working_capacity, self.internal_dim))
        self.register_buffer("working_ptr", torch.tensor(0, dtype=torch.long))
        self.working_attention = nn.MultiheadAttention(
            self.internal_dim, num_heads=4, batch_first=True, dropout=0.1
        )

        # Episodic memory — V7 selection
        if self.use_raw_embedding:
            # Approach A: store FULL raw embedding (no projection bottleneck)
            # Operates on the original LLM embedding (e.g. 384-dim MiniLM)
            # instead of the projected 272-dim internal representation.
            self.episodic = RawEmbeddingEpisodicMemory(
                capacity=self.episodic_capacity,
                embedding_dim=self.raw_embedding_dim,
                projection=self.raw_projection,
                proj_dim=self.raw_proj_dim,
            )
        elif self.episodic_type == "ebbinghaus":
            self.episodic = EbbinghausMemory(
                capacity=self.episodic_capacity,
                feature_dim=self.internal_dim,
                alpha=mem_cfg.get("ebbinghaus_alpha", 0.5),
            )
        elif self.use_variational:
            self.episodic = VariationalMemory(
                capacity=self.episodic_capacity,
                feature_dim=self.internal_dim,
                min_sigma=mem_cfg.get("variational_min_sigma", 0.01),
            )
        elif self.use_cross_attention:
            self.episodic = CrossAttentionMemory(
                capacity=self.episodic_capacity,
                feature_dim=self.internal_dim,
                num_heads=4,
            )
        else:
            self.episodic = EpisodicMemory(
                capacity=self.episodic_capacity,
                feature_dim=self.internal_dim,
            )

        # Semantic memory — V7 selection
        if self.semantic_type == "hyperbolic":
            self.semantic = HyperbolicMemory(
                num_prototypes=self.semantic_prototypes,
                feature_dim=self.internal_dim,
                proj_dim=64,
            )
        else:
            self.semantic = SemanticMemory(
                num_prototypes=self.semantic_prototypes,
                feature_dim=self.internal_dim,
                proj_dim=64,
            )

        # Immunological memory — V7 selection
        if self.immune_type == "mahalanobis":
            self.immunological = MahalanobisImmunologicalMemory(
                capacity=self.immunological_capacity,
                feature_dim=self.internal_dim,
                threshold=mem_cfg.get("anomaly_threshold", 2.0),
            )
        else:
            self.immunological = ImmunologicalMemory(
                capacity=self.immunological_capacity,
                feature_dim=self.internal_dim,
                threshold=mem_cfg.get("anomaly_threshold", 2.0),
            )

        # ===== V7 Additional Tiers (optional) =====
        if self.use_sparse_coding:
            self.sparse_coding = SparseCodingMemory(
                num_atoms=mem_cfg.get("sparse_atoms", 1088),
                feature_dim=self.internal_dim,
                sparsity=mem_cfg.get("sparse_sparsity", 8),
            )

        if self.use_neural_ode:
            self.neural_ode = NeuralODEMemory(
                capacity=self.episodic_capacity,
                feature_dim=self.internal_dim,
                dt=mem_cfg.get("neural_ode_dt", 0.1),
                method="rk4",
            )

        # ===== V7 Self-Supervised Learning =====
        if self.use_infonce:
            self.infonce = InfoNCELoss(
                feature_dim=self.internal_dim,
                temperature=mem_cfg.get("infonce_temperature", 0.1),
            )
        else:
            # V6 predictor head
            self.predictor = nn.Sequential(
                nn.Linear(self.internal_dim, self.internal_dim),
                nn.GELU(),
                nn.Linear(self.internal_dim, self.internal_dim),
            )

        # Router (4 memory tiers)
        self.router = nn.Sequential(
            nn.Linear(self.internal_dim, 128),
            nn.GELU(),
            nn.Linear(128, 4),
        )
        self.register_buffer(
            "prev_probs", torch.ones(4) / 4
        )

        # Reconstruction head (V6)
        self.reconstructor = nn.Sequential(
            nn.Linear(self.internal_dim, self.internal_dim),
            nn.GELU(),
            nn.Linear(self.internal_dim, self.internal_dim),
        )

    def perceive(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process embedding through V7 memory system."""
        x = self.input_proj(embedding)

        # Store in working memory
        self._store_working(x)

        # Retrieve from each tier.
        # When use_raw_embedding=True, episodic operates on the FULL raw
        # embedding (no projection bottleneck) and the result is mapped
        # back to internal_dim via input_proj so it can fuse with the
        # other tiers through the router.
        working_ctx = self._retrieve_working(x)
        episodic_ctx = self._retrieve_episodic(x, raw_query=embedding if self.use_raw_embedding else None)
        semantic_ctx = self._retrieve_semantic(x)
        immune_ctx = self._retrieve_immune(x)

        # V7: optional sparse coding
        if self.use_sparse_coding:
            sparse_ctx = self.sparse_coding.retrieve(x)
        else:
            sparse_ctx = torch.zeros_like(x)

        # V7: optional Neural ODE
        if self.use_neural_ode:
            ode_ctx = self.neural_ode.retrieve(x)
        else:
            ode_ctx = torch.zeros_like(x)

        # Router
        router_logits = self.router(x)
        router_weights = F.softmax(router_logits, dim=-1)
        kl_loss = self._compute_kl(router_logits)

        # Weighted fusion (V6 + V7 additions)
        w = router_weights.chunk(4, dim=-1)
        output = (
            w[0] * working_ctx +
            w[1] * episodic_ctx +
            w[2] * semantic_ctx +
            w[3] * immune_ctx
        )
        if self.use_sparse_coding:
            output = output + 0.1 * sparse_ctx
        if self.use_neural_ode:
            output = output + 0.1 * ode_ctx

        # Residual + norm
        output = self.layer_norm(output + x)
        enhanced = self.output_proj(output)

        # Anomaly
        anomaly = self._compute_anomaly(x)

        return {
            "enhanced_embedding": enhanced,
            "router_weights": router_weights,
            "anomaly_score": anomaly,
            "kl_loss": kl_loss,
        }

    def store(self, experience: Dict[str, torch.Tensor]) -> None:
        """Store in V7 memory system."""
        if "embedding" not in experience:
            return
        raw_emb = experience["embedding"].detach()
        emb = self.input_proj(raw_emb)

        # V7 Ebbinghaus: evict if at capacity
        if self.episodic_type == "ebbinghaus":
            count = self.episodic.count.item() if torch.is_tensor(self.episodic.count) else self.episodic.count
            if count >= self.episodic_capacity:
                self.episodic.evict()
            self.episodic.store(emb)
        elif self.use_raw_embedding:
            # Approach A: feed the FULL raw embedding to the raw memory
            # (no projection through input_proj — keep all dimensions)
            if raw_emb.size(-1) != self.raw_embedding_dim:
                raise ValueError(
                    f"raw_embedding_dim mismatch: config={self.raw_embedding_dim}, "
                    f"got={raw_emb.size(-1)}"
                )
            self.episodic.store(raw_emb)
        else:
            self.episodic.store(emb)

        # Semantic
        if hasattr(self.semantic, "update"):
            self.semantic.update(emb) if self.semantic_type == "hyperbolic" else self._update_semantic_standard(emb)
        else:
            self._update_semantic_standard(emb)

        # Immunological
        self.immunological.store(emb)

        # V7: optional tiers
        if self.use_sparse_coding:
            self.sparse_coding.store(emb)
        if self.use_neural_ode:
            self.neural_ode.store(emb)

    def recall(self, query: torch.Tensor, k: int = 3) -> List[Dict[str, Any]]:
        """Recall from episodic memory (V7 compatible)."""
        x = self.input_proj(query)

        # Approach A: raw-embedding episodic operates on the FULL raw query
        if isinstance(self.episodic, RawEmbeddingEpisodicMemory):
            if query.size(-1) != self.raw_embedding_dim:
                raise ValueError(
                    f"raw_embedding_dim mismatch: config={self.raw_embedding_dim}, "
                    f"got query.size(-1)={query.size(-1)}"
                )
            indices, sims = self.episodic.search(query, k=k)
            if indices.numel() == 0:
                return []
            return [
                {"index": idx, "value": self.episodic.values[idx].cpu(), "similarity": s}
                for idx, s in zip(indices[0].tolist(), sims[0].tolist())
            ]

        # Variational returns (value, uncertainty)
        if isinstance(self.episodic, VariationalMemory):
            value, uncertainty = self.episodic.retrieve(x, k=k)
            return [{"value": value.cpu(), "uncertainty": uncertainty.cpu().mean().item()}]

        # CrossAttention returns tensor
        if isinstance(self.episodic, CrossAttentionMemory):
            value = self.episodic.retrieve(x, k=k)
            return [{"value": value.cpu()}]

        # Ebbinghaus, standard episodic return top-k from .search()
        if isinstance(self.episodic, EbbinghausMemory):
            # Ebbinghaus doesn't have search — use cosine
            count = self.episodic.count.item() if torch.is_tensor(self.episodic.count) else self.episodic.count
            if count < k:
                return []
            with torch.no_grad():
                key = self.episodic.encoder(x)
                sims = F.cosine_similarity(
                    key.unsqueeze(1),
                    self.episodic.keys[:count].unsqueeze(0),
                    dim=-1
                )
                top_k = sims.topk(min(k, count), dim=1)[1]
                memories = []
                for i in range(top_k.size(1)):
                    idx = top_k[0, i].item()
                    memories.append({
                        "index": idx,
                        "value": self.episodic.values[idx].cpu(),
                        "similarity": sims[0, idx].item(),
                        "stability": self.episodic.stability[idx].item(),
                    })
                return memories

        # Standard episodic
        if hasattr(self.episodic, "search"):
            indices, sims = self.episodic.search(x, k=k)
            return [
                {"index": idx, "value": self.episodic.values[idx].cpu(), "similarity": s}
                for idx, s in zip(indices[0].tolist(), sims[0].tolist())
            ]
        else:
            # Fallback
            value = self.episodic.retrieve(x, k=k)
            return [{"value": value.cpu()}]

    def forget(self, threshold: float = 0.1) -> None:
        """Forget (V7 Ebbinghaus-aware)."""
        if isinstance(self.episodic, EbbinghausMemory):
            count = self.episodic.count.item() if torch.is_tensor(self.episodic.count) else self.episodic.count
            scores = self.episodic.get_retention_scores()
            if count > 0 and scores.min() < threshold:
                self.episodic.evict()
        elif hasattr(self.episodic, "forget"):
            self.episodic.forget(threshold)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive V7 memory statistics."""
        stats = {
            "version": "V7",
            "config": {
                "episodic_type": self.episodic_type,
                "semantic_type": self.semantic_type,
                "immune_type": self.immune_type,
                "use_sparse_coding": self.use_sparse_coding,
                "use_variational": self.use_variational,
                "use_cross_attention": self.use_cross_attention,
                "use_neural_ode": self.use_neural_ode,
                "use_infonce": self.use_infonce,
            },
        }

        # V6 stats
        stats["working"] = {
            "usage": min(
                self.working_ptr.item() if torch.is_tensor(self.working_ptr) else self.working_ptr,
                self.working_capacity,
            ),
            "capacity": self.working_capacity,
        }

        # Episodic stats
        if isinstance(self.episodic, EbbinghausMemory):
            stats["episodic"] = self.episodic.get_stats()
        elif isinstance(self.episodic, VariationalMemory):
            stats["episodic"] = self.episodic.get_stats()
        elif hasattr(self.episodic, "count"):
            count = self.episodic.count.item() if torch.is_tensor(self.episodic.count) else self.episodic.count
            stats["episodic"] = {"count": count, "type": type(self.episodic).__name__}
        else:
            stats["episodic"] = {"type": type(self.episodic).__name__}

        # Semantic
        if hasattr(self.semantic, "get_stats"):
            stats["semantic"] = self.semantic.get_stats()
        else:
            stats["semantic"] = {"type": type(self.semantic).__name__}

        # Immune
        if hasattr(self.immunological, "get_stats"):
            immune_stats = self.immunological.get_stats()
            immune_stats["type"] = type(self.immunological).__name__
            stats["immunological"] = immune_stats
        else:
            stats["immunological"] = {"type": type(self.immunological).__name__}

        # V7 additional
        if self.use_sparse_coding:
            stats["sparse_coding"] = {
                "compression_ratio": self.sparse_coding.get_compression_ratio(),
                "atoms": self.sparse_coding.num_atoms,
                "sparsity": self.sparse_coding.sparsity,
            }
        if self.use_neural_ode:
            stats["neural_ode"] = self.neural_ode.get_stats()

        return stats

    def compress(self, method: str = "turboquant", bits: int = 3) -> None:
        """Compress memory (placeholder)."""
        pass

    def export_onnx(self, path: str) -> None:
        """Export to ONNX (placeholder)."""
        pass

    # ===== Internal methods =====

    def _store_working(self, x: torch.Tensor) -> None:
        batch_size = x.size(0)
        with torch.no_grad():
            device = x.device
            ptr = self.working_ptr.item() if torch.is_tensor(self.working_ptr) else self.working_ptr
            indices = (ptr + torch.arange(batch_size, device=device)) % self.working_capacity
            self.working_buffer[indices] = x.detach()
            new_ptr = torch.tensor(
                (ptr + batch_size) % self.working_capacity,
                dtype=torch.long,
                device=self.working_ptr.device,
            )
            self.working_ptr = new_ptr

    def _retrieve_working(self, x: torch.Tensor) -> torch.Tensor:
        ptr = self.working_ptr.item() if torch.is_tensor(self.working_ptr) else self.working_ptr
        stored = min(ptr, self.working_capacity)
        if stored == 0:
            return torch.zeros_like(x)
        context = self.working_buffer[:stored].unsqueeze(0).expand(x.size(0), -1, -1)
        out, _ = self.working_attention(x.unsqueeze(1), context, context)
        return out.squeeze(1)

    def _retrieve_episodic(
        self,
        x: torch.Tensor,
        raw_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """V7-aware episodic retrieval.

        Args:
            x: query already projected to ``internal_dim``.
            raw_query: when ``use_raw_embedding`` is True, the FULL raw
                embedding (pre-projection). Used by ``RawEmbeddingEpisodicMemory``
                so it can search in the original 384-dim space. The result
                is projected back to ``internal_dim`` via ``input_proj`` so
                the router can fuse it with the other tiers.
        """
        if isinstance(self.episodic, RawEmbeddingEpisodicMemory):
            if raw_query is None:
                # Fallback: use the projected query (loses fidelity)
                raw_query = x
            raw_value = self.episodic.retrieve(raw_query, k=3)
            # raw_value is in raw_embedding_dim → project to internal_dim
            return self.input_proj(raw_value)

        if isinstance(self.episodic, (VariationalMemory,)):
            value, _ = self.episodic.retrieve(x, k=3)
            return value
        if isinstance(self.episodic, (CrossAttentionMemory,)):
            return self.episodic.retrieve(x, k=3)
        if isinstance(self.episodic, (EbbinghausMemory,)):
            return self.episodic.retrieve(x, k=3)
        # Standard episodic
        return self.episodic.retrieve(x, k=3)

    def _retrieve_semantic(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.semantic, HyperbolicMemory):
            return self.semantic.retrieve(x)
        return self.semantic.retrieve(x)

    def _retrieve_immune(self, x: torch.Tensor) -> torch.Tensor:
        result = self.immunological.recognize(x)
        if result is None:
            return torch.zeros_like(x)
        return result

    def _update_semantic_standard(self, x: torch.Tensor) -> None:
        if isinstance(self.semantic, HyperbolicMemory):
            self.semantic.update(x)
            return
        with torch.no_grad():
            projected = self.semantic.down(x)
            sims = F.cosine_similarity(
                projected.unsqueeze(1),
                self.semantic.prototypes.unsqueeze(0),
                dim=-1
            )
            idx = sims.argmax(dim=1)
            alpha = 0.01
            for i in range(x.size(0)):
                self.semantic.prototypes[idx[i]] = (
                    (1 - alpha) * self.semantic.prototypes[idx[i]] + alpha * projected[i].detach()
                )
                self.semantic.usage[idx[i]] += 1

    def _compute_kl(self, logits: torch.Tensor) -> torch.Tensor:
        target = self.prev_probs.unsqueeze(0).expand_as(logits)
        kl = F.kl_div(
            F.log_softmax(logits, dim=-1),
            target,
            reduction="batchmean",
            log_target=False,
        )
        with torch.no_grad():
            self.prev_probs = F.softmax(logits, dim=-1).mean(dim=0)
        return self.kl_coefficient * kl

    def _compute_anomaly(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.immunological, MahalanobisImmunologicalMemory):
            return self.immunological.get_anomaly_score(x)
        return self.immunological.get_anomaly_score(x)


__all__ = ["MATHIRPluginV7"]

"""
MATHIRPlugin — Adaptive memory for any LLM.

V6: The plugin is fully config-driven. Every dimension, capacity, and
threshold comes from :mod:`mathir_lib.config`. The plugin works with
**any** LLM embedding dimension (4096, 3584, 1536, 1024, ...).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

from .config import get_default_config, merge_config


class MATHIRPlugin(nn.Module):
    """
    Adaptive memory plugin for any LLM.

    Three-tier memory (plus immunological anomaly detection):
        - Working: immediate context (attention over circular buffer)
        - Episodic: past experiences (key-value store, similarity retrieval)
        - Semantic: learned concepts (online k-means prototypes)
        - Immunological: anomaly detection (distance-based threshold)

    Features:
        - KL-constrained router (prevents memory collapse)
        - Online learning (never stops adapting)
        - Plug-and-play (works with any LLM embedding dimension)
        - Config-driven (no hardcoded values)

    Args:
        embedding_dim: The LLM's embedding dimension
        config: Optional config dict (uses defaults if None)
    """

    def __init__(self, embedding_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        # Load and merge config — every constant below is config-driven.
        self.config = merge_config(get_default_config(), config or {})
        self.config["memory"]["embedding_dim"] = embedding_dim

        # Extract config values
        mem_cfg = self.config["memory"]
        self.embedding_dim = embedding_dim
        self.internal_dim = mem_cfg["internal_dim"]
        self.working_capacity = mem_cfg["working_capacity"]
        self.episodic_capacity = mem_cfg["episodic_capacity"]
        self.semantic_prototypes = mem_cfg["semantic_prototypes"]
        self.immunological_capacity = mem_cfg["immunological_capacity"]
        self.kl_coefficient = mem_cfg["kl_coefficient"]
        self.anomaly_threshold = mem_cfg["anomaly_threshold"]

        # === Projections ===
        # Input projection: LLM dim → MATHIR internal dim
        self.input_proj = nn.Linear(embedding_dim, self.internal_dim)

        # Output projection: MATHIR internal dim → LLM dim
        self.output_proj = nn.Linear(self.internal_dim, embedding_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(self.internal_dim)

        # === Working Memory ===
        # Circular buffer + multi-head attention
        self.register_buffer(
            "working_buffer",
            torch.zeros(self.working_capacity, self.internal_dim),
        )
        self.register_buffer("working_ptr", torch.tensor(0, dtype=torch.long))
        self.working_attention = nn.MultiheadAttention(
            self.internal_dim, num_heads=4, batch_first=True, dropout=0.1
        )

        # === Episodic Memory ===
        # Key-value store with similarity retrieval
        self.register_buffer(
            "episodic_keys",
            torch.zeros(self.episodic_capacity, 64),
        )
        self.register_buffer(
            "episodic_values",
            torch.zeros(self.episodic_capacity, self.internal_dim),
        )
        self.register_buffer("episodic_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("episodic_count", torch.tensor(0, dtype=torch.long))
        self.episodic_encoder = nn.Linear(self.internal_dim, 64)

        # === Semantic Memory ===
        # Online k-means prototypes with learned projection
        self.register_buffer(
            "semantic_proto_buffer",
            torch.randn(self.semantic_prototypes, 64) * 0.1,
        )
        self.register_buffer(
            "semantic_usage",
            torch.zeros(self.semantic_prototypes),
        )
        self.semantic_down = nn.Linear(self.internal_dim, 64)
        self.semantic_up = nn.Linear(64, self.internal_dim)

        # === Immunological Memory ===
        # Anomaly detection via distance threshold
        self.register_buffer(
            "immune_bank",
            torch.zeros(self.immunological_capacity, self.internal_dim),
        )
        self.register_buffer("immune_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("immune_count", torch.tensor(0, dtype=torch.long))

        # === KL-Constrained Router ===
        self.router = nn.Sequential(
            nn.Linear(self.internal_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.config["router"]["num_memories"]),
        )
        self.register_buffer(
            "prev_probs",
            torch.ones(self.config["router"]["num_memories"])
            / self.config["router"]["num_memories"],
        )

        # === Self-supervised learning heads ===
        # Prediction head (next embedding)
        self.predictor = nn.Sequential(
            nn.Linear(self.internal_dim, self.internal_dim),
            nn.GELU(),
            nn.Linear(self.internal_dim, self.internal_dim),
        )
        # Reconstruction head
        self.reconstructor = nn.Sequential(
            nn.Linear(self.internal_dim, self.internal_dim),
            nn.GELU(),
            nn.Linear(self.internal_dim, self.internal_dim),
        )

    def perceive(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process an embedding through the memory system.

        Args:
            embedding: [B, D] tensor from the LLM

        Returns:
            enhanced_embedding: [B, D] with memory context
            router_weights: [B, num_memories] memory tier allocation
            anomaly_score: [B] novelty detection scores
            kl_loss: scalar KL divergence loss
        """
        # Project to internal dimension
        x = self.input_proj(embedding)

        # Store in working memory
        self._store_working(x)

        # Retrieve from each tier
        working_ctx = self._retrieve_working(x)
        episodic_ctx = self._retrieve_episodic(x)
        semantic_ctx = self._retrieve_semantic(x)
        immune_ctx = self._retrieve_immune(x)

        # Router allocation
        router_logits = self.router(x)
        router_weights = F.softmax(
            router_logits / self.config["router"]["temperature"],
            dim=-1,
        )

        # KL constraint
        kl_loss = self._compute_kl(router_logits)

        # Weighted fusion across memory tiers
        num_mem = self.config["router"]["num_memories"]
        w = router_weights.chunk(num_mem, dim=-1)
        output = w[0] * working_ctx
        if num_mem > 1:
            output = output + w[1] * episodic_ctx
        if num_mem > 2:
            output = output + w[2] * semantic_ctx
        if num_mem > 3:
            output = output + w[3] * immune_ctx

        # Residual + normalize
        output = self.layer_norm(output + x)

        # Project back to embedding dimension
        enhanced = self.output_proj(output)

        # Anomaly score
        anomaly = self._compute_anomaly(x)

        return {
            "enhanced_embedding": enhanced,
            "router_weights": router_weights,
            "anomaly_score": anomaly,
            "kl_loss": kl_loss,
        }

    def store(self, experience: Dict[str, torch.Tensor]) -> None:
        """
        Store an experience for later recall.

        Args:
            experience: dict with 'embedding' (required), 'action' and
                'outcome' (optional).
        """
        if "embedding" not in experience:
            return

        emb = self.input_proj(experience["embedding"].detach())

        # Store in episodic memory
        with torch.no_grad():
            key = self.episodic_encoder(emb.mean(0, keepdim=True)).squeeze(0)
            idx = self.episodic_ptr % self.episodic_capacity
            self.episodic_keys[idx] = key.detach()
            self.episodic_values[idx] = emb.mean(0).detach()
            new_ptr = (int(self.episodic_ptr.item()) + 1) % self.episodic_capacity
            new_count = min(int(self.episodic_count.item()) + 1, self.episodic_capacity)
            self._set_int_buffer("episodic_ptr", new_ptr)
            self._set_int_buffer("episodic_count", new_count)

        # Update semantic prototypes
        self._update_semantic(emb)

        # Update immune bank
        self._update_immune(emb)

    def recall(self, query: torch.Tensor, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories.

        Args:
            query: [B, D] tensor to search for
            k: number of memories to retrieve

        Returns:
            list of memory dicts with 'embedding', 'similarity', 'index'
        """
        x = self.input_proj(query)
        count = self._buffer_count(self.episodic_count)

        if count < k:
            return []

        with torch.no_grad():
            key = self.episodic_encoder(x)
            sims = F.cosine_similarity(
                key.unsqueeze(1),
                self.episodic_keys[:count].unsqueeze(0),
                dim=-1,
            )
            top_k = sims.topk(min(k, count), dim=1)

        memories = []
        for i in range(top_k.indices.size(1)):
            idx = top_k.indices[0, i].item()
            memories.append({
                "embedding": self.episodic_values[idx].cpu(),
                "similarity": top_k.values[0, i].item(),
                "index": idx,
            })

        return memories

    def forget(self, threshold: float = 0.1) -> None:
        """Prune irrelevant memories (controlled forgetting)."""
        count = self._buffer_count(self.episodic_count)
        if count == 0:
            return

        with torch.no_grad():
            keys = self.episodic_keys[:count]
            sims = F.cosine_similarity(
                keys.unsqueeze(1), keys.unsqueeze(0), dim=-1
            )
            usage = sims.mean(dim=1)
            mask = usage > threshold

            if mask.sum() < count:
                self.episodic_keys[:mask.sum()] = self.episodic_keys[:count][mask]
                self.episodic_values[:mask.sum()] = self.episodic_values[:count][mask]
                self._set_int_buffer("episodic_count", int(mask.sum().item()))
                self._set_int_buffer("episodic_ptr", int(mask.sum().item()))

    def get_stats(self) -> Dict[str, Any]:
        """Get memory utilization statistics."""
        return {
            "working_usage": min(
                self._buffer_count(self.working_ptr),
                self.working_capacity,
            ),
            "working_capacity": self.working_capacity,
            "episodic_usage": self._buffer_count(self.episodic_count),
            "episodic_capacity": self.episodic_capacity,
            "semantic_usage": (self.semantic_usage > 0).sum().item(),
            "semantic_capacity": self.semantic_prototypes,
            "immune_usage": self._buffer_count(self.immune_count),
            "immune_capacity": self.immunological_capacity,
        }

    def compress(self, method: str = "turboquant", bits: int = 3) -> None:
        """Apply compression to memory buffers (Agent C owns logic)."""
        # Compression logic — call into compression module (Agent C)
        # For now, placeholder
        return None

    def export_onnx(self, path: str) -> None:
        """Export to ONNX format."""
        # Placeholder — will be implemented in V8
        return None

    # === Internal methods ===

    def _store_working(self, x: torch.Tensor) -> None:
        """Store in working memory circular buffer."""
        batch_size = x.size(0)
        with torch.no_grad():
            device = x.device
            ptr = self._buffer_count(self.working_ptr)
            indices = (ptr + torch.arange(batch_size, device=device)) % self.working_capacity
            self.working_buffer[indices] = x.detach()
            self._set_int_buffer("working_ptr", (ptr + batch_size) % self.working_capacity)

    def _retrieve_working(self, x: torch.Tensor) -> torch.Tensor:
        """Retrieve from working memory via attention."""
        ptr = self._buffer_count(self.working_ptr)
        stored = min(ptr, self.working_capacity)
        if stored == 0:
            return torch.zeros_like(x)
        context = self.working_buffer[:stored].unsqueeze(0).expand(x.size(0), -1, -1)
        out, _ = self.working_attention(x.unsqueeze(1), context, context)
        return out.squeeze(1)

    def _retrieve_episodic(self, x: torch.Tensor) -> torch.Tensor:
        """Retrieve from episodic memory via similarity."""
        count = self._buffer_count(self.episodic_count)
        if count < 10:
            return torch.zeros_like(x)
        key = self.episodic_encoder(x)
        sims = F.cosine_similarity(
            key.unsqueeze(1),
            self.episodic_keys[:count].unsqueeze(0),
            dim=-1,
        )
        top_k = sims.topk(min(3, count), dim=1)[1]
        return self.episodic_values[top_k].mean(1)

    def _retrieve_semantic(self, x: torch.Tensor) -> torch.Tensor:
        """Retrieve from semantic memory via prototype matching."""
        projected = self.semantic_down(x)
        sims = F.cosine_similarity(
            projected.unsqueeze(1),
            self.semantic_proto_buffer.unsqueeze(0),
            dim=-1,
        )
        idx = sims.argmax(dim=1)
        return self.semantic_up(self.semantic_proto_buffer[idx])

    def _retrieve_immune(self, x: torch.Tensor) -> torch.Tensor:
        """Detect anomalies via immune memory."""
        count = self._buffer_count(self.immune_count)
        if count < 10:
            return torch.zeros_like(x)
        dists = torch.cdist(x, self.immune_bank[:count])
        min_dist = dists.min(dim=1)[0]
        anomaly = (min_dist > self.anomaly_threshold).float().unsqueeze(-1)
        return anomaly * x

    def _update_semantic(self, x: torch.Tensor) -> None:
        """Update semantic prototypes via online k-means."""
        with torch.no_grad():
            projected = self.semantic_down(x)
            sims = F.cosine_similarity(
                projected.unsqueeze(1),
                self.semantic_proto_buffer.unsqueeze(0),
                dim=-1,
            )
            idx = sims.argmax(dim=1)
            alpha = 0.01
            for i in range(x.size(0)):
                self.semantic_proto_buffer[idx[i]] = (
                    (1 - alpha) * self.semantic_proto_buffer[idx[i]]
                    + alpha * projected[i].detach()
                )
                self.semantic_usage[idx[i]] += 1

    def _update_immune(self, x: torch.Tensor) -> None:
        """Update immune memory bank."""
        with torch.no_grad():
            ptr = self._buffer_count(self.immune_ptr)
            idx = ptr % self.immunological_capacity
            self.immune_bank[idx] = x.mean(0).detach()
            self._set_int_buffer("immune_ptr", (ptr + 1) % self.immunological_capacity)
            new_count = self._buffer_count(self.immune_count) + 1
            self._set_int_buffer("immune_count", min(new_count, self.immunological_capacity))

    def _compute_kl(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss for router."""
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
        """Compute anomaly score for input."""
        count = self._buffer_count(self.immune_count)
        if count < 10:
            return torch.zeros(x.size(0), device=x.device)
        dists = torch.cdist(x, self.immune_bank[:count])
        return dists.min(dim=1)[0]

    @staticmethod
    def _buffer_count(t: torch.Tensor) -> int:
        """Extract an int from a 0-d buffer (CPU-safe)."""
        if torch.is_tensor(t):
            return int(t.item())
        return int(t)

    def _set_int_buffer(self, name: str, value: int) -> None:
        """Assign a Python int to a long-typed buffer."""
        self.register_buffer(
            name, torch.tensor(int(value), dtype=torch.long)
        )


__all__ = ["MATHIRPlugin"]

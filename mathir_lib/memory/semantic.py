"""
Semantic Memory — Learned concepts.
Uses online k-means prototypes with learned projection.
"""

import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SemanticMemory(nn.Module):
    """
    Semantic memory: online k-means prototypes.
    
    Capacity: configurable (default 256 prototypes)
    Update: every 100 steps (configurable)
    Retrieval: prototype matching via cosine similarity
    """
    
    def __init__(
        self,
        num_prototypes: int = 256,
        feature_dim: int = 272,
        proj_dim: int = 64,
        update_rate: float = 0.01,
    ):
        super().__init__()
        self._lock = threading.RLock()
        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.update_rate = update_rate
        
        # Prototypes (online k-means)
        self.register_buffer(
            "prototypes",
            torch.randn(num_prototypes, proj_dim) * 0.1
        )
        self.register_buffer("usage", torch.zeros(num_prototypes))
        
        # Learned projection: features ↔ prototype space
        self.down = nn.Linear(feature_dim, proj_dim)
        self.up = nn.Linear(proj_dim, feature_dim)
    
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from semantic memory via prototype matching.
        
        Args:
            query: [B, D] tensor to match
            
        Returns:
            [B, D] retrieved prototype (projected back) + query
        """
        with self._lock:
            with torch.no_grad():
                projected = self.down(query)
                sims = F.cosine_similarity(
                    projected.unsqueeze(1),
                    self.prototypes.unsqueeze(0),
                    dim=-1
                )
                idx = sims.argmax(dim=1)
            retrieved = self.up(self.prototypes[idx])
            return retrieved + query  # Residual
    
    def update(self, features: torch.Tensor) -> None:
        """
        Update prototypes via online k-means.
        O(B * num_prototypes) for cosine scan, O(B) for scatter update.
        Fully vectorized — no Python for-loop.
        """
        with self._lock:
            with torch.no_grad():
                projected = self.down(features)
                sims = F.cosine_similarity(
                    projected.unsqueeze(1),
                    self.prototypes.unsqueeze(0),
                    dim=-1
                )
                idx = sims.argmax(dim=1)  # [B]

                # Vectorized EMA update — no Python for-loop
                alpha = self.update_rate
                # Scatter: prototypes[idx] = (1-alpha) * prototypes[idx] + alpha * projected
                old = self.prototypes[idx]  # [B, proj_dim]
                self.prototypes[idx] = (1 - alpha) * old + alpha * projected.detach()

                # Vectorized usage increment
                self.usage.scatter_add_(0, idx, torch.ones_like(idx, dtype=self.usage.dtype))
    
    def reset(self) -> None:
        """Reset semantic memory (re-initialize prototypes)."""
        with torch.no_grad():
            self.prototypes.copy_(torch.randn(self.num_prototypes, self.proj_dim) * 0.1)
            self.usage.zero_()
    
    def get_usage(self) -> int:
        """Get number of prototypes that have been used."""
        return (self.usage > 0).sum().item()
    
    def get_stats(self) -> dict:
        """Get semantic memory statistics."""
        return {
            "num_prototypes": self.num_prototypes,
            "used_prototypes": (self.usage > 0).sum().item(),
            "avg_usage": self.usage.mean().item(),
            "max_usage": self.usage.max().item(),
        }

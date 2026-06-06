"""
Ebbinghaus Memory — Spaced-repetition forgetting.
Replaces FIFO eviction with biologically-inspired forgetting curves.

Theory (THEORY.md, Algorithm 4):
    R(t) = exp(-t / S)             # recall probability
    S_{n+1} = S_n * (1 + alpha)^recall_count
    t_1/2 = S * log(2)
"""

import math
import time
import torch
import torch.nn as nn
from typing import Optional


class EbbinghausMemory(nn.Module):
    """
    Memory with Ebbinghaus forgetting and spaced repetition.

    Each memory has a stability S that grows with each recall:
        R(t) = exp(-t / S)  where t is time since last access.

    Stability update:
        S_{n+1} = S_n * (1 + alpha)^recall_count

    The half-life t_1/2 = S * log(2) determines when recall probability
    drops to 50%.
    """

    def __init__(self, capacity: int = 1000, feature_dim: int = 272,
                 initial_stability: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.initial_stability = initial_stability
        self.alpha = alpha

        # Memory store
        self.register_buffer("values", torch.zeros(capacity, feature_dim))
        self.register_buffer("keys", torch.zeros(capacity, 64))
        self.register_buffer("stability", torch.ones(capacity) * initial_stability)
        self.register_buffer("last_access", torch.zeros(capacity))
        self.register_buffer("recall_count", torch.zeros(capacity))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("current_time", torch.tensor(0.0))

        # Encoder
        self.encoder = nn.Linear(feature_dim, 64)

    def store(self, features: torch.Tensor) -> None:
        """Store features with initial stability."""
        with torch.no_grad():
            key = self.encoder(features.detach().mean(0, keepdim=True)).squeeze(0)

            idx = self.ptr % self.capacity
            self.keys[idx] = key.detach()
            self.values[idx] = features.detach().mean(0)
            self.stability[idx] = self.initial_stability
            self.last_access[idx] = self.current_time
            self.recall_count[idx] = 0

            self.ptr = (self.ptr + 1) % self.capacity
            self.count = torch.minimum(
                self.count + 1,
                torch.tensor(self.capacity, dtype=torch.long, device=self.count.device),
            )
            self.current_time = self.current_time + 1

    def retrieve(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Retrieve top-k memories by similarity, update stability on access.
        """
        count = self.count.item() if torch.is_tensor(self.count) else self.count
        if count < k:
            return query

        # Find top-k
        key = self.encoder(query)
        sims = torch.nn.functional.cosine_similarity(
            key.unsqueeze(1),
            self.keys[:count].unsqueeze(0),
            dim=-1,
        )
        top_k = sims.topk(min(k, count), dim=1)[1]

        # Update stability (spaced repetition)
        with torch.no_grad():
            for idx in top_k[0]:
                idx_item = idx.item()
                # Ebbinghaus stability boost
                self.recall_count[idx_item] = self.recall_count[idx_item] + 1
                self.stability[idx_item] = self.stability[idx_item] * (1 + self.alpha)
                self.last_access[idx_item] = self.current_time

        retrieved = self.values[top_k].mean(1)
        return retrieved + query  # Residual

    def evict(self) -> int:
        """
        Evict the memory with the lowest recall probability R(t).
        Returns index evicted.
        """
        count = self.count.item() if torch.is_tensor(self.count) else self.count
        if count == 0:
            return -1

        with torch.no_grad():
            # R(t) = exp(-(current_time - last_access) / stability)
            time_since_access = self.current_time - self.last_access[:count]
            recall_prob = torch.exp(-time_since_access / (self.stability[:count] + 1e-6))

            # Evict lowest
            min_idx = recall_prob.argmin().item()

            # Compact: move last to evicted slot
            if min_idx < count - 1:
                self.keys[min_idx] = self.keys[count - 1]
                self.values[min_idx] = self.values[count - 1]
                self.stability[min_idx] = self.stability[count - 1]
                self.last_access[min_idx] = self.last_access[count - 1]
                self.recall_count[min_idx] = self.recall_count[count - 1]

            self.count = torch.tensor(count - 1, dtype=torch.long, device=self.count.device)
            self.ptr = (self.ptr - 1) % self.capacity

        return min_idx

    def get_retention_scores(self) -> torch.Tensor:
        """Get current recall probability for all memories."""
        count = self.count.item() if torch.is_tensor(self.count) else self.count
        if count == 0:
            return torch.zeros(0)
        time_since_access = self.current_time - self.last_access[:count]
        return torch.exp(-time_since_access / (self.stability[:count] + 1e-6))

    def get_stats(self) -> dict:
        """Get memory statistics."""
        count = self.count.item() if torch.is_tensor(self.count) else self.count
        scores = self.get_retention_scores()
        return {
            "count": count,
            "capacity": self.capacity,
            "avg_stability": self.stability[:count].mean().item() if count > 0 else 0,
            "avg_recall_count": self.recall_count[:count].mean().item() if count > 0 else 0,
            "avg_retention": scores.mean().item() if count > 0 else 0,
            "min_retention": scores.min().item() if count > 0 else 0,
        }

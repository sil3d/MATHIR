"""
Variational Memory — Point estimates with Gaussian uncertainty.
Each slot stores (mu, sigma) instead of a point.

Theory (THEORY.md, Algorithm 1):
    log p(q | m_i) >= -||q - mu_i||^2 / (2*sigma_i^2)
                     - 0.5*log(sigma_i^2) + const
    Reparameterization: m_hat = mu + sigma * eps, eps ~ N(0, I)

Storage cost: 2x dense (stores both mu and log_sigma).
Benefit: Provides confidence scores, handles noisy inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VariationalMemory(nn.Module):
    """
    Variational memory with Gaussian uncertainty per slot.

    Each slot is a distribution N(mu, sigma^2) instead of a point.
    Allows the system to express uncertainty about stored memories.

    Storage: 2x dense (stores both mu and log_sigma).
    Benefit: Provides confidence scores, handles noisy inputs.
    """

    def __init__(self, capacity: int = 1000, feature_dim: int = 272,
                 min_sigma: float = 0.01):
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.min_sigma = min_sigma

        # Mean and log-sigma
        self.register_buffer("mu", torch.zeros(capacity, feature_dim))
        self.register_buffer("log_sigma", torch.zeros(capacity, feature_dim))
        self.register_buffer("keys", torch.zeros(capacity, 64))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        # Encoder
        self.encoder = nn.Linear(feature_dim, 64)

    def store(self, features: torch.Tensor) -> None:
        """Store features as a Gaussian distribution."""
        with torch.no_grad():
            mu = features.detach().mean(0)
            # Initial uncertainty = small
            log_sigma = torch.full_like(mu, -3.0)  # sigma = e^-3 ~ 0.05

            key = self.encoder(features.detach().mean(0, keepdim=True)).squeeze(0)

            idx = self.ptr % self.capacity
            self.mu[idx] = mu
            self.log_sigma[idx] = log_sigma
            self.keys[idx] = key.detach()

            self.ptr = (self.ptr + 1) % self.capacity
            self.count = torch.minimum(
                self.count + 1,
                torch.tensor(self.capacity, dtype=torch.long, device=self.count.device),
            )

    def sample(self, idx: int) -> torch.Tensor:
        """Sample from the variational distribution using reparameterization trick."""
        mu = self.mu[idx]
        sigma = torch.exp(self.log_sigma[idx]).clamp(min=self.min_sigma)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def retrieve(self, query: torch.Tensor, k: int = 3,
                 sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k memories and return (retrieved, uncertainty).

        Args:
            query: [B, D] query
            k: number of memories
            sample: if True, sample; if False, use mean

        Returns:
            retrieved: [B, D] reconstructed
            uncertainty: [B] average uncertainty of retrieved memories
        """
        count = self.count.item() if torch.is_tensor(self.count) else self.count
        if count < k:
            return query, torch.ones(query.size(0), device=query.device)

        with torch.no_grad():
            key = self.encoder(query)
            sims = F.cosine_similarity(
                key.unsqueeze(1),
                self.keys[:count].unsqueeze(0),
                dim=-1,
            )
            top_k = sims.topk(min(k, count), dim=1)[1]

            if sample:
                # Sample from each distribution
                retrieved_list = []
                uncertainty_list = []
                for i in range(top_k.size(1)):
                    idx = top_k[0, i].item()
                    s = self.sample(idx)
                    retrieved_list.append(s)
                    sigma = torch.exp(self.log_sigma[idx]).mean()
                    uncertainty_list.append(sigma)
                retrieved = torch.stack(retrieved_list, dim=0).mean(dim=0, keepdim=True)
                uncertainty = torch.stack(uncertainty_list, dim=0).mean().unsqueeze(0)
            else:
                # Use means
                retrieved = self.mu[top_k].mean(dim=1)
                uncertainty = torch.exp(self.log_sigma[top_k]).mean(dim=(1, 2))

        return retrieved + query, uncertainty

    def update_uncertainty(self, idx: int, evidence_quality: float) -> None:
        """
        Update uncertainty based on how good the evidence was.
        Lower uncertainty for high-quality evidence.
        """
        with torch.no_grad():
            # If evidence is good (high similarity), reduce uncertainty
            # If evidence is bad (low similarity), keep uncertainty high
            adjustment = -0.1 * evidence_quality
            self.log_sigma[idx] = (self.log_sigma[idx] + adjustment).clamp(min=-5, max=2)

    def get_stats(self) -> dict:
        """Get statistics about uncertainty."""
        count = self.count.item() if torch.is_tensor(self.count) else self.count
        if count == 0:
            return {"count": 0}
        sigmas = torch.exp(self.log_sigma[:count])
        return {
            "count": count,
            "mean_sigma": sigmas.mean().item(),
            "min_sigma": sigmas.min().item(),
            "max_sigma": sigmas.max().item(),
            "avg_uncertainty_bits": (torch.log2(1.0 / sigmas)).mean().item(),
        }

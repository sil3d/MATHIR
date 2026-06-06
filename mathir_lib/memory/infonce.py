"""
InfoNCE Contrastive Learning — Self-supervised objectives.
Replaces MSE predictor head with mutual information bound.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for self-supervised memory learning.

    Theory (Oord et al., 2018): minimizing InfoNCE maximizes a lower bound
    on mutual information:
        I(f(x_t); f(x_{t+k})) >= log(N) - L_InfoNCE

    Args:
        feature_dim: dimension of representations
        temperature: softmax temperature (lower = sharper)
        projection_dim: projection head dimension
    """

    def __init__(self, feature_dim: int = 272, temperature: float = 0.1,
                 projection_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature

        # Projection head (SimCLR-style)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, projection_dim),
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, z_t: torch.Tensor, z_tk: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss between two batches of representations.

        Args:
            z_t: [B, D] representations at time t
            z_tk: [B, D] representations at time t+k (positive pairs)

        Returns:
            scalar InfoNCE loss
        """
        B = z_t.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z_t.device)

        # Project
        p_t = self.predictor(self.projection(z_t))  # [B, P]
        z_tk_proj = self.projection(z_tk)  # [B, P]

        # Normalize
        p_t = F.normalize(p_t, dim=-1)
        z_tk_proj = F.normalize(z_tk_proj, dim=-1)

        # Similarity matrix
        sim = p_t @ z_tk_proj.T / self.temperature  # [B, B]

        # Labels: diagonal is positive pair
        labels = torch.arange(B, device=z_t.device)

        # Symmetric loss
        loss_t_to_tk = F.cross_entropy(sim, labels)
        loss_tk_to_t = F.cross_entropy(sim.T, labels)

        return (loss_t_to_tk + loss_tk_to_t) / 2

    def get_mutual_information_bound(self, loss_value: float, n_negatives: int) -> float:
        """
        Compute the lower bound on mutual information from the loss.
        I >= log(N) - L
        """
        import math
        return math.log(n_negatives) - loss_value.item()

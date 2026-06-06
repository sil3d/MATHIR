"""
Hyperbolic Memory — Poincaré ball embeddings for hierarchies.
Better for tree-like semantic structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HyperbolicMemory(nn.Module):
    """
    Memory in Poincaré ball (hyperbolic space).

    Distance: d_H(u, v) = arccosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))

    Advantage: hyperbolic space grows exponentially with radius, so trees embed
    with low distortion. Good for hierarchical semantic memory.
    """

    def __init__(self, num_prototypes: int = 256, feature_dim: int = 272,
                 proj_dim: int = 64, c: float = 1.0, epsilon: float = 1e-5):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.c = c  # Curvature
        self.epsilon = epsilon

        # Projections
        self.down = nn.Linear(feature_dim, proj_dim)
        self.up = nn.Linear(proj_dim, feature_dim)

        # Prototypes in hyperbolic space (initialized at origin = 0)
        # Use small random initialization in Poincaré ball
        self.register_buffer(
            "prototypes",
            torch.randn(num_prototypes, proj_dim) * 0.01
        )
        # Project to ball (norm < 1)
        with torch.no_grad():
            norms = self.prototypes.norm(dim=-1, keepdim=True)
            self.prototypes.data = self.prototypes.data / (norms + epsilon) * 0.1

        self.register_buffer("usage", torch.zeros(num_prototypes))

    def exp_map(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from tangent space to Poincaré ball."""
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.epsilon)
        return torch.tanh(v_norm / (2 * self.c ** 0.5)) * v / v_norm

    def log_map(self, x: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from Poincaré ball to tangent space."""
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=self.epsilon, max=1 - self.epsilon)
        return 2 * (x / x_norm) * torch.atanh(x_norm / (1 + self.epsilon))

    def poincare_distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic distance in Poincaré ball.

        d_H(u, v) = arccosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2) + eps))
        """
        diff_norm_sq = (u - v).pow(2).sum(dim=-1)
        u_norm_sq = u.pow(2).sum(dim=-1)
        v_norm_sq = v.pow(2).sum(dim=-1)
        denom = (1 - u_norm_sq) * (1 - v_norm_sq)
        arg = 1 + 2 * diff_norm_sq / (denom.clamp(min=self.epsilon))
        return torch.acosh(arg.clamp(min=1.0 + self.epsilon))

    def project_to_ball(self, x: torch.Tensor) -> torch.Tensor:
        """Project to inside the Poincaré ball (norm < 1)."""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.epsilon)
        max_norm = 1.0 - self.epsilon
        return x * torch.minimum(torch.ones_like(norm), max_norm / norm)

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve prototype closest in hyperbolic distance.

        Args:
            query: [B, D]

        Returns:
            [B, D] projected back to Euclidean + query (residual)
        """
        # Project query to Poincaré ball
        q_proj = self.down(query)
        q_proj = self.project_to_ball(q_proj)

        # Compute distances
        dists = self.poincare_distance(
            q_proj.unsqueeze(1),  # [B, 1, proj_dim]
            self.prototypes.unsqueeze(0)  # [1, P, proj_dim]
        )  # [B, P]

        # Find closest
        idx = dists.argmin(dim=-1)

        # Map back to Euclidean
        retrieved_proto = self.prototypes[idx]
        retrieved = self.up(retrieved_proto)
        return retrieved + query  # Residual

    def update(self, features: torch.Tensor, learning_rate: float = 0.01) -> None:
        """Update prototypes in hyperbolic space."""
        with torch.no_grad():
            x_proj = self.down(features)
            x_proj = self.project_to_ball(x_proj)

            # Find nearest prototype (Riemannian gradient step)
            dists = self.poincare_distance(
                x_proj.unsqueeze(1),
                self.prototypes.unsqueeze(0)
            )
            idx = dists.argmin(dim=-1)

            # Move prototype toward input (using exp_map)
            for i in range(features.size(0)):
                p = self.prototypes[idx[i]]
                direction = x_proj[i] - p
                # Riemannian gradient: scale by (1 - ||p||^2)^2 / 2
                p_norm_sq = p.pow(2).sum()
                scale = (1 - p_norm_sq).pow(2) / 2
                # Step in tangent space
                step = self.exp_map(direction * learning_rate * scale)
                new_p = p + step
                self.prototypes[idx[i]] = self.project_to_ball(new_p)
                self.usage[idx[i]] += 1

    def get_stats(self) -> dict:
        """Get hyperbolic memory statistics."""
        norms = self.prototypes.norm(dim=-1)
        return {
            "num_prototypes": self.num_prototypes,
            "used": (self.usage > 0).sum().item(),
            "mean_norm": norms.mean().item(),
            "max_norm": norms.max().item(),
            "min_norm": norms.min().item(),
        }

"""
Sparse Coding Memory — 5th memory tier.
Uses ISTA (Iterative Shrinkage-Thresholding Algorithm) for sparse codes.

Theory (THEORY.md, Algorithm 2 + Theorem 5):
    z* = argmin_z 0.5||x - D^T z||^2 + lambda ||z||_1
    Storage: O(s*K) per memory (only non-zeros) vs O(d) for dense.
    Compression: ~d/s (17x for d=272, s=8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SparseCodingMemory(nn.Module):
    """
    Sparse coding memory tier.

    Each memory is a sparse linear combination of basis vectors.
    Reconstruction: x_hat = D @ z where z is sparse.
    """

    def __init__(self, num_atoms: int = 1088, feature_dim: int = 272,
                 sparsity: int = 8, lambda_l1: float = 0.1, n_iter: int = 50):
        super().__init__()
        self.num_atoms = num_atoms
        self.feature_dim = feature_dim
        self.sparsity = sparsity
        self.lambda_l1 = lambda_l1
        self.n_iter = n_iter

        # Dictionary: [num_atoms, feature_dim]
        self.dictionary = nn.Parameter(torch.randn(num_atoms, feature_dim) * 0.1)
        # Normalize dictionary atoms
        with torch.no_grad():
            self.dictionary.data = F.normalize(self.dictionary.data, dim=1)

        # Encoder: feature -> sparse code (warm start for ISTA)
        self.encoder = nn.Linear(feature_dim, num_atoms)

    def ista(self, x: torch.Tensor) -> torch.Tensor:
        """
        Iterative Shrinkage-Thresholding Algorithm with hard-thresholding.

        Solve: z* = argmin_z 0.5||x - D^T z||^2 + lambda||z||_1
                                        s.t. ||z||_0 <= sparsity

        Uses ISTA + top-k hard thresholding (a common variant) to enforce
        the desired sparsity level. Reconstruction: x_hat = z @ D.

        Args:
            x: [B, D] input features

        Returns:
            z: [B, num_atoms] sparse codes (only `sparsity` non-zeros per row)
        """
        B = x.size(0)
        # Initialize with encoder
        z = self.encoder(x)

        Dt = self.dictionary  # [K, D]

        # Step size: 1 / Lipschitz constant
        # Lipschitz = max eigenvalue of D D^T
        DtD = Dt @ Dt.T  # [K, K]
        L = DtD.max() + 1e-6
        step = 1.0 / L.item()

        # Soft thresholding
        def soft_threshold(v, t):
            return torch.sign(v) * F.relu(v.abs() - t)

        for _ in range(self.n_iter):
            gradient = z - step * (z @ DtD - x @ Dt.T)  # [B, K]
            z = soft_threshold(gradient, step * self.lambda_l1)
            # Enforce top-k sparsity (hard thresholding)
            if self.sparsity < self.num_atoms:
                topk_vals, topk_idx = z.abs().topk(self.sparsity, dim=-1)
                mask = torch.zeros_like(z)
                mask.scatter_(-1, topk_idx, 1.0)
                z = z * mask

        return z

    def store(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features as sparse codes and store.

        Returns the sparse codes (for inspection).
        """
        with torch.no_grad():
            z = self.ista(features)
            return z

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct features from sparse codes.

        Args:
            query: [B, D] query features

        Returns:
            [B, D] reconstructed features + query (residual)
        """
        z = self.ista(query)
        reconstructed = z @ self.dictionary  # [B, D]
        return reconstructed + query  # Residual

    def train_dictionary(self, features: torch.Tensor, n_steps: int = 10) -> float:
        """
        Update dictionary atoms via KSVD-like alternating minimization.

        Returns reconstruction loss.
        """
        with torch.no_grad():
            z = self.ista(features)  # [B, K]

            for step in range(n_steps):
                # Sparse coding step (already done in ista)
                # Dictionary update: project features onto used atoms
                for k in range(self.num_atoms):
                    if (z[:, k].abs() < 1e-6).all():
                        continue
                    # Update only this atom using samples that use it
                    mask = z[:, k].abs() > 1e-6
                    if mask.sum() < 2:
                        continue

                    # Reconstruction without atom k
                    residual = (features[mask]
                                - z[mask] @ self.dictionary
                                + z[mask, k:k+1] * self.dictionary[k:k+1])

                    # Update atom k
                    self.dictionary.data[k] = F.normalize(residual.mean(0), dim=0)

            return F.mse_loss(z @ self.dictionary, features).item()

    def get_compression_ratio(self) -> float:
        """Get compression ratio vs dense storage."""
        dense_size = self.feature_dim  # float32 = 4 bytes per value
        sparse_size = self.sparsity * 8  # 8 bytes per (index, value) pair
        return dense_size / sparse_size

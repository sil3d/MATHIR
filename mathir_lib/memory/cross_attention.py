"""
Cross-Attention Memory — Learned addressing.
Replaces cosine similarity with learned Q/K/V projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossAttentionMemory(nn.Module):
    """
    Memory with cross-attention addressing.

    Replaces fixed cosine similarity with learned Q/K/V projections.
    Better than cosine because it learns the task-specific similarity metric.

    Score: alpha_i = softmax((W_Q q)^T (W_K m_i) / sqrt(d))
    Retrieval: x_hat = sum alpha_i * (W_V m_i)
    """

    def __init__(self, capacity: int = 1000, feature_dim: int = 272,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # Memory store
        self.register_buffer("values", torch.zeros(capacity, feature_dim))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        # Learned Q/K/V projections
        self.w_q = nn.Linear(feature_dim, feature_dim)
        self.w_k = nn.Linear(feature_dim, feature_dim)
        self.w_v = nn.Linear(feature_dim, feature_dim)
        self.w_o = nn.Linear(feature_dim, feature_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def store(self, features: torch.Tensor) -> None:
        """Store features in memory."""
        with torch.no_grad():
            ptr = int(self.ptr.item() if torch.is_tensor(self.ptr) else self.ptr)
            count = int(self.count.item() if torch.is_tensor(self.count) else self.count)

            idx = ptr % self.capacity
            self.values[idx] = features.detach().mean(0)

            self.ptr = torch.tensor((ptr + 1) % self.capacity, dtype=torch.long, device=self.values.device)
            self.count = torch.tensor(min(count + 1, self.capacity), dtype=torch.long, device=self.values.device)

    def retrieve(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Retrieve using cross-attention.

        Args:
            query: [B, D] query
            k: top-k memories to attend to

        Returns:
            [B, D] retrieved + query (residual)
        """
        count = self.count.item() if torch.is_tensor(self.count) else self.count
        if count < 1:
            return query

        B = query.size(0)

        # Q from query, K/V from memory
        Q = self.w_q(query).unsqueeze(1)  # [B, 1, D]
        K = self.w_k(self.values[:count]).unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        V = self.w_v(self.values[:count]).unsqueeze(0).expand(B, -1, -1)  # [B, N, D]

        # Multi-head reshape
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, head_dim]
        K = K.view(B, count, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        V = V.view(B, count, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]

        # Attention scores
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, 1, N]

        # Top-k masking
        if k < count:
            top_k_scores, top_k_idx = scores.topk(k, dim=-1)  # [B, H, 1, k]
            mask = torch.full_like(scores, float('-inf'))
            mask.scatter_(-1, top_k_idx, top_k_scores)
            scores = mask

        attn = F.softmax(scores, dim=-1)  # [B, H, 1, N]
        attn = self.dropout(attn)

        # Apply attention to V
        out = (attn @ V).transpose(1, 2).contiguous().view(B, 1, self.feature_dim)  # [B, 1, D]
        out = self.w_o(out).squeeze(1)  # [B, D]

        # Residual + layer norm
        return self.layer_norm(out + query)

    def get_attention_weights(self, query: torch.Tensor) -> torch.Tensor:
        """Get attention weights for inspection."""
        count = self.count.item() if torch.is_tensor(self.count) else self.count
        if count < 1:
            return torch.zeros(query.size(0), 0)

        B = query.size(0)
        Q = self.w_q(query).unsqueeze(1)
        K = self.w_k(self.values[:count]).unsqueeze(0).expand(B, -1, -1)
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, count, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        return F.softmax(scores.squeeze(2), dim=-1)  # [B, H, N]

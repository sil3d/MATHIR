"""
Working Memory — Immediate context (last N steps).
Uses circular buffer with multi-head attention retrieval.
"""

import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WorkingMemory(nn.Module):
    """
    Working memory: circular buffer with attention-based retrieval.
    
    Capacity: configurable (default 64 slots)
    Update: every step (overwrite oldest)
    Retrieval: multi-head attention over stored items
    """
    
    def __init__(self, capacity: int = 64, feature_dim: int = 272, num_heads: int = 4):
        super().__init__()
        self._lock = threading.RLock()
        self.capacity = capacity
        self.feature_dim = feature_dim
        
        # Circular buffer
        self.register_buffer("buffer", torch.zeros(capacity, feature_dim))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        
        # Multi-head attention for retrieval
        self.attention = nn.MultiheadAttention(
            feature_dim, num_heads=num_heads, batch_first=True, dropout=0.1
        )
    
    def store(self, features: torch.Tensor) -> None:
        """Store features in circular buffer."""
        with self._lock:
            with torch.no_grad():
                batch_size = features.size(0)
                indices = (self.ptr + torch.arange(batch_size, device=features.device)) % self.capacity
                self.buffer[indices] = features.detach()
                self.ptr = (self.ptr + batch_size) % self.capacity
    
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve context using attention over buffer.
        
        Args:
            query: [B, D] tensor to retrieve for
            
        Returns:
            [B, D] retrieved context + query (residual)
        """
        with self._lock:
            stored = min(self.ptr.item(), self.capacity)
            if stored == 0:
                return query  # No memory yet, return query
            
            context = self.buffer[:stored].unsqueeze(0).expand(query.size(0), -1, -1)
            out, _ = self.attention(query.unsqueeze(1), context, context)
            return out.squeeze(1) + query  # Residual
    
    def reset(self) -> None:
        """Reset working memory."""
        self.buffer.zero_()
        self.ptr = torch.tensor(0, dtype=torch.long)
    
    def get_usage(self) -> int:
        """Get number of slots used."""
        return min(self.ptr.item(), self.capacity)

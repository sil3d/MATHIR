"""
Episodic Memory — Past experiences.
Uses key-value store with cosine similarity retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EpisodicMemory(nn.Module):
    """
    Episodic memory: key-value store with similarity retrieval.
    
    Capacity: configurable (default 1000 slots)
    Update: on event (when store() is called)
    Retrieval: top-k cosine similarity
    """
    
    def __init__(self, capacity: int = 1000, feature_dim: int = 272, key_dim: int = 64):
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        
        # Key-value store
        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, feature_dim))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        
        # Encoder: features → keys
        self.encoder = nn.Linear(feature_dim, key_dim)
    
    def store(self, features: torch.Tensor) -> None:
        """
        Store features as an episodic memory.
        
        Args:
            features: [B, D] tensor to store
        """
        with torch.no_grad():
            key = self.encoder(features.detach().mean(0, keepdim=True)).squeeze(0)
            idx = self.ptr % self.capacity
            self.keys[idx] = key.detach()
            self.values[idx] = features.detach().mean(0)
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.count = torch.minimum(self.count + 1, torch.tensor(self.capacity, dtype=torch.long))
    
    def retrieve(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Retrieve top-k most similar memories.
        
        Args:
            query: [B, D] tensor to search for
            k: number of memories to retrieve
            
        Returns:
            [B, D] averaged retrieved values + query (residual)
        """
        count = self.count.item()
        if count < k:
            return query
        
        key = self.encoder(query)
        sims = F.cosine_similarity(
            key.unsqueeze(1),
            self.keys[:count].unsqueeze(0),
            dim=-1
        )
        top_k = sims.topk(min(k, count), dim=1)[1]
        retrieved = self.values[top_k].mean(1)
        return retrieved + query  # Residual
    
    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search for top-k most similar memories.
        
        Returns:
            (indices, similarities)
        """
        count = self.count.item()
        if count == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0)
        
        key = self.encoder(query)
        sims = F.cosine_similarity(
            key.unsqueeze(1),
            self.keys[:count].unsqueeze(0),
            dim=-1
        )
        top_k = sims.topk(min(k, count), dim=1)
        return top_k.indices, top_k.values
    
    def forget(self, threshold: float = 0.1) -> int:
        """
        Prune low-similarity memories.
        
        Returns:
            number of memories kept
        """
        count = self.count.item()
        if count < 2:
            return count
        
        with torch.no_grad():
            sims = F.cosine_similarity(
                self.keys[:count].unsqueeze(1),
                self.keys[:count].unsqueeze(0),
                dim=-1
            )
            usage = sims.mean(dim=1)
            mask = usage > threshold
            
            kept = int(mask.sum().item())
            if kept < count:
                self.keys[:kept] = self.keys[:count][mask]
                self.values[:kept] = self.values[:count][mask]
                self.count = torch.tensor(kept, dtype=torch.long)
                self.ptr = torch.tensor(kept, dtype=torch.long)
                return kept
        return count
    
    def reset(self) -> None:
        """Reset episodic memory."""
        self.keys.zero_()
        self.values.zero_()
        self.ptr = torch.tensor(0, dtype=torch.long)
        self.count = torch.tensor(0, dtype=torch.long)
    
    def get_usage(self) -> int:
        """Get number of memories stored."""
        return self.count.item()

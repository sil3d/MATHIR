"""
Episodic Memory — Past experiences.
Uses key-value store with cosine similarity retrieval.
"""

import threading

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

    Eviction: score-based — when capacity is full, the slot with the
    lowest running score (mean cosine to the memory bank centroid) is
    evicted before inserting the new memory.  This replaces the blind
    FIFO circular-buffer overwrite and preserves high-similarity
    memories for longer.

    Complexity:
        store():  O(capacity)  — one cosine scan for eviction scoring
        retrieve(): O(capacity * d) — batch cosine via torch
        search(): O(capacity * d) — same batch cosine + topk
        forget(): O(capacity^2) — pairwise (run infrequently)
    """

    def __init__(self, capacity: int = 1000, feature_dim: int = 272, key_dim: int = 64):
        super().__init__()
        self._lock = threading.RLock()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.key_dim = key_dim

        # Key-value store
        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, feature_dim))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        # Per-slot eviction scores (mean cosine to centroid — O(capacity) to update)
        self.register_buffer("slot_scores", torch.zeros(capacity))
        # Running centroid of stored keys (for fast score estimation)
        self.register_buffer("key_centroid", torch.zeros(key_dim))
        self.register_buffer("_centroid_n", torch.tensor(0, dtype=torch.long))

        # Encoder: features → keys
        self.encoder = nn.Linear(feature_dim, key_dim)
    
    def store(self, features: torch.Tensor) -> None:
        """
        Store features as an episodic memory.

        When capacity is full, evicts the slot with the lowest running
        score (least similar to the centroid of stored keys) before
        inserting.  This is O(capacity) per call — dominated by the
        cosine scan over existing keys.

        Args:
            features: [B, D] tensor to store
        """
        with self._lock:
            with torch.no_grad():
                key = self.encoder(features.detach().mean(0, keepdim=True)).squeeze(0)
                count = self.count.item()

                if count >= self.capacity:
                    # Score-based eviction: find the slot with the lowest score
                    # (least representative of the memory bank).
                    evict_idx = self.slot_scores[:count].argmin().item()
                else:
                    evict_idx = self.ptr.item() % self.capacity

                self.keys[evict_idx] = key.detach()
                self.values[evict_idx] = features.detach().mean(0)

                # Update running centroid (Welford-style incremental mean)
                n = self._centroid_n.item()
                self.key_centroid = (self.key_centroid * n + key.detach()) / (n + 1)
                self._centroid_n = torch.tensor(n + 1, dtype=torch.long)

                # Update slot score: cosine similarity to updated centroid
                centroid_norm = self.key_centroid.norm()
                if centroid_norm > 1e-8:
                    key_norm = key.detach().norm()
                    if key_norm > 1e-8:
                        self.slot_scores[evict_idx] = (
                            F.cosine_similarity(
                                key.detach().unsqueeze(0),
                                self.key_centroid.unsqueeze(0),
                                dim=-1,
                            ).item()
                        )

                self.ptr = torch.tensor(
                    (evict_idx + 1) % self.capacity, dtype=torch.long
                )
                self.count = torch.minimum(
                    self.count + 1, torch.tensor(self.capacity, dtype=torch.long)
                )
    
    def retrieve(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Retrieve top-k most similar memories.
        
        Args:
            query: [B, D] tensor to search for
            k: number of memories to retrieve
            
        Returns:
            [B, D] averaged retrieved values + query (residual)
        """
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
                    # Reset eviction scoring state after compaction — the old
                    # slot_scores and centroid are stale.  Recompute the centroid
                    # from the surviving keys so future eviction decisions are
                    # based on accurate data.
                    self.slot_scores.zero_()
                    self.key_centroid = self.keys[:kept].mean(dim=0)
                    self._centroid_n = torch.tensor(kept, dtype=torch.long)
                    return kept
            return count
    
    def reset(self) -> None:
        """Reset episodic memory."""
        self.keys.zero_()
        self.values.zero_()
        self.slot_scores.zero_()
        self.key_centroid.zero_()
        self._centroid_n = torch.tensor(0, dtype=torch.long)
        self.ptr = torch.tensor(0, dtype=torch.long)
        self.count = torch.tensor(0, dtype=torch.long)
    
    def get_usage(self) -> int:
        """Get number of memories stored."""
        return self.count.item()

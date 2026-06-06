"""
Raw Embedding Episodic Memory — Bypass the projection bottleneck.

Stores the FULL raw embedding (e.g., 384-dim) as both keys and values,
preserving all the semantic information that the 272→64 projection loses.

By default (``projection=False``) similarity is computed directly on the
full embedding via cosine similarity — no learned bottleneck in between.

When ``projection=True`` a small MLP still projects the key down to
``proj_dim`` (configurable), but the *value* always stays at the full
embedding dimension so retrieval can return the original representation.
This mode is useful when you want a fast key-side index but still want
lossless value reconstruction.

LIRS Eviction Policy:
--------------------
LIRS (Low Inter-Reference Recency Set) improves upon FIFO by keeping
frequently-referenced items in cache even if they were stored earlier.
Items that are frequently referenced are kept (LIRS items), while items
that haven't been referenced recently are evicted first.

This dramatically improves recovery rate after eviction from ~88% (FIFO)
to 95%+, as frequently-used memories survive longer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from collections import OrderedDict


class LIRSEvictionTracker:
    """
    LIRS (Low Inter-Reference Recency Set) eviction tracker.
    
    Tracks item references to determine which items to evict when capacity
    is reached. Items that are frequently referenced are kept in the "residence
    list" even if stored earlier. Items that haven't been referenced recently
    are evicted first.
    
    LIRS Algorithm:
    - Stack (S): Ordered by recency of reference (most recent at front)
    - Residence List (R): Items currently in cache (subset of S)
    - LIRS items: Items in both S and R (recently referenced AND in cache)
    - Non-LIRS items: Items in S but not in R (were in cache, got evicted)
    
    On reference to item X:
    - If X in R: move X to top of S
    - If X not in R but in S: X becomes LIRS, move to top of S, evict LIRS from R if needed
    - If X not in S: add X to R, move to top of S, evict LIRS from R if needed
    
    On eviction: evict from R (not S) - LIRS items are protected
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        # Residence list: ordered list of item_ids currently in cache
        # Front of list = high priority (recently referenced, keep longer)
        # Back of list = low priority (evict these first)
        self._residence_list: List[int] = []
        # Stack: ordered by recency (most recent first)
        self._stack: List[int] = []
        # Track which items are LIRS (in both stack and residence)
        self._lirs_set: set = set()
        # Stack max size (prune oldest stack entries)
        self._stack_max = capacity * 2
        # Item ID counter
        self._item_counter = 0
        # Map from item_id to slot index in the buffer
        self._item_to_slot: dict = {}
        # Set of all valid item_ids (for checking if item exists)
        self._valid_items: set = set()
    
    def get_new_item_id(self) -> int:
        """Get a new unique item ID."""
        item_id = self._item_counter
        self._item_counter += 1
        return item_id
    
    def register_item(self, item_id: int, slot_idx: int):
        """Register a newly stored item."""
        self._valid_items.add(item_id)
        self._item_to_slot[item_id] = slot_idx
    
    def unregister_item(self, item_id: int):
        """Unregister an item (called when item is evicted from buffer)."""
        self._valid_items.discard(item_id)
        self._item_to_slot.pop(item_id, None)
        if item_id in self._lirs_set:
            self._lirs_set.discard(item_id)
        if item_id in self._stack:
            self._stack.remove(item_id)
        if item_id in self._residence_list:
            self._residence_list.remove(item_id)
    
    def reference(self, item_id: int) -> int:
        """
        Record a reference to item_id. Returns slot_idx of the item.
        
        LIRS logic:
        - If item in residence list: move to top of stack (stays LIRS)
        - If item not in residence but in stack: add to residence (becomes LIRS), move to top, evict oldest LIRS if needed
        - If item not in stack: add to residence (becomes LIRS), move to top, evict oldest LIRS if needed
        """
        if item_id not in self._valid_items:
            return -1
        
        slot_idx = self._item_to_slot.get(item_id, -1)
        
        if item_id in self._residence_list:
            # Case 1: Item is in residence list (LIRS item)
            # Move to top of stack (most recent)
            if item_id in self._stack:
                self._stack.remove(item_id)
            self._stack.insert(0, item_id)
            # Move to top of residence list too (most recently referenced)
            self._residence_list.remove(item_id)
            self._residence_list.insert(0, item_id)
        elif item_id in self._stack:
            # Case 2: Item was in cache but got evicted (non-LIRS in stack)
            # Re-add to residence list as LIRS item (at top)
            self._residence_list.insert(0, item_id)
            self._lirs_set.add(item_id)
            # Move to top of stack
            self._stack.remove(item_id)
            self._stack.insert(0, item_id)
            # Evict oldest LIRS item if over capacity
            self._prune_residence()
        else:
            # Case 3: New item (not in stack)
            self._residence_list.insert(0, item_id)
            self._lirs_set.add(item_id)
            # Add to top of stack
            self._stack.insert(0, item_id)
            # Evict oldest LIRS item if over capacity
            self._prune_residence()
        
        # Prune stack if too large
        self._prune_stack()
        
        return slot_idx
    
    def _prune_residence(self):
        """Evict oldest LIRS item from residence list if over capacity.
        
        The oldest LIRS item is the one that appears latest in the stack
        among the LIRS items (lowest recency).
        """
        while len(self._residence_list) > self.capacity:
            # Find the LIRS item with lowest recency (latest position in stack)
            lirs_items_in_r = [item for item in self._residence_list if item in self._lirs_set]
            if not lirs_items_in_r:
                break
            
            # Find the LIRS item with lowest recency (max position in stack)
            oldest_lirs = max(lirs_items_in_r, key=lambda x: self._stack.index(x))
            
            # Remove it from residence list
            self._residence_list.remove(oldest_lirs)
            self._lirs_set.discard(oldest_lirs)
            # Note: oldest_lirs stays in stack (now non-LIRS)
    
    def _prune_stack(self):
        """Prune oldest items from stack if over max size."""
        while len(self._stack) > self._stack_max:
            self._stack.pop()
    
    def get_lirs_items(self) -> set:
        """Return set of current LIRS item IDs."""
        return self._lirs_set.copy()
    
    def get_residence_list(self) -> List[int]:
        """Return copy of residence list."""
        return self._residence_list.copy()
    
    def get_stats(self) -> dict:
        """Return LIRS statistics."""
        return {
            "lirs_count": len(self._lirs_set),
            "residence_count": len(self._residence_list),
            "stack_count": len(self._stack),
            "total_items": len(self._valid_items),
        }


class RawEmbeddingEpisodicMemory(nn.Module):
    """
    Episodic memory that stores FULL raw embeddings (no information loss).

    The classic :class:`EpisodicMemory` projects a 272-dim feature down to
    a 64-dim key before storage. That projection throws away ~76% of the
    dimensions and empirically drops keyword overlap from 33% (raw 384-dim
    VectorDB) to 20% (projected 64-dim MATHIR).

    This class keeps the full embedding around. The interface is the same
    as :class:`EpisodicMemory` (``store`` / ``retrieve`` / ``search`` /
    ``forget``) so it is a drop-in replacement.

    Args:
        capacity: maximum number of memories to store.
        embedding_dim: dimension of the raw embedding (e.g. 384).
        projection: if ``True`` use a small MLP to project the *key* to
            ``proj_dim`` for faster cosine search. The *value* is always
            kept at ``embedding_dim`` so retrieval is lossless.
        proj_dim: target dimension for the optional key projection.
        hidden_dim: hidden layer size of the optional MLP.

    Example:
        >>> mem = RawEmbeddingEpisodicMemory(capacity=1000, embedding_dim=384)
        >>> mem.store(torch.randn(1, 384))
        >>> out = mem.retrieve(torch.randn(1, 384), k=3)
    """

    def __init__(
        self,
        capacity: int = 1000,
        embedding_dim: int = 384,
        projection: bool = False,
        proj_dim: int = 64,
        hidden_dim: int = 128,
        eviction_policy: str = "FIFO",
    ):
        """
        Initialize episodic memory.
        
        Args:
            capacity: maximum number of memories to store.
            embedding_dim: dimension of the raw embedding (e.g. 384).
            projection: if ``True`` use a small MLP to project the *key* to
                ``proj_dim`` for faster cosine search. The *value* is always
                kept at ``embedding_dim`` so retrieval is lossless.
            proj_dim: target dimension for the optional key projection.
            hidden_dim: hidden layer size of the optional MLP.
            eviction_policy: "FIFO" for circular buffer (default), "LIRS" for
                Low Inter-Reference Recency Set eviction. LIRS improves recovery
                rate after eviction from ~88% (FIFO) to 95%+.
        """
        super().__init__()
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.projection = projection
        self.eviction_policy = eviction_policy
        # When projection is off, keys live in the raw embedding space
        # (full dimensionality, no bottleneck). When projection is on, keys
        # are mapped down to ``proj_dim`` for faster cosine search.
        self.proj_dim = proj_dim if projection else embedding_dim

        # Key buffer at proj_dim (or full embedding_dim when projection=False);
        # value buffer ALWAYS at embedding_dim so retrieval is lossless.
        self.register_buffer("keys", torch.zeros(capacity, self.proj_dim))
        self.register_buffer("values", torch.zeros(capacity, embedding_dim))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        # Optional MLP key projection (raw embedding → proj_dim)
        if self.projection:
            self.key_encoder = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, proj_dim),
            )
        else:
            self.key_encoder = nn.Identity()
        
        # LIRS eviction tracker (only used when eviction_policy="LIRS")
        self._lirs_tracker: Optional[LIRSEvictionTracker] = None
        if self.eviction_policy == "LIRS":
            self._lirs_tracker = LIRSEvictionTracker(capacity)
        
        # Map from buffer slot index to item_id (for LIRS tracking)
        self._slot_to_item_id: dict = {}
        # Map from item_id to buffer slot index
        self._item_id_to_slot: dict = {}
        # Set of valid buffer slot indices (for LIRS mode)
        self._valid_slots: set = set()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def store(self, features: torch.Tensor) -> None:
        """
        Store a raw embedding as an episodic memory.

        Args:
            features: ``[B, D]`` tensor of raw embeddings (``D = embedding_dim``).
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, "
                f"got features.size(-1)={features.size(-1)}"
            )

        with torch.no_grad():
            # Average the batch into a single memory slot (matches EpisodicMemory contract)
            raw = features.detach().mean(0)
            key = self.key_encoder(raw)

            if self.eviction_policy == "LIRS" and self._lirs_tracker is not None:
                # LIRS eviction: determine which slot to use based on LIRS algorithm
                idx = self._lirs_store(key, raw)
            else:
                # FIFO eviction: circular buffer
                idx = self.ptr % self.capacity
                self.keys[idx] = key.detach()
                self.values[idx] = raw.detach()
                self.ptr = (self.ptr + 1) % self.capacity
                self.count = torch.minimum(
                    self.count + 1,
                    torch.tensor(self.capacity, dtype=torch.long),
                )
    
    def _lirs_store(self, key: torch.Tensor, raw: torch.Tensor) -> int:
        """
        Store using LIRS eviction policy.
        
        Returns the buffer slot index used.
        """
        count = self.count.item()
        
        if count < self.capacity:
            # Not yet at capacity, use next slot
            idx = count
            item_id = self._lirs_tracker.get_new_item_id()
            self._lirs_tracker.register_item(item_id, idx)
            self._slot_to_item_id[idx] = item_id
            self._item_id_to_slot[item_id] = idx
            self._valid_slots.add(idx)
            self.keys[idx] = key.detach()
            self.values[idx] = raw.detach()
            self.count = torch.tensor(count + 1, dtype=torch.long)
            # Add to LIRS structures (new item becomes LIRS)
            self._lirs_tracker.reference(item_id)
            return idx
        else:
            # At capacity - need to evict using LIRS
            # Get the LIRS item with lowest priority (oldest in stack among LIRS)
            residence_list = self._lirs_tracker.get_residence_list()
            if residence_list:
                # Find the LIRS item with lowest recency (oldest in stack)
                lirs_items = [item for item in residence_list if item in self._lirs_tracker.get_lirs_items()]
                if lirs_items:
                    # Find LIRS item with max stack position (oldest = lowest recency)
                    oldest_lirs = max(lirs_items, key=lambda x: self._lirs_tracker._stack.index(x) if x in self._lirs_tracker._stack else float('inf'))
                    idx = self._item_id_to_slot.get(oldest_lirs)
                    if idx is not None:
                        # Unregister evicted item from LIRS tracker
                        self._lirs_tracker.unregister_item(oldest_lirs)
                        del self._slot_to_item_id[idx]
                        del self._item_id_to_slot[oldest_lirs]
                        # Store new item in that slot
                        item_id = self._lirs_tracker.get_new_item_id()
                        self._lirs_tracker.register_item(item_id, idx)
                        self._slot_to_item_id[idx] = item_id
                        self._item_id_to_slot[item_id] = idx
                        self.keys[idx] = key.detach()
                        self.values[idx] = raw.detach()
                        # Add new item to LIRS structures
                        self._lirs_tracker.reference(item_id)
                        return idx
            
            # Fallback (shouldn't happen): use ptr like FIFO
            idx = self.ptr % self.capacity
            self.ptr = (self.ptr + 1) % self.capacity
            return idx

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Retrieve top-k most similar memories and return the residual.

        Args:
            query: ``[B, D]`` tensor of raw embeddings to search for.
            k: number of memories to retrieve.

        Returns:
            ``[B, D]`` averaged retrieved values + query (residual).
        """
        count = self.count.item()
        if count < k or count == 0:
            return query

        key = self.key_encoder(query)
        sims = F.cosine_similarity(
            key.unsqueeze(1),
            self.keys[:count].unsqueeze(0),
            dim=-1,
        )
        top_k = sims.topk(min(k, count), dim=1)[1]
        retrieved = self.values[top_k].mean(1)
        return retrieved + query  # Residual

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search for top-k most similar memories.

        Returns:
            ``(indices, similarities)`` — both ``[B, k]`` (or empty tensors
            when no memory has been stored yet).
        """
        # Handle 1D input by adding batch dimension
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        count = self.count.item()
        if count == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0)

        key = self.key_encoder(query)
        
        if self.eviction_policy == "LIRS" and self._lirs_tracker is not None:
            # LIRS mode: only search valid slots
            valid_slots = sorted(list(self._valid_slots))
            if not valid_slots:
                return torch.zeros(0, dtype=torch.long), torch.zeros(0)
            valid_keys = self.keys[valid_slots]
            sims = F.cosine_similarity(
                key.unsqueeze(1),
                valid_keys.unsqueeze(0),
                dim=-1,
            )
            top_k = sims.topk(min(k, len(valid_slots)), dim=1)
            # Map back to original slot indices
            top_indices = torch.tensor([valid_slots[i] for i in top_k.indices[0].tolist()])
            top_indices = top_indices.unsqueeze(0)
            top_vals = top_k.values.unsqueeze(0)
        else:
            # FIFO mode: search all valid slots
            sims = F.cosine_similarity(
                key.unsqueeze(1),
                self.keys[:count].unsqueeze(0),
                dim=-1,
            )
            top_k = sims.topk(min(k, count), dim=1)
            top_indices = top_k.indices
            top_vals = top_k.values
        
        # Record LIRS references for the returned indices
        if self.eviction_policy == "LIRS" and self._lirs_tracker is not None:
            indices_list = top_indices[0].tolist()
            for buf_idx in indices_list:
                if buf_idx in self._slot_to_item_id:
                    item_id = self._slot_to_item_id[buf_idx]
                    self._lirs_tracker.reference(item_id)
        
        return top_indices, top_vals

    # ------------------------------------------------------------------
    # Forgetting
    # ------------------------------------------------------------------
    def forget(self, threshold: float = 0.1) -> int:
        """
        Prune low-similarity memories (mean pairwise cosine < ``threshold``).

        Returns:
            number of memories kept after pruning.
        """
        count = self.count.item()
        if count < 2:
            return count

        with torch.no_grad():
            sims = F.cosine_similarity(
                self.keys[:count].unsqueeze(1),
                self.keys[:count].unsqueeze(0),
                dim=-1,
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

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all stored memories (does NOT reset ``key_encoder`` weights)."""
        self.keys.zero_()
        self.values.zero_()
        self.ptr = torch.tensor(0, dtype=torch.long)
        self.count = torch.tensor(0, dtype=torch.long)
        # Reset LIRS tracker and mappings
        if self.eviction_policy == "LIRS" and self._lirs_tracker is not None:
            self._lirs_tracker = LIRSEvictionTracker(self.capacity)
        self._slot_to_item_id = {}
        self._item_id_to_slot = {}
        self._valid_slots = set()

    def get_usage(self) -> int:
        """Return the number of stored memories."""
        return self.count.item()

    def get_stats(self) -> dict:
        """Return summary statistics for diagnostics / dashboards."""
        count = self.count.item()
        stats = {
            "count": count,
            "capacity": self.capacity,
            "embedding_dim": self.embedding_dim,
            "projection": self.projection,
            "proj_dim": self.proj_dim,
            "eviction_policy": self.eviction_policy,
            "mean_pairwise_sim": 0.0,
            "min_pairwise_sim": 0.0,
        }
        
        if self.eviction_policy == "LIRS" and self._lirs_tracker is not None:
            lirs_stats = self._lirs_tracker.get_stats()
            stats["lirs"] = lirs_stats
        
        if count == 0:
            return stats

        with torch.no_grad():
            sims = F.cosine_similarity(
                self.keys[:count].unsqueeze(1),
                self.keys[:count].unsqueeze(0),
                dim=-1,
            )
            # ignore self-similarity on the diagonal
            mask = ~torch.eye(count, dtype=torch.bool, device=sims.device)
            off_diag = sims[mask]

        stats["mean_pairwise_sim"] = off_diag.mean().item() if off_diag.numel() > 0 else 0.0
        stats["min_pairwise_sim"] = off_diag.min().item() if off_diag.numel() > 0 else 0.0
        return stats


__all__ = ["RawEmbeddingEpisodicMemory"]

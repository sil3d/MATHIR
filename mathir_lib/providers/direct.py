"""
Direct Embedding Provider
For when you already have embeddings (local LLM, pre-computed, etc.).
"""

import torch
from typing import List
from .base import EmbeddingProvider


class DirectProvider(EmbeddingProvider):
    """
    Direct tensor input provider.
    
    Use this when you already have embeddings from a local LLM.
    No network calls, no API — just pass tensors directly.
    
    The embedding_dim is inferred from the first tensor passed.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model_name = config.get("model", "user-provided") if config else "user-provided"
        self._dim = None
    
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Direct provider doesn't embed text — raises NotImplementedError.
        Use embed_tensor() instead.
        """
        raise NotImplementedError(
            "DirectProvider doesn't embed text. "
            "Use embed_tensor() with pre-computed embeddings."
        )
    
    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """Not implemented for DirectProvider."""
        raise NotImplementedError(
            "DirectProvider doesn't embed text. "
            "Use embed_batch_tensor() with pre-computed embeddings."
        )
    
    def embed_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pass through a pre-computed embedding tensor.
        
        Args:
            tensor: [B, D] or [D] tensor
            
        Returns:
            same tensor (with dim recorded)
        """
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if self._dim is None:
            self._dim = tensor.size(-1)
        return tensor
    
    def embed_batch_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Alias for embed_tensor() for API consistency."""
        return self.embed_tensor(tensor)
    
    def provider_id(self) -> tuple:
        return ("direct", self.model_name, self._dim)
    
    def embedding_dim(self) -> int:
        if self._dim is None:
            raise ValueError(
                "Embedding dim not set. Call embed_tensor() first."
            )
        return self._dim

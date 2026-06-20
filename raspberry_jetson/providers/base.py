"""
Embedding Provider Base Class
Abstract interface for getting embeddings from any LLM.
"""

from abc import ABC, abstractmethod
from typing import List
import torch


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    All providers must implement:
        - provider_id() → (provider_name, model_name, embedding_dim)
        - embed_text(text) → [1, D] tensor
        - embed_batch(texts) → [B, D] tensor
        - embedding_dim() → int
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    @abstractmethod
    def provider_id(self) -> tuple:
        """
        Returns a (provider_name, model_name, embedding_dim) tuple.
        
        Returns:
            tuple: (provider_name, model_name, embedding_dim)
        """
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Get embedding for a single text string.
        
        Args:
            text: input text
            
        Returns:
            [1, D] tensor
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: list of input texts
            
        Returns:
            [B, D] tensor
        """
        pass
    
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass
    
    def __repr__(self) -> str:
        try:
            d = self.embedding_dim()
        except (ValueError, NotImplementedError):
            d = "?"
        return f"{self.__class__.__name__}(dim={d})"

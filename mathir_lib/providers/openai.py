"""
OpenAI Embedding Provider
Uses OpenAI's text-embedding-3-* models.
"""

import os
from typing import List
import torch
from .base import EmbeddingProvider


class OpenAIProvider(EmbeddingProvider):
    """
    OpenAI embedding provider.
    
    Requires OPENAI_API_KEY environment variable (or pass api_key in config).
    
    Config:
        api_key: OpenAI API key (default: from env)
        model: "text-embedding-3-small" (1536) or "text-embedding-3-large" (3072)
    """
    
    # Known model dimensions
    MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.model = self.config.get("model", "text-embedding-3-small")
        self._client = None
        self._dim = self.MODEL_DIMS.get(self.model)
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. "
                "Set OPENAI_API_KEY env var or pass api_key in config."
            )
    
    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
        return self._client
    
    def embed_text(self, text: str) -> torch.Tensor:
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=text,
        )
        emb = response.data[0].embedding
        return torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
    
    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=texts,
        )
        embeddings = [d.embedding for d in response.data]
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def provider_id(self) -> tuple:
        return ("openai", self.model, self._dim)
    
    def embedding_dim(self) -> int:
        return self._dim

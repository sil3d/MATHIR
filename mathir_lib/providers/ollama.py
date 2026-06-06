"""
Ollama Embedding Provider
Uses Ollama's /api/embeddings endpoint.
"""

import json
import urllib.request
import urllib.error
from typing import List
import torch
from .base import EmbeddingProvider


class OllamaProvider(EmbeddingProvider):
    """
    Ollama embedding provider.
    
    Requires Ollama running locally (or remote).
    Default: http://localhost:11434
    
    Config:
        url: Ollama server URL (default: "http://localhost:11434")
        model: model name (default: "llama3.2:3b")
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.url = self.config.get("url", "http://localhost:11434")
        self.model = self.config.get("model", "llama3.2:3b")
        self._dim = None
        self._check_available()
    
    def _check_available(self) -> None:
        """Check if Ollama is reachable."""
        try:
            req = urllib.request.Request(f"{self.url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                pass
        except (urllib.error.URLError, OSError):
            pass  # Silent fail — will error on actual embed call
    
    def _call_api(self, text: str) -> List[float]:
        """Call Ollama /api/embeddings endpoint."""
        data = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/api/embeddings",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["embedding"]
    
    def embed_text(self, text: str) -> torch.Tensor:
        emb = self._call_api(text)
        if self._dim is None:
            self._dim = len(emb)
        return torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
    
    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        embeddings = [self._call_api(t) for t in texts]
        if self._dim is None:
            self._dim = len(embeddings[0])
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def provider_id(self) -> tuple:
        return ("ollama", self.model, self._dim)
    
    def embedding_dim(self) -> int:
        if self._dim is None:
            # Try to get dim by embedding a test string
            self.embed_text("test")
        return self._dim

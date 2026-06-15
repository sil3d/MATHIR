"""
MATHIR Embedding Providers
Abstraction layer for getting embeddings from any LLM.
"""

from .base import EmbeddingProvider
from .direct import DirectProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .huggingface import HuggingFaceProvider
from .onnx import ONNXProvider


def get_provider(name: str, config: dict = None) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider by name.
    
    Args:
        name: "direct" | "ollama" | "openai" | "huggingface" | "onnx"
        config: provider-specific config dict
        
    Returns:
        EmbeddingProvider instance
    """
    providers = {
        "direct": DirectProvider,
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "huggingface": HuggingFaceProvider,
        "onnx": ONNXProvider,
    }
    
    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers.keys())}")
    
    return providers[name](config or {})


def get_embedding_with_provider(provider: EmbeddingProvider, texts: list[str]):
    """
    Convenience factory that returns (embeddings, provider_id) tuple.
    
    Args:
        provider: EmbeddingProvider instance
        texts: list of texts to embed
        
    Returns:
        tuple: (embeddings tensor, provider_id tuple)
    """
    embeddings = provider.embed_batch(texts)
    return embeddings, provider.provider_id()


__all__ = [
    "EmbeddingProvider",
    "DirectProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "HuggingFaceProvider",
    "get_provider",
]

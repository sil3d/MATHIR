"""
ONNX Embedding Provider
Uses ONNX Runtime for fast inference with quantized models.
"""

from typing import List
import numpy as np
from .base import EmbeddingProvider


class ONNXProvider(EmbeddingProvider):
    """
    ONNX embedding provider.
    
    Uses ONNX Runtime for fast inference with quantized models.
    Supports INT8 quantization for smaller memory footprint.
    
    Config:
        model_dir: path to model directory (required)
        provider: "CPUExecutionProvider" | "DmlExecutionProvider" (default: "CPUExecutionProvider")
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model_dir = self.config.get("model_dir")
        self.provider = self.config.get("provider", "CPUExecutionProvider")
        
        self._session = None
        self._tokenizer = None
        self._dim = None
    
    def _load_model(self):
        """Load ONNX model and tokenizer."""
        if self._session is not None:
            return
        
        from pathlib import Path
        from onnxruntime import InferenceSession
        from transformers import AutoTokenizer
        
        model_path = Path(self.model_dir) / "model.int8.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        self._session = InferenceSession(
            str(model_path),
            providers=[self.provider, 'CPUExecutionProvider']
        )
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        # Get dim from config or model
        config_path = Path(self.model_dir) / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                config = json.load(f)
            self._dim = config.get("hidden_size", 1024)
        else:
            self._dim = 1024  # Default for Octen
    
    def embed_text(self, text: str) -> 'torch.Tensor':
        result = self.embed_batch([text])
        return result
    
    def embed_batch(self, texts: List[str]) -> 'torch.Tensor':
        import torch
        self._load_model()
        
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        
        outputs = self._session.run(None, dict(inputs))
        
        # Mean pooling
        attention_mask = inputs["attention_mask"][:, :, np.newaxis]
        embeddings = (outputs[0] * attention_mask).sum(axis=1) / attention_mask.sum(axis=1)
        
        # L2 normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return torch.from_numpy(embeddings.astype(np.float32))
    
    def provider_id(self) -> tuple:
        return ("onnx", self.model_dir, self._dim)
    
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._load_model()
        return self._dim

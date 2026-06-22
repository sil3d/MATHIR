"""
ONNX Embedder for MATHIR — uses Octen-Embedding-0.6B-ONNX-INT8 directly.
Faster and better quality than MiniLM for multilingual content.
"""

import numpy as np
from pathlib import Path
from typing import List, Union
import logging

log = logging.getLogger("mathir-onnx")


class OctenEmbedder:
    """Embedder using Octen-Embedding-0.6B-ONNX-INT8 via ONNX Runtime."""
    
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = str(Path(__file__).parent.parent / "models" / "octen-int8")
        
        self.model_dir = Path(model_dir)
        self._dim = 1024
        
        # Load immediately for better caching
        self._load()
    
    def _load(self):
        """Load model and tokenizer."""
        from onnxruntime import InferenceSession
        from transformers import AutoTokenizer
        
        model_path = self.model_dir / "model.int8.onnx"
        log.info(f"Loading Octen INT8 from {model_path}...")
        
        self._session = InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        log.info(f"Octen INT8 loaded: dim={self._dim}")
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts to embeddings."""
        self._load()
        
        if isinstance(texts, str):
            texts = [texts]
        
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
        
        return embeddings.astype(np.float32)
    
    def encode_to_tensor(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts to embeddings (alias for compatibility)."""
        return self.encode(texts, **kwargs)
    
    def __call__(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Make callable like SentenceTransformer."""
        return self.encode(texts, **kwargs)


def get_onnx_embedder(model_dir: str = None) -> OctenEmbedder:
    """Get or create ONNX embedder instance."""
    return OctenEmbedder(model_dir)


if __name__ == "__main__":
    # Quick test
    embedder = OctenEmbedder()
    texts = ["Hello world", "Test embedding", "Bonjour le monde"]
    embeddings = embedder.encode(texts)
    print(f"Shape: {embeddings.shape}")
    print(f"Norms: {np.linalg.norm(embeddings, axis=1)}")

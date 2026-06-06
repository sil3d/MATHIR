"""
HuggingFace Embedding Provider
Loads any open-weight model and extracts hidden states.
"""

from typing import List
import torch
from .base import EmbeddingProvider


class HuggingFaceProvider(EmbeddingProvider):
    """
    HuggingFace embedding provider.
    
    Loads any AutoModel and extracts last hidden state.
    Works with Qwen, LLaMA, Mistral, Gemma, DeepSeek, etc.
    
    Config:
        model: model name (default: "Qwen/Qwen2.5-7B-Instruct")
        device: "cpu" | "cuda" | "auto" (default: "auto")
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model_name = self.config.get("model", "Qwen/Qwen2.5-7B-Instruct")
        device_cfg = self.config.get("device", "auto")
        if device_cfg == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_cfg
        
        self._model = None
        self._tokenizer = None
        self._dim = None
    
    def _load_model(self):
        """Lazy-load model and tokenizer."""
        if self._model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self._model.eval()
                # Get embedding dim from model config
                self._dim = self._model.config.hidden_size
            except ImportError:
                raise ImportError(
                    "transformers package required. "
                    "Install with: pip install transformers"
                )
    
    def embed_text(self, text: str) -> torch.Tensor:
        result = self.embed_batch([text])
        return result
    
    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        self._load_model()
        
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        # Use last hidden state of last token (or mean-pool)
        hidden = outputs.last_hidden_state
        # Mean pooling (more robust than last token)
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        embeddings = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        return embeddings.cpu().float()
    
    def provider_id(self) -> tuple:
        return ("huggingface", self.model_name, self._dim)
    
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._load_model()
        return self._dim

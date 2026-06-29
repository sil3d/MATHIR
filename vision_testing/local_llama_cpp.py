#!/usr/bin/env python3
"""
LocalLlamaCppClient — Local GGUF inference via llama-cpp-python.

Supports:
- Pure text GGUF models (chat, completion)
- Multimodal GGUF (mmproj) models (vision input + text generation)

Why this exists alongside OpenRouterClient:
- Edge / offline / no-internet scenarios
- Privacy-sensitive data (memory never leaves the machine)
- Cost (free local inference vs paid cloud API)
- Latency (no network round-trip)

Install:
  pip install llama-cpp-python

If the package is not installed, this module still imports but `LlamaCppBackend`
raises a clear ImportError at instantiation time. The UI gracefully falls back
to OpenRouter when local backend is unavailable.

For multimodal (vision) GGUF:
  - model_path : path to the LLM GGUF (e.g. Qwen2-VL, Llama-3.2-Vision, etc.)
  - mmproj_path : path to the matching mmproj GGUF (vision encoder)

Configuration in config.json:
  "llama_local": {
    "models": {
      "qwen2-vl-7b-local": {
        "enabled": true,
        "type": "vision-language",
        "display_name": "Qwen2-VL 7B (local GGUF)",
        "path": "models/qwen2-vl-7b-instruct-q4_k_m.gguf",
        "mmproj": "models/qwen2-vl-7b-instruct-mmproj-f16.gguf",
        "context_length": 4096,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "supports_vision": true
      }
    }
  }
"""
import base64
import time
from pathlib import Path
from typing import List, Dict, Optional, Any


def _to_data_url(path: str, mime: str = "image/jpeg") -> str:
    """Read a local image file and return a data URL."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class LlamaCppBackend:
    """Wrapper around llama-cpp-python with multimodal (mmproj) support.

    The underlying library exposes two main classes:
    - `Llama` for text-only models
    - `Llama` with `chat_format="llava"` + mmproj_path for multimodal models

    Both share the same underlying engine. We abstract over the differences
    via a single `chat(messages, ...)` method that handles both text and image
    messages uniformly.
    """

    def __init__(self, model_paths: dict):
        try:
            from llama_cpp import Llama  # noqa: F401 — import check
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. Run: pip install llama-cpp-python\n"
                f"Original error: {exc}"
            )

        self.model_id = model_paths.get("name", "local-model")
        self.display_name = model_paths.get("display_name", self.model_id)
        self.path = model_paths.get("path")
        self.mmproj_path = model_paths.get("mmproj")
        self.context_length = int(model_paths.get("context_length", 4096))
        self.n_threads = int(model_paths.get("n_threads", 4))
        self.n_gpu_layers = int(model_paths.get("n_gpu_layers", 0))  # 0 = CPU only
        self.chat_format = model_paths.get("chat_format")  # auto-detect if None

        if not self.path:
            raise ValueError(f"Model {self.model_id} missing 'path' in config.json")
        if not Path(self.path).exists():
            raise FileNotFoundError(f"GGUF not found: {self.path}")

        # Lazy init: don't load the model until first chat() call
        self._llm = None
        self._model_kind = None  # 'text' or 'multimodal'

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._llm is not None:
            return
        from llama_cpp import Llama

        kwargs = dict(
            model_path=self.path,
            n_ctx=self.context_length,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        if self.mmproj_path and Path(self.mmproj_path).exists():
            kwargs["mmproj_path"] = self.mmproj_path
            # If chat_format not specified, let llama-cpp auto-detect
            if self.chat_format:
                kwargs["chat_format"] = self.chat_format
            self._model_kind = "multimodal"
        else:
            self._model_kind = "text"

        self._llm = Llama(**kwargs)

    def chat(self, messages, max_tokens=512, temperature=0.7) -> str:
        """Send chat messages and return the assistant's reply.

        Supports image content in messages (OpenAI-style content arrays).
        For multimodal models, images are passed via the standard mmproj path.
        """
        self._ensure_loaded()
        t0 = time.time()
        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response["choices"][0]["message"]["content"]
            return content or ""
        except Exception as exc:
            return f"ERROR (local llama.cpp): {exc}"
        finally:
            elapsed = time.time() - t0
            if hasattr(self, "_last_latency"):
                self._last_latency = elapsed

    def describe(self, image_path: str, prompt: str, max_tokens=512, temperature=0.7) -> str:
        """Describe an image (multimodal only — raises for text models)."""
        if self._model_kind != "multimodal":
            return "ERROR: current local model is text-only (no mmproj configured)"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _to_data_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)

    def stats(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "path": self.path,
            "mmproj": self.mmproj_path,
            "kind": self._model_kind or "not-loaded",
            "context_length": self.context_length,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
        }


def get_llama_cpp_available() -> bool:
    """True if llama-cpp-python is importable in this environment."""
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False

"""
MATHIR Config System
Loads and validates configuration from YAML files.

V6: All parameters config-driven. No hardcoded values in plugin.
"""

import os
import copy
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "memory": {
        "embedding_dim": 4096,
        "internal_dim": 272,
        "working_capacity": 64,
        "episodic_capacity": 1000,
        "semantic_prototypes": 256,
        "immunological_capacity": 100,
        "kl_coefficient": 0.001,  # Reduced10x to allow tier-specific routing
        "anomaly_threshold": 2.0,
        "decay_rates": [0.9, 0.7, 0.5],

        # V7 — high impact
        "episodic_type": "standard",      # "standard" | "ebbinghaus"
        "semantic_type": "standard",      # "standard" | "hyperbolic"
        "immune_type": "standard",        # "standard" | "mahalanobis"
        "use_variational": False,         # Replace episodic with variational
        "use_sparse_coding": False,       # Add sparse coding tier
        "use_cross_attention": False,     # Replace episodic with cross-attention
        "use_neural_ode": False,          # Add Neural ODE tier
        "use_infonce": False,             # Use InfoNCE for self-supervised learning

        # V7 hyperparameters
        "ebbinghaus_alpha": 0.5,          # Spaced repetition strength
        "sparse_atoms": 1088,             # Number of dictionary atoms
        "sparse_sparsity": 8,             # Non-zeros per code
        "variational_min_sigma": 0.01,    # Minimum uncertainty
        "neural_ode_dt": 0.1,             # ODE time step
        "neural_ode_steps": 3,            # Integration steps
        "infonce_temperature": 0.1,       # InfoNCE temperature

        # Raw-embedding episodic (Approach A: bypass the projection bottleneck)
        "use_raw_embedding": False,       # If True, swap EpisodicMemory for RawEmbeddingEpisodicMemory
        "raw_embedding_dim": 384,         # Dimension of the raw embedding (e.g., 384 for MiniLM)
        "raw_projection": False,          # If True, project the key but keep value at full dim
        "raw_proj_dim": 64,               # Target dim for the optional key projection
    },
    "compression": {
        "enabled": True,
        "method": "turboquant",
        "bits": 3,
        "episodic_only": False,
    },
    "inference": {
        "backend": "pytorch",
        "device": "auto",
        "device_map": None,  # Set to dict for hybrid mode (e.g., {"working": "cuda:0", "episodic": "cpu"})
        "precision": "float32",
    },
    "router": {
        "type": "kl_constrained",
        "num_memories": 4,
        "entropy_coefficient": 0.01,
        "temperature": 0.5,  # Lower temperature for sharper routing decisions
    },
    "providers": {
        "default": "direct",
        "ollama": {
            "url": "http://localhost:11434",
            "model": "llama3.2:3b",
        },
        "openai": {
            "api_key": None,  # From env: OPENAI_API_KEY
            "model": "text-embedding-3-small",
        },
        "huggingface": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "device": "auto",
        },
        "onnx": {
            "model_dir": None,  # Path to ONNX model directory
            "provider": "CPUExecutionProvider",
        },
    },
}


def get_default_config() -> Dict[str, Any]:
    """Return a deep copy of the default config."""
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config from YAML file, merged with defaults.

    If ``path`` is None or does not exist, the default config is returned.
    Otherwise the file is parsed with :func:`yaml.safe_load` and deep-merged
    on top of the defaults.
    """
    config = get_default_config()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config = merge_config(config, user_config)
    return config


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge ``override`` into ``base`` and return the result.

    Nested dictionaries are merged recursively; non-dict values in
    ``override`` replace the corresponding entries in ``base``.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: Dict[str, Any]) -> None:
    """Validate config values. Raise ``ValueError`` if invalid."""
    if config["memory"]["embedding_dim"] <= 0:
        raise ValueError("embedding_dim must be positive")
    if config["memory"]["working_capacity"] <= 0:
        raise ValueError("working_capacity must be positive")
    if config["compression"]["bits"] not in (2, 3, 4, 8, 16, 32):
        raise ValueError("compression bits must be 2, 3, 4, 8, 16, or 32")
    if config["inference"]["backend"] not in ("pytorch", "onnx", "rust"):
        raise ValueError("inference backend must be pytorch, onnx, or rust")
    if config["inference"]["precision"] not in ("float32", "float16", "int8"):
        raise ValueError("precision must be float32, float16, or int8")


__all__ = [
    "DEFAULT_CONFIG",
    "get_default_config",
    "load_config",
    "merge_config",
    "validate_config",
]

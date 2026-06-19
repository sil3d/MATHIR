"""
MATHIR Library — V7.2 (Memory-Augmented Tensor Hybrid with Intelligent Routing)
================================================================================

V7 is the current and only supported version of MATHIR.

Legacy V1-V5 code has been moved to `_deprecated/`.

Usage:
    from mathir_lib import MATHIRPluginV7
    plugin = MATHIRPluginV7(embedding_dim=4096)
    output = plugin.perceive(llm_embedding)

Version: 7.2
Author: MATHIR Research Team
"""

__version__ = "7.2.0"
__author__ = "MATHIR Research Team"

# ---------------------------------------------------------------------------
# V7 — Doctoral-grade memory plugin with 8 novel algorithms
# ---------------------------------------------------------------------------
from .plugin_v7 import MATHIRPluginV7

# ---------------------------------------------------------------------------
# Hybrid CPU/GPU Device Manager
# ---------------------------------------------------------------------------
from .hybrid_device import HybridDeviceManager

# ---------------------------------------------------------------------------
# V7.1 — Novel retrieval approaches
# ---------------------------------------------------------------------------
from .memory import RawEmbeddingEpisodicMemory  # Approach A: raw embedding
from .memory import HybridEpisodicMemory        # Approach D: BM25 + Dense + CE

# ---------------------------------------------------------------------------
# V7 Compression (TurboQuant)
# ---------------------------------------------------------------------------
from .compression import TurboQuantCompression

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
from .config import (
    get_default_config,
    load_config,
    merge_config,
    validate_config,
)

# ---------------------------------------------------------------------------
# Dynamic device detection
# ---------------------------------------------------------------------------
from .device_utils import detect_device, get_device_info, auto_device_map


__all__ = [
    # ===== V7 (Current) =====
    "MATHIRPluginV7",

    # ===== Hybrid Device Manager =====
    "HybridDeviceManager",

    # ===== V7.1 retrieval approaches =====
    "RawEmbeddingEpisodicMemory",
    "HybridEpisodicMemory",

    # ===== Core components =====
    "TurboQuantCompression",

    # ===== Device utilities =====
    "detect_device",
    "get_device_info",
    "auto_device_map",

    # ===== Config helpers =====
    "get_default_config",
    "load_config",
    "merge_config",
    "validate_config",
]

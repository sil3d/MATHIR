"""
MATHIR Library — V7.2 (Memory-Augmented Tensor Hybrid with Intelligent Routing)
=================================================================================

V7 is the current and only supported version of MATHIR.

Older architectures (V4 = `MATHIR`, V5 = `MATHIRv5`) have been moved to
`mathir_lib/legacy/` and are NOT exported by default. If you need them
for legacy code, import directly:

    from mathir_lib.legacy.mathir_V4 import MATHIR, MATHIRMemory
    from mathir_lib.legacy.mathir_v5 import MATHIRv5

Modern usage:
    from mathir_lib import MATHIRPluginV7
    plugin = MATHIRPluginV7(embedding_dim=4096)
    output = plugin.perceive(llm_embedding)

Version: 7.2
Author: MATHIR Research Team
"""

__version__ = "7.2.0"
__author__ = "MATHIR Research Team"

# ---------------------------------------------------------------------------
# V6 — Config-driven memory plugin
# ---------------------------------------------------------------------------
from .plugin import MATHIRPlugin

# ---------------------------------------------------------------------------
# V7 — Doctoral-grade memory plugin with 8 novel algorithms + 6 theorems
# ---------------------------------------------------------------------------
from .plugin_v7 import MATHIRPluginV7

# ---------------------------------------------------------------------------
# Hybrid CPU/GPU Device Manager
# ---------------------------------------------------------------------------
from .hybrid_device import HybridDeviceManager

# ---------------------------------------------------------------------------
# V7.1 — Novel retrieval approaches (closing the 12-14pp quality gap)
# ---------------------------------------------------------------------------
from .memory import RawEmbeddingEpisodicMemory  # Approach A: raw 384-dim
from .memory import HybridEpisodicMemory        # Approach D: BM25 + Dense + CE

# ---------------------------------------------------------------------------
# V6 Manifold-Constrained Hyper-Connections (V6 mHC shim)
# ---------------------------------------------------------------------------
from .mhc import ManifoldConstrainedLinearV2

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

    # ===== V6 (Still supported, LLM-agnostic API) =====
    "MATHIRPlugin",

    # ===== V7.1 retrieval approaches =====
    "RawEmbeddingEpisodicMemory",
    "HybridEpisodicMemory",

    # ===== Core components =====
    "ManifoldConstrainedLinearV2",
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


# ---------------------------------------------------------------------------
# Legacy V1-V5 architectures (V1-V3 in mathir_model.py, V4-V5 in legacy/)
# ---------------------------------------------------------------------------
# These are kept for backward compatibility with old research code.
# They are NOT part of the supported V7 API.
#
# If you need to import them:
#   from mathir_lib.legacy.mathir_V4 import MATHIR, MATHIRMemory
#   from mathir_lib.legacy.mathir_v5 import MATHIRv5
#
# For NEW projects, use MATHIRPluginV7 only.

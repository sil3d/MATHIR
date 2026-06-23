"""
mhc.py — V6 Manifold-Constrained Hyper-Connections shim
==========================================================

This module re-exports ManifoldConstrainedLinearV2 from the legacy
V4 implementation for backward compatibility with V6 plugin code.

The original implementation was in mhc_V4.py (DeepSeek-mHC v3).
V6 plugin.py imports from .mhc (not .mhc_V4), so this shim ensures
the import chain still works without keeping V4 at the top level.

For new code, use the V7 plugin directly:
    from mathir_lib import MATHIRPluginV7
"""

from .legacy.mhc_V4 import ManifoldConstrainedLinearV2
try:
    from .legacy.mhc_V4 import ManifoldConstrainedLinearV4  # alias
except ImportError:
    pass

__all__ = ["ManifoldConstrainedLinearV2"]

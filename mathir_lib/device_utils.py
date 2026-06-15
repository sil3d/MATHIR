"""
Dynamic device detection for MATHIR.
Auto-detects GPU/CPU and provides optimal device assignment.

Usage::

    from mathir_lib.device_utils import detect_device, get_device_info, auto_device_map

    device = detect_device()          # "cuda:0" or "cpu"
    info = get_device_info()          # dict with GPU details
    dmap = auto_device_map(768)       # tier→device map, or None on CPU
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def detect_device() -> str:
    """Detect the best available device.

    Returns
    -------
    str
        ``"cuda:0"`` if a CUDA GPU is available *and* has at least 1 GB
        of free VRAM; ``"cpu"`` otherwise.

    Notes
    -----
    The 1 GB free-VRAM threshold avoids selecting a GPU that is already
    saturated by other workloads.  When ``torch.cuda.mem_get_info`` is
    unavailable (e.g. ROCm without smi support) we fall back to
    ``torch.cuda.is_available()`` and assume the GPU is usable.
    """
    if torch.cuda.is_available():
        try:
            free_mem = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)  # GB
            if free_mem >= 1.0:
                return "cuda:0"
        except Exception:
            # mem_get_info failed — assume the GPU is usable anyway.
            return "cuda:0"
    return "cpu"


def get_device_info() -> Dict:
    """Get detailed device information.

    Returns
    -------
    dict
        Keys:

        * ``device``           — result of :func:`detect_device`
        * ``cuda_available``   — ``bool``
        * ``gpu_name``         — ``str | None``
        * ``gpu_total_vram_gb`` — ``float``
        * ``gpu_free_vram_gb``  — ``float``
    """
    info: Dict = {
        "device": detect_device(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_total_vram_gb": 0.0,
        "gpu_free_vram_gb": 0.0,
    }
    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            free = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
            info["gpu_total_vram_gb"] = round(total, 2)
            info["gpu_free_vram_gb"] = round(free, 2)
        except Exception:
            pass
    return info


def auto_device_map(embedding_dim: int = 768) -> Optional[Dict[str, str]]:
    """Generate an optimal device map based on available hardware.

    Strategy
    --------
    * **GPU with ≥ 1 GB free VRAM** — all components on GPU for
      maximum throughput (single-device avoids cross-device transfer
      overhead and parameter/tensor mismatches).
    * **CPU only** — everything on CPU; returns ``None`` (no map needed).

    Parameters
    ----------
    embedding_dim:
        The dimensionality of incoming embeddings.  Currently unused but
        reserved for future heuristics (e.g. choosing GPU vs CPU based on
        tensor size).

    Returns
    -------
    dict | None
        ``{tier_name: device_str}`` when GPU is available, ``None`` when
        CPU-only (meaning all components stay on CPU by default).
    """
    device = detect_device()
    if device == "cpu":
        return None  # Everything on CPU — no map needed.

    # GPU available: single-device strategy.
    # All components on GPU avoids cross-device tensor/parameter mismatches
    # and eliminates inter-device transfer overhead.
    return {
        "working": "cuda:0",
        "episodic": "cuda:0",
        "semantic": "cuda:0",
        "immune": "cuda:0",
        "router": "cuda:0",
        "input_proj": "cuda:0",
        "output_proj": "cuda:0",
    }


__all__ = ["detect_device", "get_device_info", "auto_device_map"]

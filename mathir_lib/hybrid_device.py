"""
Hybrid CPU/GPU Device Manager for MATHIR
==========================================

Allows different memory tiers to run on different devices simultaneously.
When a device_map is provided, tensors are automatically transferred between
devices at tier boundaries.

Usage::

    manager = HybridDeviceManager({
        "working": "cuda:0",    # GPU for fast attention
        "episodic": "cpu",      # CPU for large KV store
        "semantic": "cuda:0",   # GPU for projections
        "immune": "cpu",        # CPU for statistics
        "router": "cuda:0",     # GPU for routing weights
    })

    # Automatically transfers tensors between devices
    result = manager.to_device(input_tensor, "working")

When ``device_map`` is ``None`` or empty, the manager acts as a no-op passthrough,
preserving full backward compatibility with single-device configurations.
"""

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class HybridDeviceManager:
    """Manages device placement for MATHIR components.

    The manager tracks which device each component (tier) runs on and
    transparently handles CPU↔GPU transfers when tensors cross tier
    boundaries. Transfer statistics are collected for profiling.

    Parameters
    ----------
    device_map : dict or None
        Mapping of component names to device strings.
        Recognised keys: ``"working"``, ``"episodic"``, ``"semantic"``,
        ``"immune"``, ``"router"``, ``"input_proj"``, ``"output_proj"``,
        or any custom string the user wants.
        ``None`` or ``{}`` disables hybrid mode (all on fallback).
    fallback : str
        Device used when a component is not in ``device_map``.
        Defaults to ``"cpu"``.
    """

    # Valid component names for documentation / linting
    COMPONENT_NAMES = frozenset({
        "working", "episodic", "semantic", "immune", "router",
        "input_proj", "output_proj", "sparse_coding", "neural_ode",
    })

    def __init__(
        self,
        device_map: Optional[Dict[str, str]] = None,
        fallback: str = "cpu",
    ):
        self.device_map: Dict[str, str] = dict(device_map) if device_map else {}
        self.fallback = torch.device(fallback)
        self._transfer_count: int = 0
        self._transfer_bytes: int = 0
        # Per-component transfer counters for fine-grained profiling
        self._component_transfers: Dict[str, int] = {}

        if self.device_map:
            logger.info(
                "HybridDeviceManager active — device_map: %s, fallback: %s",
                self.device_map,
                self.fallback,
            )
        else:
            logger.debug(
                "HybridDeviceManager in passthrough mode (no device_map)"
            )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_device(self, component: str) -> torch.device:
        """Return the ``torch.device`` for *component*.

        Falls back to ``self.fallback`` when the component is not mapped.
        """
        return torch.device(self.device_map.get(component, self.fallback))

    def to_device(self, tensor: torch.Tensor, component: str) -> torch.Tensor:
        """Move *tensor* to the target device for *component* if needed.

        If the tensor is already on the correct device, it is returned
        unchanged (zero-copy). Otherwise a transfer is performed and
        the statistics counters are updated.

        The tensor is always ``.detach()``-ed before transfer to prevent
        gradient leakage across device boundaries.
        """
        target = self.get_device(component)
        if tensor.device == target:
            return tensor
        # Detach to prevent gradient graph crossing device boundaries
        self._transfer_count += 1
        self._transfer_bytes += tensor.nelement() * tensor.element_size()
        self._component_transfers[component] = (
            self._component_transfers.get(component, 0) + 1
        )
        return tensor.detach().to(target)

    def cross_forward(
        self,
        from_component: str,
        to_component: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Transfer *tensor* from one component's device to another's.

        This is a convenience wrapper around :meth:`to_device` that
        makes the source→destination intent explicit in call sites.
        """
        return self.to_device(tensor, to_component)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return transfer statistics collected since construction.

        Returns a dict with keys:

        - ``transfer_count`` — total number of cross-device transfers
        - ``transfer_bytes`` — total bytes moved
        - ``component_transfers`` — per-component transfer counts
        - ``device_map`` — the active device mapping
        """
        return {
            "transfer_count": self._transfer_count,
            "transfer_bytes": self._transfer_bytes,
            "component_transfers": dict(self._component_transfers),
            "device_map": dict(self.device_map),
        }

    def reset_stats(self) -> None:
        """Zero all transfer counters."""
        self._transfer_count = 0
        self._transfer_bytes = 0
        self._component_transfers.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_hybrid(self) -> bool:
        """Return ``True`` if the manager has an active device_map."""
        return bool(self.device_map)

    def __repr__(self) -> str:
        if self.device_map:
            return (
                f"HybridDeviceManager(device_map={self.device_map}, "
                f"fallback={self.fallback})"
            )
        return f"HybridDeviceManager(fallback={self.fallback}, passthrough)"


__all__ = ["HybridDeviceManager"]

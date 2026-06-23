"""
MATHIR Drop-in — Configuration.

A flat dict-of-dicts is intentionally used instead of a dataclass for two
reasons:

1. Users can edit the config in JSON / YAML and load it directly.
2. The plugin reads entries defensively (``cfg.get(...)``), so adding
   new keys never breaks old code.

The helpers ``configure()`` and ``get_default_config()`` are convenience
wrappers; the underlying constant is ``DEFAULT_CONFIG``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "memory": {
        # External embedding size — the dim of whatever model is producing
        # the vectors (e.g. 384 for MiniLM, 768 for BERT-base, 4096 for
        # large LLM hidden states). This is the ONE size the user MUST
        # set correctly.
        "embedding_dim": 384,

        # Internal projection — 272-dim was selected in V6/V7 papers as a
        # sweet spot between compression and expressivity. Users normally
        # do not change this.
        "internal_dim": 272,

        # Per-tier capacity. The drop-in version uses circular buffers so
        # these act as soft limits — store() always succeeds.
        "working_capacity": 64,
        "episodic_capacity": 1000,
        "semantic_prototypes": 256,
        "immunological_capacity": 100,

        # Router / forgetting parameters
        "kl_coefficient": 0.01,
        "anomaly_threshold": 2.0,
        "decay_rate": 0.95,
    },
    "router": {
        # Strategy for allocating attention across the 4 tiers.
        #   "kl_constrained" — softmax over a learned MLP, regularized
        #                       to a uniform prior (default, never collapses)
        #   "uniform"        — equal weights, no learning
        "type": "kl_constrained",
        "kl_coefficient": 0.01,
        "hidden_dim": 128,
    },
    "storage": {
        # "sqlite" persists to a single .db file you can open with any
        # SQLite browser. "memory" keeps everything in RAM and is faster
        # but loses data on exit.
        "type": "sqlite",
        "db_path": "mathir.db",

        # Auto-save after every store(). Disable for batch workloads.
        "auto_save": True,
    },
    "perception": {
        # Used by the simplified 4-tier router. The full V7 plugin exposes
        # 8 algorithms; the drop-in version uses one canonical pipeline.
        "use_residual": True,    # Add the original embedding back after fusion
        "use_layer_norm": True,  # Stabilize the fused output
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_default_config() -> Dict[str, Any]:
    """Return a deep copy of ``DEFAULT_CONFIG``.

    Always deep-copy: returning the live dict would let a caller mutate
    the global default and silently break other instances.
    """
    return copy.deepcopy(DEFAULT_CONFIG)


def configure(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a config dict by deep-merging ``overrides`` onto the defaults.

    Example::

        cfg = configure({
            "memory": {"embedding_dim": 768},
            "storage": {"db_path": "/var/data/agent.db"},
        })

    Nested dicts are merged recursively; non-dict values in ``overrides``
    replace the corresponding entries in the base. ``None`` returns a
    fresh default.
    """
    return merge_config(get_default_config(), overrides or {})


def merge_config(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """Recursive deep-merge used by :func:`configure` and the YAML loader.

    The result is a brand-new dict — neither input is mutated. The merge
    follows the ``base ← override`` direction: leaf values in
    ``override`` always win.
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(config: Dict[str, Any]) -> None:
    """Raise ``ValueError`` if any required field is missing or invalid.

    Called automatically by ``MATHIRMemory.__init__``; users can also
    call it directly to validate a config loaded from disk.
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict, got {type(config).__name__}")

    mem = config.get("memory", {})
    if mem.get("embedding_dim", 0) <= 0:
        raise ValueError("memory.embedding_dim must be a positive integer")
    if mem.get("internal_dim", 0) <= 0:
        raise ValueError("memory.internal_dim must be a positive integer")
    for cap in ("working_capacity", "episodic_capacity", "semantic_prototypes",
                "immunological_capacity"):
        if mem.get(cap, 0) <= 0:
            raise ValueError(f"memory.{cap} must be a positive integer")

    rtype = config.get("router", {}).get("type", "kl_constrained")
    if rtype not in ("kl_constrained", "uniform"):
        raise ValueError(
            f"router.type must be 'kl_constrained' or 'uniform', got {rtype!r}"
        )

    stype = config.get("storage", {}).get("type", "sqlite")
    if stype not in ("sqlite", "memory"):
        raise ValueError(
            f"storage.type must be 'sqlite' or 'memory', got {stype!r}"
        )


__all__ = [
    "DEFAULT_CONFIG",
    "get_default_config",
    "configure",
    "merge_config",
    "validate_config",
]

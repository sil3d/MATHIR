"""
MATHIR Drop-in — Custom exceptions.

All MATHIRDropin errors inherit from ``MATHIRError`` so users can catch
the whole family with a single ``except`` clause, while still being able
to handle specific failure modes (dimension mismatches, full memory,
storage issues) when they want fine-grained control.
"""

from __future__ import annotations

from typing import Optional


class MATHIRError(Exception):
    """Base class for every MATHIR-dropin error.

    Catching this catches all MATHIR-specific failures. The ``hint`` field
    is a human-readable suggestion printed by the default ``__str__`` so
    error messages are actionable, not just descriptive.
    """

    def __init__(self, message: str, hint: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.hint:
            return f"{self.message}\n  → Hint: {self.hint}"
        return self.message


class DimensionMismatchError(MATHIRError):
    """Raised when an embedding has a different dimension than configured.

    This is the most common user error: a 768-dim model fed into a 384-dim
    memory. The exception carries both expected and actual sizes for
    easier debugging.
    """

    def __init__(self, expected: int, got: int, where: str = "input") -> None:
        msg = f"Dimension mismatch in {where}: expected {expected}, got {got}"
        hint = (
            f"Create the memory with the correct dim: "
            f"MATHIRMemory(embedding_dim={got}, ...)"
        )
        super().__init__(msg, hint=hint)
        self.expected = expected
        self.got = got


class MemoryFullError(MATHIRError):
    """Raised when a tier is at capacity and cannot accept more memories.

    The drop-in version uses circular buffers, so a full tier overwrites
    the oldest entry silently. This exception is reserved for cases where
    the user has explicitly requested strict capacity enforcement.
    """

    def __init__(self, tier: str, capacity: int) -> None:
        super().__init__(
            f"Memory tier '{tier}' is full (capacity={capacity})",
            hint=(
                "Increase the tier capacity in config, call memory.forget() "
                "to prune, or set strict_capacity=False to allow overwrite."
            ),
        )
        self.tier = tier
        self.capacity = capacity


class StorageError(MATHIRError):
    """Raised for SQLite I/O, schema, or serialization problems.

    The underlying ``sqlite3`` exception (if any) is preserved as
    ``__cause__`` so ``raise StorageError(...) from sqle`` keeps the
    original traceback intact.
    """

    def __init__(self, message: str, original: Optional[BaseException] = None) -> None:
        super().__init__(
            message,
            hint=(
                "Check the db_path is writable, the SQLite file is not "
                "locked by another process, and the schema is up to date."
            ),
        )
        self.original = original


__all__ = [
    "MATHIRError",
    "DimensionMismatchError",
    "MemoryFullError",
    "StorageError",
]

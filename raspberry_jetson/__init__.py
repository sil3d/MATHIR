"""MATHIR Memory for Raspberry Pi / Jetson - CPU-only portable subset."""
__version__ = "8.3.0"

# Import the portable mcp package — re-export key APIs
try:
    from mathir_lib import __version__ as _mcp_version
    __mcp_version__ = _mcp_version
except ImportError:
    __mcp_version__ = "unknown"

# Jetson/Pi-specific config defaults — set BEFORE daemon imports embedder.
import os
os.environ.setdefault("MATHIR_EMBEDDING_MODEL", "Xenova/all-MiniLM-L6-v2-onnx")
os.environ.setdefault("MATHIR_EMBEDDING_DIM", "384")
os.environ.setdefault("MATHIR_DEVICE", "cpu")
os.environ.setdefault("MATHIR_USE_ONNX", "1")


def get_data_dir():
    """Pi/Jetson data dir override — prefer /var/lib/mathir if writable, else ~/.local/share/mathir."""
    import os
    from pathlib import Path
    system_wide = Path("/var/lib/mathir")
    try:
        system_wide.mkdir(parents=True, exist_ok=True)
        test = system_wide / ".write_test"
        test.write_text("ok")
        test.unlink()
        return system_wide
    except (PermissionError, OSError):
        return Path.home() / ".local" / "share" / "mathir"


__all__ = ["__version__", "__mcp_version__", "get_data_dir"]
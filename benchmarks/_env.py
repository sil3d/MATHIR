#!/usr/bin/env python3
"""Auto-load MATHIR benchmark environment from .env at the benchmarks/ root.

Usage:
    from _env import *          # imports all env vars as module-level
    # or
    import _env; _env.load()    # load manually

Put .env at the benchmarks/ root (sibling of _env.py). One file for all
benchmarks — no per-directory duplication.

If python-dotenv is installed, .env is parsed strictly (no shell expansion).
If not, we fall back to a minimal regex parser that handles KEY=VALUE lines.
"""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ENV_PATH = _HERE / ".env"
_ENV_EXAMPLE = _HERE / ".env.example"


def _parse_simple(text: str) -> dict:
    """Minimal KEY=VALUE parser (no shell expansion, no comments magic)."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes (single or double)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        # Inline comments after unquoted value
        if value and not value.startswith(('"', "'")):
            # Only strip trailing # if preceded by whitespace
            for sep in (" #", "\t#"):
                if sep in value:
                    value = value.split(sep, 1)[0].rstrip()
                    break
        out[key] = value
    return out


def load(env_path: Path = None, override: bool = False) -> dict:
    """Load env vars from .env into os.environ. Returns parsed dict.

    - override=True: replace existing os.environ values (use only for tests)
    - override=False (default): only set if not already in os.environ
      (lets users export MATHIR_API_KEY=... at the shell to override)
    """
    path = env_path or _ENV_PATH
    if not path.exists():
        return {}

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    try:
        from dotenv import dotenv_values
        parsed = {k: v for k, v in dotenv_values(path).items() if v is not None}
    except ImportError:
        parsed = _parse_simple(text)

    for key, value in parsed.items():
        if not override and os.environ.get(key):
            continue
        os.environ[key] = value
    return parsed


def _ensure_loaded() -> dict:
    """Lazy load — only parses the .env file once per process."""
    if not hasattr(_ensure_loaded, "_cache"):
        _ensure_loaded._cache = load()
    return _ensure_loaded._cache


# Auto-load on import (matches MATHIR's "env should just work" UX)
_ensure_loaded()


__all__ = ["load", "_ENV_PATH", "_ENV_EXAMPLE", "_ensure_loaded"]


if __name__ == "__main__":
    # Diagnostic: `python benchmarks/_env.py` shows which vars were loaded.
    if not _ENV_PATH.exists():
        print(f"[_env] no .env at {_ENV_PATH}")
        print(f"[_env] copy {_ENV_EXAMPLE.name} to .env and fill in your keys")
        sys.exit(0)
    parsed = load(override=False)
    if not parsed:
        print(f"[_env] .env found at {_ENV_PATH} but empty")
        sys.exit(0)
    print(f"[_env] loaded {len(parsed)} vars from {_ENV_PATH.name}:")
    for k, v in sorted(parsed.items()):
        # Mask secret-looking values
        if any(s in k.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD")):
            v = v[:4] + "***" if len(v) > 4 else "***"
        print(f"  {k}={v}")

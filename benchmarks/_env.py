# Auto-generated env loader — reads benchmarks/.env into os.environ
# This MUST be importable as `_env` from benchmarks/08_industry_validation/
import os
from pathlib import Path

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Don't override already-set shell env vars
                if key not in os.environ or not os.environ.get(key):
                    os.environ[key] = value

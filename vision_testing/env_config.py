"""MATHIR Playground — environment loader.

Loads .env from the package directory at import time so OpenRouter API keys,
MATHIR daemon config, etc. are picked up automatically.

Priority: real env vars > .env file > defaults.
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ENV_PATH = _HERE / ".env"


def _parse_env_line(line: str):
    """Parse a single KEY=VALUE line. Returns (key, value) or (None, None)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None, None
    if "=" not in line:
        return None, None
    key, _, value = line.partition("=")
    key = key.strip()
    value = value.strip()
    # Strip optional quotes
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return key, value


def load_env(env_path: Path = None) -> dict:
    """Load environment variables from .env file. Returns the parsed dict.

    Real env vars already set take precedence over .env (not overwritten).
    """
    path = env_path or _ENV_PATH
    loaded = {}
    if not path.exists():
        return loaded
    try:
        with open(path, encoding="utf-8") as f:
            for raw_line in f:
                key, value = _parse_env_line(raw_line)
                if key is None:
                    continue
                if key not in os.environ:
                    os.environ[key] = value
                loaded[key] = os.environ.get(key, value)
    except Exception as e:
        # .env load errors should never crash the app
        print(f"  [WARN] Failed to load {path}: {e}")
    return loaded


# Auto-load on import
load_env()


# ---------------------------------------------------------------------------
# Typed accessors with sensible defaults
# ---------------------------------------------------------------------------
def get_openrouter_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "")


def get_openrouter_api_base() -> str:
    return os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")


def get_opencode_zen_api_key() -> str:
    return os.environ.get("OPENCODE_ZEN_API_KEY", "")


def get_opencode_zen_api_base() -> str:
    return os.environ.get("OPENCODE_ZEN_API_BASE", "https://opencode.ai/zen/v1")


def get_openrouter_timeout() -> int:
    try:
        return int(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "120"))
    except ValueError:
        return 120


def get_openrouter_max_retries() -> int:
    try:
        return int(os.environ.get("OPENROUTER_MAX_RETRIES", "2"))
    except ValueError:
        return 2


def get_openrouter_default_model() -> str:
    return os.environ.get("OPENROUTER_DEFAULT_MODEL", "google/gemini-2.0-flash-exp:free")


def get_mathir_daemon_host() -> str:
    return os.environ.get("MATHIR_DAEMON_HOST", "127.0.0.1")


def get_mathir_daemon_port() -> int:
    try:
        return int(os.environ.get("MATHIR_DAEMON_PORT", "7338"))
    except ValueError:
        return 7338


def get_ui_host() -> str:
    return os.environ.get("UI_HOST", "127.0.0.1")


def get_ui_port() -> int:
    try:
        return int(os.environ.get("UI_PORT", "5000"))
    except ValueError:
        return 5000


def get_ui_debug() -> bool:
    return os.environ.get("UI_DEBUG", "false").lower() in ("1", "true", "yes")


def get_camera_config() -> dict:
    try:
        return {
            "device_id": int(os.environ.get("CAMERA_DEVICE_ID", "0")),
            "width": int(os.environ.get("CAMERA_WIDTH", "1280")),
            "height": int(os.environ.get("CAMERA_HEIGHT", "720")),
            "fps": int(os.environ.get("CAMERA_FPS", "30")),
        }
    except ValueError:
        return {"device_id": 0, "width": 1280, "height": 720, "fps": 30}


def get_audio_config() -> dict:
    try:
        return {
            "push_to_talk_key": os.environ.get("AUDIO_PUSH_TO_TALK_KEY", "Space"),
            "max_record_seconds": int(os.environ.get("AUDIO_MAX_RECORD_SECONDS", "30")),
        }
    except ValueError:
        return {"push_to_talk_key": "Space", "max_record_seconds": 30}


def get_ui_theme() -> str:
    return os.environ.get("UI_THEME", "auto")
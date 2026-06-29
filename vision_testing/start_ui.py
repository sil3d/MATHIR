#!/usr/bin/env python3
"""
Start the MATHIR Playground UI server.
Reads config from config.json + ui_config.json + .env (NO hardcoded paths).

Usage:
  python start_ui.py                 # use defaults
  python start_ui.py --port 5050    # override port
  python start_ui.py --host 0.0.0.0 # listen on all interfaces

.env loading: copied from .env.example. Edit the file with your API keys,
or set them as env vars (OPENROUTER_API_KEY=sk-or-v1-...).
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

REQUIREMENTS = ["flask", "flask-cors", "opencv-python", "requests"]


def check_and_install():
    missing = []
    for pkg in REQUIREMENTS:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing, check=True)


def load_env_file():
    """Load .env file from the package directory if present. Silent if missing."""
    env_path = HERE / ".env"
    if not env_path.exists():
        return
    try:
        from env_config import load_env
        loaded = load_env(env_path)
        if loaded:
            print(f"  Loaded {len(loaded)} env var(s) from .env")
    except Exception as e:
        print(f"  [WARN] .env load failed: {e}")


if __name__ == "__main__":
    check_and_install()
    load_env_file()

    print("=" * 60)
    print("MATHIR Playground  (v8.5.1)")
    print("=" * 60)
    print("Configuration sources:")
    print("  - .env (gitignored, your secrets)")
    print("  - config.json (models, openrouter section)")
    print("  - ui_config.json (UI, camera, audio)")
    print()
    print("Starting server...")
    print("Open the URL shown below in your browser.")
    print()

    # Run the server (foreground)
    try:
        subprocess.run([sys.executable, str(HERE / "ui_server.py")] + sys.argv[1:], check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
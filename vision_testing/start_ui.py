#!/usr/bin/env python3
"""
Start the MATHIR Vision Testing UI server.
Reads config from config.json and ui_config.json - NO hardcoded paths.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Install requirements if missing
REQUIREMENTS = ["flask", "flask-cors", "opencv-python"]

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


if __name__ == "__main__":
    check_and_install()
    print("=" * 60)
    print("MATHIR Vision Testing UI")
    print("=" * 60)
    print("Starting server...")
    print("Open the URL shown below in your browser.")
    print()
    # Run the server
    subprocess.run([sys.executable, str(HERE / "ui_server.py")] + sys.argv[1:])
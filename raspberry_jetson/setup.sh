#!/bin/bash
# setup.sh — Install MATHIR for Raspberry Pi / Jetson.
# Thin wrapper: installs the portable ``mcp/`` package and Pi/Jetson extras.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "MATHIR — Raspberry Pi / Jetson installer"
echo ""

# System prerequisites
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Optional: venv (recommended on Pi)
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "[1/3] Creating venv..."
    python3 -m venv "$SCRIPT_DIR/.venv"
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.venv/bin/activate"

# Install the portable mcp package (editable, so changes apply)
echo "[2/3] Installing mathir-mcp from ../mcp/..."
pip install --quiet -e "$PROJECT_ROOT/mcp"

# Pi/Jetson extras
echo "[3/3] Installing ARM-friendly extras..."
pip install --quiet \
    "onnxruntime>=1.16" \
    "numpy>=1.24"

echo ""
echo "Done. Start with:  ./start.sh"
echo "Test:  python3 -c 'import mathir_lib; print(mathir_lib.__version__)'"

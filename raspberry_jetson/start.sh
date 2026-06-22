#!/bin/bash
# start.sh — Start MATHIR on Raspberry Pi / Jetson.
# Thin wrapper: delegates to the portable ``mathir_mcp/`` package and forces
# CPU + ONNX-friendly defaults for resource-constrained boards.
#
# Usage: ./start.sh [config-override.json]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Pi/Jetson defaults — ONNX quantised, CPU only, smaller memory footprint.
export MATHIR_EMBEDDING_MODEL="${MATHIR_EMBEDDING_MODEL:-Xenova/all-MiniLM-L6-v2-onnx}"
export MATHIR_EMBEDDING_DIM="${MATHIR_EMBEDDING_DIM:-384}"
export MATHIR_DEVICE="${MATHIR_DEVICE:-cpu}"
export MATHIR_USE_ONNX="${MATHIR_USE_ONNX:-1}"
export MATHIR_PORT="${MATHIR_PORT:-7338}"

# Optional config override (first positional arg)
if [ -n "$1" ] && [ -f "$1" ]; then
    export MATHIR_CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
fi

echo "MATHIR — Raspberry Pi / Jetson launcher"
echo "  Model:    $MATHIR_EMBEDDING_MODEL"
echo "  Dim:      $MATHIR_EMBEDDING_DIM"
echo "  Device:   $MATHIR_DEVICE"
echo "  Port:     $MATHIR_PORT"
echo "  Config:   ${MATHIR_CONFIG:-(default: ~/.config/mathir/mathir.json)}"
echo ""

# Install Pi/Jetson extras if missing
python3 -c "import onnxruntime" 2>/dev/null || {
    echo "[setup] Installing onnxruntime for ARM..."
    pip3 install --quiet onnxruntime
}

# Delegate to the portable mathir_mcp package
exec python3 -m mathir_mcp

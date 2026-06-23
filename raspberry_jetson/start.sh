#!/bin/bash
# start.sh — Start MATHIR on Raspberry Pi / Jetson.
# Thin wrapper: delegates to the portable ``mathir_mcp/`` package and forces
# CPU + small-model defaults for resource-constrained boards.
#
# Usage: ./start.sh [config-override.json]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Pi/Jetson defaults — small model, CPU only, ~50MB memory footprint.
# Note: in v8.4.0 the daemon always uses sentence-transformers (no ONNX
# switch yet — see mathir_mcp/mathir_lib/mathir_mcp_server.py:get_embedder).
# We pick the smallest MiniLM variant to fit Pi 4 / Jetson Nano RAM.
export MATHIR_EMBEDDING_MODEL="${MATHIR_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
export MATHIR_EMBEDDING_DIM="${MATHIR_EMBEDDING_DIM:-384}"
export MATHIR_DEVICE="${MATHIR_DEVICE:-cpu}"
export MATHIR_PORT="${MATHIR_PORT:-7338}"

# Optional config override (first positional arg)
if [ -n "$1" ] && [ -f "$1" ]; then
    export MATHIR_CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
fi

echo "MATHIR — Raspberry Pi / Jetson launcher (v8.4.0)"
echo "  Model:    $MATHIR_EMBEDDING_MODEL"
echo "  Dim:      $MATHIR_EMBEDDING_DIM"
echo "  Device:   $MATHIR_DEVICE"
echo "  Port:     $MATHIR_PORT"
echo "  Config:   ${MATHIR_CONFIG:-(default: ~/.config/mathir/mathir.json)}"
echo ""

# Delegate to the portable mathir_mcp package
exec python3 -m mathir_mcp

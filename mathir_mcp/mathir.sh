#!/bin/bash
# MATHIR Playground — Stats Dashboard launcher (portable, no hardcoded paths)
#
# Usage: ./mathir.sh
# - Launches the stats server in background
# - Opens browser to http://127.0.0.1:7420
# - Works on Linux, macOS, and WSL
# - Script MUST be in the mathir_mcp/ root (it looks for mathir_lib/ subdir)

# Resolve the script's own directory (no hardcoded opencode path)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# The stats server lives in mathir_lib/ subdirectory (portable, same dir structure on Pi)
STATS_SERVER="$SCRIPT_DIR/mathir_lib/mathir_stats_server.py"

if [ ! -f "$STATS_SERVER" ]; then
    echo "ERROR: stats server not found at:"
    echo "  $STATS_SERVER"
    echo ""
    echo "This script must be in the mathir_mcp/ root directory."
    exit 1
fi

echo "=========================================="
echo "  MATHIR Playground — Stats Dashboard"
echo "=========================================="
echo ""
echo "  Script dir:  $SCRIPT_DIR"
echo "  Server file: $STATS_SERVER"
echo ""

# Launch the stats server in background
echo "Starting MATHIR Stats Dashboard..."
python "$STATS_SERVER" &
DASHBOARD_PID=$!
echo "  Dashboard PID: $DASHBOARD_PID"

# Wait 3s for the server to bind port 7420
sleep 3

# Open browser (cross-platform)
if command -v xdg-open &>/dev/null; then
    xdg-open http://127.0.0.1:7420 2>/dev/null
elif command -v open &>/dev/null; then
    open http://127.0.0.1:7420
elif command -v wslview &>/dev/null; then
    wslview http://127.0.0.1:7420
else
    echo ""
    echo "  Open http://127.0.0.1:7420 in your browser"
fi

echo ""
echo "Dashboard running at http://127.0.0.1:7420"
echo "  (PID=$DASHBOARD_PID — kill with: kill $DASHBOARD_PID)"
echo "  (or: pkill -f mathir_stats_server.py)"
echo ""

# Wait for the background process
wait $DASHBOARD_PID
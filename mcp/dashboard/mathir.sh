#!/bin/bash
# MATHIR Stats Dashboard launcher
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting MATHIR Stats Dashboard..."
python "$SCRIPT_DIR/mathir_stats_server.py" &
sleep 2

# Open browser (cross-platform)
if command -v xdg-open &>/dev/null; then
    xdg-open http://127.0.0.1:7420
elif command -v open &>/dev/null; then
    open http://127.0.0.1:7420
else
    echo "Open http://127.0.0.1:7420 in your browser"
fi

echo "Dashboard running at http://127.0.0.1:7420"
echo "Press Ctrl+C to stop."
wait

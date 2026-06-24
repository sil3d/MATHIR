#!/bin/bash
# MATHIR Smart Installer — Installs deps + auto-detects agents + injects config
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo ""
echo "  MATHIR Smart Installer - Install + Auto-Detect + Inject"
echo "  5-tier cognitive memory for 40+ coding agents"
echo ""
echo "================================================================"
echo ""

# STEP 1: Install Python dependencies
echo "[1/3] Installing Python dependencies..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQ_FILE="$SCRIPT_DIR/../requirements.txt"

if [ ! -f "$REQ_FILE" ]; then
    REQ_FILE="$SCRIPT_DIR/../../requirements.txt"
fi

if [ -f "$REQ_FILE" ]; then
    echo "  Using: $REQ_FILE"
    python3 -m pip install --upgrade pip --quiet 2>&1 | tail -3
    python3 -m pip install -r "$REQ_FILE" 2>&1 | tail -5
    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Failed to install dependencies."
        echo "Try manually: python3 -m pip install -r $REQ_FILE"
        read -p "Press Enter to continue..."
        exit 1
    fi
    echo "  Dependencies installed."
else
    echo "  [WARNING] requirements.txt not found. Install deps manually:"
    echo "    python3 -m pip install torch sentence-transformers usearch sqlite-vec numpy PyYAML"
    read -p "Press Enter to continue (or Ctrl+C to abort)..."
fi
echo ""

# STEP 2: Auto-detect coding agents
echo "[2/3] Detecting coding agents..."
echo ""

python3 install_smart.py
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Python3 not found or smart installer failed."
    echo "Install Python from https://python.org"
    echo ""
    read -p "Press Enter to continue..."
    exit 1
fi
echo ""

# STEP 3: Auto-start setup
echo "[3/3] Setting up auto-start..."
python3 install_smart.py --autostart-only 2>&1 | tail -5

echo ""
echo "================================================================"
echo "  MATHIR installed!"
echo "  - Daemon on port 7338"
echo "  - Stats server on port 7420 (auto-started)"
echo "  - Auto-start configured (launchd / systemd / Startup folder)"
echo "  - Dashboard: http://localhost:7420"
echo "================================================================"
echo ""
read -p "Press Enter to exit..."

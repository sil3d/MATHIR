#!/bin/bash
# MATHIR Smart Installer — Auto-detect coding agents, inject config + system prompt
cd "$(dirname "$0")"
python3 install_smart.py
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Python3 not found or script failed."
    echo "Install Python from https://python.org"
    echo ""
    read -p "Press Enter to continue..."
fi

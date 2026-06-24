#!/bin/bash
# MATHIR Smart Installer — Installs deps + auto-detects agents + injects config
# Supports: Linux (x86_64, aarch64), macOS (Intel, Apple Silicon), Raspberry Pi

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo ""
echo "  MATHIR Smart Installer - Install + Auto-Detect + Inject"
echo "  5-tier cognitive memory for 40+ coding agents"
echo ""
echo "================================================================"
echo ""

# Detect Python
PYTHON=""
for cmd in python3.11 python3.10 python3 python; do
    if command -v $cmd &>/dev/null; then
        VERSION=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON=$cmd
            echo "[OK] Found Python $VERSION at: $(command -v $cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python 3.10+ not found."
    echo "Install Python 3.10+ first:"
    echo "  - Ubuntu/Debian: sudo apt install python3.10 python3.10-venv python3-pip"
    echo "  - Fedora: sudo dnf install python3.10"
    echo "  - macOS: brew install python@3.10"
    echo "  - Raspberry Pi: sudo apt install python3.10"
    exit 1
fi

# Detect platform
ARCH=$(uname -m)
OS=$(uname -s)
echo "[INFO] Platform: $OS $ARCH"

IS_RPI=false
IS_APPLE_SILICON=false
HAS_CUDA=false

if [[ "$OS" == "Linux" && "$ARCH" == "aarch64" ]]; then
    # Could be Raspberry Pi or Apple Silicon under Linux (Asahi)
    if [ -f /proc/device-tree/model ] && grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
        IS_RPI=true
        echo "[INFO] Detected Raspberry Pi"
    fi
elif [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    IS_APPLE_SILICON=true
    echo "[INFO] Detected Apple Silicon (M1/M2/M3/M4)"
fi

# Detect NVIDIA GPU (for CUDA)
if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi &>/dev/null; then
        HAS_CUDA=true
        echo "[INFO] NVIDIA GPU detected (CUDA available)"
    fi
fi

# STEP 1: Install Python dependencies
echo ""
echo "[1/3] Installing Python dependencies..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQ_FILE="$SCRIPT_DIR/../requirements.txt"

if [ ! -f "$REQ_FILE" ]; then
    REQ_FILE="$SCRIPT_DIR/../../requirements.txt"
fi

if [ ! -f "$REQ_FILE" ]; then
    echo "[ERROR] requirements.txt not found at $REQ_FILE"
    exit 1
fi

echo "  Using: $REQ_FILE"
$PYTHON -m pip install --upgrade pip --quiet 2>&1 | tail -2

# Install PyTorch with appropriate backend
echo "  Installing PyTorch..."
if [ "$HAS_CUDA" = true ]; then
    echo "    Using CUDA 12.4 (NVIDIA GPU detected)"
    $PYTHON -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
elif [ "$IS_APPLE_SILICON" = true ]; then
    echo "    Using MPS (Apple Silicon)"
    $PYTHON -m pip install torch torchvision torchaudio --quiet
elif [ "$IS_RPI" = true ]; then
    echo "    Using CPU-only (Raspberry Pi, no GPU)"
    echo "    [NOTE] On RPi, embedding will be slow (~5s per recall with paraphrase-multilingual-MiniLM-L12-v2)"
    $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
else
    echo "    Using CPU-only (no NVIDIA GPU detected)"
    $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# Install remaining deps from requirements.txt
echo "  Installing remaining deps from requirements.txt..."
$PYTHON -m pip install -r "$REQ_FILE" --quiet

echo "  Dependencies installed."
echo ""

# STEP 2: Auto-detect coding agents
echo "[2/3] Detecting coding agents..."
echo ""

if ! $PYTHON install_smart.py; then
    echo ""
    echo "[ERROR] Smart installer failed."
    exit 1
fi
echo ""

# STEP 3: Auto-start setup
echo "[3/3] Setting up auto-start..."
$PYTHON install_smart.py --autostart-only 2>&1 | tail -5

echo ""
echo "================================================================"
echo "  MATHIR installed!"
echo "  - Python: $($PYTHON --version)"
echo "  - Platform: $OS $ARCH"
if [ "$HAS_CUDA" = true ]; then
    echo "  - PyTorch: CUDA (GPU)"
elif [ "$IS_APPLE_SILICON" = true ]; then
    echo "  - PyTorch: MPS (Apple GPU)"
else
    echo "  - PyTorch: CPU"
fi
echo "  - Daemon on port 7338"
echo "  - Stats server on port 7420 (auto-started)"
if [ "$IS_RPI" = true ]; then
    echo "  - [NOTE] RPi: no GPU, embedding is slow but works"
fi
echo "  - Dashboard: http://localhost:7420"
echo "================================================================"
echo ""

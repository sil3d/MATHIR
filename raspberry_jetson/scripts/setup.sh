#!/bin/bash
# setup.sh — Full MATHIR setup for Raspberry Pi / Jetson
# Usage: ./setup.sh [ollama|onnx|llamacpp] [cpu|gpu]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MATHIR_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

MODE="${1:-ollama}"
DEVICE="${2:-cpu}"

echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         MATHIR Setup — Raspberry Pi / Jetson             ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Provider: ${YELLOW}$MODE${NC} | Device: ${YELLOW}$DEVICE${NC}"
echo ""

# Step 1: System dependencies
echo -e "${CYAN}[1/5] System dependencies...${NC}"
sudo apt update
sudo apt install -y python3 python3-pip python3-venv curl wget git

# Step 2: Python dependencies
echo -e "${CYAN}[2/5] Python dependencies...${NC}"
pip3 install --upgrade pip
pip3 install -r "$MATHIR_DIR/requirements.txt"

# GPU-specific
if [ "$DEVICE" = "gpu" ]; then
    echo -e "${YELLOW}Installing CUDA support...${NC}"
    # Check if Jetson
    if [ -f "/etc/nv_tegra_release" ]; then
        echo -e "  Jetson detected — using pre-installed CUDA"
    else
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    fi
fi

# Step 3: Download model
echo -e "${CYAN}[3/5] Downloading model...${NC}"
"$SCRIPT_DIR/download_model.sh" "$MODE" medium

# Step 4: Configure
echo -e "${CYAN}[4/5] Configuring MATHIR...${NC}"
CONFIG_FILE="$MATHIR_DIR/config/mathir.json"

case $MODE in
    ollama)
        EMBED_DIM=768
        EMBED_MODEL="nomic-embed-text"
        if [ "$DEVICE" = "gpu" ]; then
            CHAT_MODEL="llama3.2:3b"
        else
            CHAT_MODEL="llama3.2:1b"
        fi
        ;;
    onnx)
        EMBED_DIM=1024
        EMBED_MODEL="octen-int8"
        CHAT_MODEL="none"
        ;;
    llamacpp)
        EMBED_DIM=384
        EMBED_MODEL="default"
        CHAT_MODEL="local"
        ;;
esac

cat > "$CONFIG_FILE" << EOF
{
  "model": "$MODE",
  "device": "$DEVICE",
  "embedding_dim": $EMBED_DIM,
  "port": 7338,
  "db_path": ".mathir/mathir.db",
  "memory": {
    "working_capacity": 32,
    "episodic_capacity": 256,
    "semantic_prototypes": 64,
    "immunological_capacity": 32
  },
  "provider": {
    "type": "$MODE",
    "url": "http://localhost:11434",
    "model": "$EMBED_MODEL"
  },
  "brain": {
    "enabled": false,
    "proxy_port": 8182,
    "llm_port": 8181
  }
}
EOF

echo -e "  ${GREEN}Config written to $CONFIG_FILE${NC}"

# Step 5: Test
echo -e "${CYAN}[5/5] Testing...${NC}"
echo -e "  ${YELLOW}Start MATHIR with: ./start.sh $MODE${NC}"
echo ""

# Summary
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo -e "  Start MATHIR:    ${CYAN}./start.sh $MODE${NC}"
echo -e "  Start dashboard: ${CYAN}python3 dashboard/dashboard_server.py${NC}"
echo -e "  Run benchmarks:  ${CYAN}python3 benchmarks/benchmark_all.py${NC}"
echo ""
echo -e "  Docs:            ${CYAN}cat docs/WHEN_TO_USE.md${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

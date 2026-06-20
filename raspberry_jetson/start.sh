#!/bin/bash
# start.sh — Start MATHIR on Raspberry Pi / Jetson
# Usage: ./start.sh [ollama|onnx|llamacpp] [cpu|gpu]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MATHIR_DIR="$SCRIPT_DIR"
CONFIG_DIR="$MATHIR_DIR/config"
LIB_DIR="$MATHIR_DIR/mathir_lib"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

MODE="${1:-ollama}"
DEVICE="${2:-cpu}"

echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         MATHIR — Raspberry Pi / Jetson                  ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Provider: ${YELLOW}$MODE${NC} | Device: ${YELLOW}$DEVICE${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 not found${NC}"
    exit 1
fi

# Check GPU
if [ "$DEVICE" = "gpu" ]; then
    echo -e "${YELLOW}[GPU] Checking CUDA...${NC}"
    python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || {
        echo -e "  ${RED}CUDA not available. Falling back to CPU.${NC}"
        DEVICE="cpu"
    }
fi

# Check provider
echo -e "${YELLOW}[1/3] Checking $MODE...${NC}"
case $MODE in
    ollama)
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "  ${GREEN}Ollama: running${NC}"
            curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; models=[m['name'] for m in json.loads(sys.stdin.read()).get('models',[])]; print(f'  Models: {models}')" 2>/dev/null
        else
            echo -e "  ${RED}Ollama: not running${NC}"
            echo "  Start: ollama serve"
            exit 1
        fi
        ;;
    llamacpp)
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            echo -e "  ${GREEN}llama.cpp: running${NC}"
        else
            echo -e "  ${RED}llama.cpp: not running${NC}"
            echo "  Start: ./llama-server -m model.gguf --port 8080"
            exit 1
        fi
        ;;
    onnx)
        python3 -c "import onnxruntime; print(f'  ONNX Runtime: {onnxruntime.__version__}')" 2>/dev/null || {
            echo -e "  ${RED}onnxruntime not installed${NC}"
            echo "  Install: pip3 install onnxruntime"
            exit 1
        }
        ;;
esac

# Write config
echo -e "${YELLOW}[2/3] Writing config...${NC}"
case $MODE in
    ollama)
        if [ "$DEVICE" = "gpu" ]; then
            EMBED_MODEL="mxbai-embed-large"
            EMBED_DIM=1024
        else
            EMBED_MODEL="nomic-embed-text"
            EMBED_DIM=768
        fi
        cat > "$CONFIG_DIR/mathir.json" << EOF
{
  "model": "ollama",
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
    "type": "ollama",
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
        ;;
    llamacpp)
        cat > "$CONFIG_DIR/mathir.json" << EOF
{
  "model": "llamacpp",
  "device": "$DEVICE",
  "embedding_dim": 384,
  "port": 7338,
  "db_path": ".mathir/mathir.db",
  "memory": {
    "working_capacity": 32,
    "episodic_capacity": 256,
    "semantic_prototypes": 64,
    "immunological_capacity": 32
  },
  "provider": {
    "type": "llamacpp",
    "url": "http://localhost:8080",
    "model": "default"
  },
  "brain": {
    "enabled": false,
    "proxy_port": 8182,
    "llm_port": 8181
  }
}
EOF
        ;;
    onnx)
        cat > "$CONFIG_DIR/mathir.json" << EOF
{
  "model": "onnx",
  "device": "cpu",
  "embedding_dim": 1024,
  "port": 7338,
  "db_path": ".mathir/mathir.db",
  "memory": {
    "working_capacity": 32,
    "episodic_capacity": 256,
    "semantic_prototypes": 64,
    "immunological_capacity": 32
  },
  "provider": {
    "type": "onnx",
    "model_dir": "./models/octen-int8",
    "provider": "CPUExecutionProvider"
  },
  "brain": {
    "enabled": false,
    "proxy_port": 8182,
    "llm_port": 8181
  }
}
EOF
        ;;
esac

echo -e "  ${GREEN}Config written${NC}"

# Start daemon
echo -e "${YELLOW}[3/3] Starting MATHIR daemon...${NC}"
echo -e "${GREEN}Port: 7338 | Device: $DEVICE${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

python3 "$LIB_DIR/mathir_daemon.py" --config "$CONFIG_DIR/mathir.json"

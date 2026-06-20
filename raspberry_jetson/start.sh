#!/bin/bash
# MATHIR — Raspberry Pi / Jetson Start Script
# Usage: ./start.sh [ollama|llamacpp|onnx]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MATHIR_DIR="$SCRIPT_DIR"
CONFIG_DIR="$MATHIR_DIR/config"
LIB_DIR="$MATHIR_DIR/mathir_lib"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         MATHIR — Raspberry Pi / Jetson                  ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 not found${NC}"
    echo "Install: sudo apt install python3 python3-pip"
    exit 1
fi

# Check dependencies
echo -e "${YELLOW}[1/3] Checking dependencies...${NC}"
python3 -c "import sqlite3; print('  sqlite3: OK')" 2>/dev/null || echo -e "  ${RED}sqlite3: MISSING${NC}"
python3 -c "import numpy; print('  numpy: OK')" 2>/dev/null || echo -e "  ${RED}numpy: MISSING${NC}"

# Check for sqlite-vec
python3 -c "import sqlite_vec; print('  sqlite-vec: OK')" 2>/dev/null || echo -e "  ${YELLOW}sqlite-vec: not installed (optional)${NC}"

# Check provider
MODE="${1:-ollama}"
echo -e "${YELLOW}[2/3] Provider: $MODE${NC}"

case $MODE in
    ollama)
        # Check Ollama
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "  ${GREEN}Ollama: running${NC}"
            # List models
            MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; print('  Models: ' + ', '.join([m['name'] for m in json.loads(sys.stdin.read()).get('models', [])]))" 2>/dev/null)
            echo "$MODELS"
        else
            echo -e "  ${RED}Ollama: not running${NC}"
            echo "  Start with: ollama serve"
            echo "  Then pull a model: ollama pull nomic-embed-text"
            exit 1
        fi
        ;;
    llamacpp)
        # Check llama.cpp
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            echo -e "  ${GREEN}llama.cpp: running${NC}"
        else
            echo -e "  ${RED}llama.cpp: not running${NC}"
            echo "  Start with: ./llama-server -m model.gguf --port 8080"
            exit 1
        fi
        ;;
    onnx)
        echo -e "  ${GREEN}ONNX: local inference${NC}"
        # Check onnxruntime
        python3 -c "import onnxruntime; print('  onnxruntime: OK')" 2>/dev/null || {
            echo -e "  ${RED}onnxruntime: not installed${NC}"
            echo "  Install: pip3 install onnxruntime"
            exit 1
        }
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: ./start.sh [ollama|llamacpp|onnx]"
        exit 1
        ;;
esac

# Update config for selected provider
echo -e "${YELLOW}[3/3] Configuring MATHIR...${NC}"
case $MODE in
    ollama)
        cat > "$CONFIG_DIR/mathir.json" << 'EOF'
{
  "model": "ollama",
  "device": "cpu",
  "embedding_dim": 768,
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
    "model": "nomic-embed-text"
  }
}
EOF
        ;;
    llamacpp)
        cat > "$CONFIG_DIR/mathir.json" << 'EOF'
{
  "model": "llamacpp",
  "device": "cpu",
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
  }
}
EOF
        ;;
    onnx)
        cat > "$CONFIG_DIR/mathir.json" << 'EOF'
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
  }
}
EOF
        ;;
esac

echo -e "  ${GREEN}Config written to $CONFIG_DIR/mathir.json${NC}"

# Start daemon
echo ""
echo -e "${GREEN}Starting MATHIR daemon on port 7338...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

python3 "$LIB_DIR/mathir_daemon.py" --config "$CONFIG_DIR/mathir.json"

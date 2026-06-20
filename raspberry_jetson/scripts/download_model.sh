#!/bin/bash
# download_model.sh — Auto-download models for MATHIR on Raspberry Pi / Jetson
# Usage: ./download_model.sh [ollama|onnx|llamacpp] [small|medium|large]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

MODE="${1:-ollama}"
SIZE="${2:-medium}"

echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         MATHIR Model Downloader                         ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Mode: ${YELLOW}$MODE${NC} | Size: ${YELLOW}$SIZE${NC}"
echo ""

case $MODE in
    ollama)
        # Check Ollama installed
        if ! command -v ollama &> /dev/null; then
            echo -e "${RED}Ollama not found. Installing...${NC}"
            curl -fsSL https://ollama.ai/install.sh | sh
        fi

        echo -e "${GREEN}[1/2] Downloading embedding model...${NC}"
        case $SIZE in
            small)
                echo -e "  -> nomic-embed-text (768d, ~274MB)"
                ollama pull nomic-embed-text
                ;;
            medium)
                echo -e "  -> nomic-embed-text (768d, ~274MB)"
                ollama pull nomic-embed-text
                ;;
            large)
                echo -e "  -> mxbai-embed-large (1024d, ~670MB)"
                ollama pull mxbai-embed-large
                ;;
        esac

        echo -e "${GREEN}[2/2] Downloading chat model (for brain proxy)...${NC}"
        case $SIZE in
            small)
                echo -e "  -> llama3.2:1b (1B, ~1.3GB)"
                ollama pull llama3.2:1b
                ;;
            medium)
                echo -e "  -> llama3.2:3b (3B, ~2.0GB)"
                ollama pull llama3.2:3b
                ;;
            large)
                echo -e "  -> llama3.1:8b (8B, ~4.7GB)"
                ollama pull llama3.1:8b
                ;;
        esac

        echo -e "${GREEN}Models downloaded:${NC}"
        ollama list
        ;;

    onnx)
        echo -e "${GREEN}[1/1] Downloading ONNX model...${NC}"
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        MODEL_DIR="$SCRIPT_DIR/../models/octen-int8"

        if [ -d "$MODEL_DIR" ]; then
            echo -e "  ${YELLOW}Model already exists at $MODEL_DIR${NC}"
        else
            echo -e "  -> Octen-Embedding-0.6B-ONNX-INT8 (1024d, ~600MB)"
            mkdir -p "$MODEL_DIR"

            # Download from HuggingFace
            if command -v huggingface-cli &> /dev/null; then
                huggingface-cli download OctenAI/Octen-Embedding-0.6B-ONNX-INT8 \
                    --local-dir "$MODEL_DIR"
            else
                echo -e "  ${YELLOW}Installing huggingface_hub...${NC}"
                pip3 install huggingface_hub

                python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'OctenAI/Octen-Embedding-0.6B-ONNX-INT8',
    local_dir='$MODEL_DIR'
)
print('Downloaded to $MODEL_DIR')
"
            fi
        fi
        ;;

    llamacpp)
        echo -e "${GREEN}[1/1] llama.cpp model download info${NC}"
        echo ""
        echo -e "  ${YELLOW}For llama.cpp, download GGUF models manually:${NC}"
        echo ""
        echo "  Small (1B):"
        echo "    wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        echo ""
        echo "  Medium (3B):"
        echo "    wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        echo ""
        echo "  Large (8B):"
        echo "    wget https://huggingface.co/bartowski/Llama-3.1-8B-Instruct-GGUF/resolve/main/Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        echo ""
        echo "  Then start: ./llama-server -m <model>.gguf --port 8080"
        ;;

    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: ./download_model.sh [ollama|onnx|llamacpp] [small|medium|large]"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done! Next: ./start.sh $MODE${NC}"

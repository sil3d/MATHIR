#!/bin/bash
# run_all.sh — Run all MATHIR benchmarks on Raspberry Pi / Jetson
# Usage: ./run_all.sh [cpu|gpu]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${1:-cpu}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         MATHIR Benchmarks — Raspberry Pi / Jetson       ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Device: ${YELLOW}$DEVICE${NC}"
echo ""

# Check MATHIR daemon is running
echo -e "${YELLOW}[0/5] Checking MATHIR daemon...${NC}"
if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',7338)); s.close()" 2>/dev/null; then
    echo -e "  ${GREEN}Daemon: running on port 7338${NC}"
else
    echo -e "  ${RED}Daemon: NOT running${NC}"
    echo "  Start with: cd .. && ./start.sh ollama $DEVICE"
    exit 1
fi

# Check provider
echo -e "${YELLOW}[0/5] Checking provider...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "  ${GREEN}Ollama: running${NC}"
    PROVIDER="ollama"
elif curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}llama.cpp: running${NC}"
    PROVIDER="llamacpp"
else
    echo -e "  ${YELLOW}No LLM server detected — using ONNX if available${NC}"
    PROVIDER="onnx"
fi

# Run benchmarks
echo ""
echo -e "${CYAN}[1/5] 01_cross_llm_benchmark...${NC}"
cd "$SCRIPT_DIR/01_cross_llm_benchmark"
python3 benchmark.py --device "$DEVICE" --provider "$PROVIDER" 2>&1 | tee "$SCRIPT_DIR/06_results/01_cross_llm_results.txt" || echo -e "  ${RED}Skipped${NC}"

echo ""
echo -e "${CYAN}[2/5] 02_memory_risks...${NC}"
cd "$SCRIPT_DIR/02_memory_risks"
python3 memory_risks.py 2>&1 | tee "$SCRIPT_DIR/06_results/02_memory_risks_results.txt" || echo -e "  ${RED}Skipped${NC}"

echo ""
echo -e "${CYAN}[3/5] 03_vector_search_benchmarks...${NC}"
cd "$SCRIPT_DIR/03_vector_search_benchmarks"
for f in test_*.py; do
    echo -e "  Running: $f"
    python3 "$f" 2>&1 | tee "$SCRIPT_DIR/06_results/$(basename "$f" .py)_results.txt" || echo -e "  ${RED}Skipped: $f${NC}"
done

echo ""
echo -e "${CYAN}[4/5] 04_provider_benchmarks...${NC}"
cd "$SCRIPT_DIR/04_provider_benchmarks"
python3 ollama_one_by_one.py 2>&1 | tee "$SCRIPT_DIR/06_results/04_ollama_results.txt" || echo -e "  ${RED}Skipped${NC}"

echo ""
echo -e "${CYAN}[5/5] Provider benchmark...${NC}"
cd "$SCRIPT_DIR"
python3 -c "
import json, time, urllib.request

def benchmark_provider(url='http://localhost:11434', model='nomic-embed-text', n=10):
    times = []
    for i in range(n):
        payload = json.dumps({'model': model, 'prompt': f'test sentence {i}'}).encode()
        req = urllib.request.Request(f'{url}/api/embeddings', data=payload, headers={'Content-Type': 'application/json'})
        start = time.time()
        urllib.request.urlopen(req, timeout=30)
        times.append((time.time() - start) * 1000)
    avg = sum(times) / len(times)
    return {'avg_ms': round(avg, 1), 'min_ms': round(min(times), 1), 'max_ms': round(max(times), 1), 'n': n}

print('=== Provider Benchmark ===')
result = benchmark_provider()
print(json.dumps(result, indent=2))
with open('06_results/provider_benchmark.json', 'w') as f:
    json.dump(result, f, indent=2)
" 2>&1 | tee "$SCRIPT_DIR/06_results/provider_benchmark_results.txt" || echo -e "  ${RED}Skipped${NC}"

# Summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Benchmarks complete!${NC}"
echo ""
echo -e "Results saved to: ${CYAN}$SCRIPT_DIR/06_results/${NC}"
echo ""
echo -e "Compare with PC results:"
echo -e "  ${CYAN}cat $SCRIPT_DIR/06_results/current/README.md${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

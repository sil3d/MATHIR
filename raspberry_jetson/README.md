# MATHIR — Raspberry Pi / Jetson

**Self-contained MATHIR package for edge devices. Plug and play.**

---

## What's Inside

This is a COMPLETE MATHIR installation, ready to deploy:

```
raspberry_jetson/
├── mathir_lib/              ← Real MATHIR daemon + client + MCP
│   ├── mathir_daemon.py     ← Persistent daemon (keeps model loaded)
│   ├── mathir_client.py     ← CLI client (save/recall/search)
│   ├── mathir_mcp_server.py ← MCP server (9 tools)
│   ├── mathir_vec.py        ← VecMemory (SQLite + vectors)
│   ├── mathir_search.py     ← HybridSearch (vector + BM25)
│   ├── mathir_push.py       ← Proactive memory delivery
│   ├── memory_risks.py      ← Risk mitigation
│   └── mathir_onnx_embedder.py ← ONNX INT8 embedder
├── providers/               ← LLM providers
│   ├── base.py              ← Abstract provider
│   ├── ollama.py            ← Ollama API
│   ├── onnx.py              ← ONNX Runtime
│   └── onnx_embedder.py     ← Octen INT8 embedder
├── config/
│   ├── mathir.json          ← Pre-configured for edge
│   └── edge.yaml            ← Edge device config
├── start.sh                 ← Start script (ollama/llamacpp/onnx)
├── requirements.txt         ← Python dependencies
└── README.md                ← This file
```

---

## Quick Start (3 Commands)

```bash
# 1. Copy to your device
scp -r raspberry_jetson/ pi@raspberrypi:~/mathir

# 2. SSH into device
ssh pi@raspberrypi

# 3. Start
cd ~/mathir
chmod +x start.sh
./start.sh ollama
```

---

## Setup Ollama (Recommended)

```bash
# On the device
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull nomic-embed-text
ollama serve &

# Start MATHIR
./start.sh ollama
```

---

## Setup llama.cpp

```bash
# On the device
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build -DLLAMA_NATIVE=ON && cmake --build build -j4

# Download model (GGUF format)
# ...

# Start server
./build/bin/llama-server -m model.gguf --port 8080 &

# Start MATHIR
./start.sh llamacpp
```

---

## Test

```bash
# Ping daemon
python3 mathir_lib/mathir_client.py ping

# Save a memory
python3 mathir_lib/mathir_client.py save "Hello from Raspberry Pi" -a test -t semantic -l test

# Recall
python3 mathir_lib/mathir_client.py recall "Hello" -k 1

# Stats
python3 mathir_lib/mathir_client.py stats
```

---

## Config

Edit `config/mathir.json` to change provider:

```json
{
  "model": "ollama",
  "device": "cpu",
  "embedding_dim": 768,
  "port": 7338,
  "provider": {
    "type": "ollama",
    "url": "http://localhost:11434",
    "model": "nomic-embed-text"
  }
}
```

---

## Model Recommendations

| Device | Provider | Model | Dims | Speed |
|--------|----------|-------|------|-------|
| Raspberry Pi 4 (2GB) | Ollama | llama3.2:1b | 768 | ~200ms |
| Raspberry Pi 4 (4GB) | Ollama | nomic-embed-text | 768 | ~100ms |
| Raspberry Pi 5 | Ollama | nomic-embed-text | 768 | ~50ms |
| Jetson Nano | llama.cpp | llama3.2:3b | 768 | ~80ms |
| Jetson Orin | Ollama | nomic-embed-text | 768 | ~20ms |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Connection refused" | Start Ollama/llama.cpp first |
| "Model not found" | Check `ollama list` |
| "CUDA out of memory" | Use `--device cpu` |
| "Slow" | Expected on CPU, use smaller model |

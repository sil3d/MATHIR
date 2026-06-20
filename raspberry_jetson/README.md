# MATHIR — Raspberry Pi / Jetson

**Plug-and-play MATHIR package for edge devices. Connect your local LLM.**

---

## What This Is

This is a **complete, self-contained MATHIR installation** ready to deploy to Raspberry Pi or Jetson. Everything is copied from the real MATHIR project — no generic code, no placeholders.

```
raspberry_jetson/
├── mathir_lib/              ← Real MATHIR daemon + client + MCP (9 tools)
├── providers/               ← Real LLM providers (Ollama, ONNX)
├── brain/                   ← Brain architecture (auto-inject memories into LLM)
├── dashboard/               ← Real-time monitoring dashboard
├── benchmarks/              ← Performance benchmarks for your device
├── config/                  ← Pre-configured for edge
├── scripts/                 ← Auto-download + setup
├── docs/                    ← GPU vs CPU, when to use what
├── start.sh                 ← ./start.sh ollama|onnx|llamacpp [cpu|gpu]
└── requirements.txt         ← Python dependencies
```

---

## Quick Start (3 Commands)

```bash
# 1. Copy to device
scp -r raspberry_jetson/ pi@raspberrypi:~/mathir

# 2. Setup (installs everything)
ssh pi@raspberrypi
cd ~/mathir
chmod +x scripts/*.sh start.sh
./scripts/setup.sh ollama cpu

# 3. Start
./start.sh ollama cpu
```

---

## What Connects to What

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Device (RPi / Jetson)                │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │  LLM Server  │    │  MATHIR      │    │  Dashboard     │ │
│  │              │    │              │    │                │ │
│  │  Ollama      │───▶│  daemon      │───▶│  :7420         │ │
│  │  llama.cpp   │    │  :7338       │    │  (browser)     │ │
│  │  ONNX        │    │              │    │                │ │
│  └──────────────┘    └──────────────┘    └────────────────┘ │
│         │                   │                    │           │
│    Port 11434          Port 7338            Auto-created     │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │  Brain Proxy │    │  Benchmarks  │    │  Database      │ │
│  │              │    │              │    │                │ │
│  │  :8182       │    │  CPU/GPU     │    │  .mathir/      │ │
│  │  (optional)  │    │  testing     │    │  mathir.db     │ │
│  └──────────────┘    └──────────────┘    └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Provider Options

| Provider | Setup | GPU | Best For |
|----------|-------|-----|----------|
| **Ollama** | `ollama pull` | Auto | Simple, embedding + chat |
| **llama.cpp** | Build from source | Manual | Max GPU performance |
| **ONNX** | `pip install` | CPU only | Minimal RAM/CPU |

**See `docs/WHEN_TO_USE.md` for detailed comparison.**

---

## GPU vs CPU

| Device | CPU Speed | GPU Speed | Use GPU? |
|--------|-----------|-----------|----------|
| Raspberry Pi 4 | ~100ms | N/A | No GPU |
| Raspberry Pi 5 | ~50ms | N/A | No GPU |
| Jetson Nano | ~80ms | ~30ms | YES |
| Jetson Orin | ~40ms | ~10ms | YES |

**See `docs/GPU_VS_CPU.md` for optimization guide.**

---

## Auto-Download Models

```bash
# Ollama (recommended)
./scripts/download_model.sh ollama small   # ~274MB embedding
./scripts/download_model.sh ollama medium  # ~274MB embedding + 1.3GB chat
./scripts/download_model.sh ollama large   # ~670MB embedding + 4.7GB chat

# ONNX
./scripts/download_model.sh onnx medium   # ~600MB INT8

# llama.cpp
./scripts/download_model.sh llamacpp medium  # Manual GGUF download
```

---

## Dashboard

```bash
# Start dashboard
python3 dashboard/dashboard_server.py

# Open in browser
# http://<device-ip>:7420
```

Shows: 4-tier memory, per-agent stats, timeline, search.

---

## Benchmarks

```bash
# Run all benchmarks
python3 benchmarks/benchmark_edge.py --provider all --device cpu

# Specific provider
python3 benchmarks/benchmark_edge.py --provider ollama --device gpu

# Save results
python3 benchmarks/benchmark_edge.py --output results.json
```

Measures: cold start, warm embedding, batch, recall, memory usage.

---

## Brain Proxy (Optional)

Connect MATHIR to your LLM so memories are auto-injected into every call:

```bash
# Enable in config/mathir.json
"brain": {
  "enabled": true,
  "proxy_port": 8182,
  "llm_port": 8181
}

# Start brain stack
python3 brain/mathir_brain.py start

# Point your LLM client to port 8182 instead of 8181
```

---

## Documentation

| File | Description |
|------|-------------|
| `docs/WHEN_TO_USE.md` | ONNX vs Ollama vs llama.cpp |
| `docs/GPU_VS_CPU.md` | Performance guide |
| `benchmarks/benchmark_edge.py` | Device benchmarks |
| `scripts/setup.sh` | Full setup script |
| `scripts/download_model.sh` | Auto-download models |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Ollama not running" | `ollama serve` |
| "Model not found" | `ollama pull nomic-embed-text` |
| "CUDA not available" | `pip3 install torch torchvision --index-url ...` |
| "Slow on CPU" | Use ONNX: `pip3 install onnxruntime` |
| "Out of memory" | Use smaller model: `ollama pull llama3.2:1b` |

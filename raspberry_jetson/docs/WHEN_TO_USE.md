# When to Use What — ONNX vs Ollama vs llama.cpp

**Choose the right provider for your device.**

---

## Decision Tree

```
Do you have a GPU?
├── YES (Jetson / Desktop GPU)
│   ├── Want simplicity? → Ollama
│   ├── Want max performance? → llama.cpp
│   └── Want Python integration? → ONNX
│
└── NO (Raspberry Pi / CPU only)
    ├── Want simplicity? → Ollama
    ├── Want smallest footprint? → ONNX INT8
    └── Want GGUF models? → llama.cpp
```

---

## Provider Comparison

| Feature | Ollama | llama.cpp | ONNX |
|---------|--------|-----------|------|
| **Setup** | `curl \| sh` + `ollama pull` | Build from source | `pip install onnxruntime` |
| **GPU support** | Auto-detect | Manual (CUDA flags) | CPU only (unless DML) |
| **RAM usage** | ~500MB base | ~200MB base | ~100MB base |
| **Model format** | Ollama format | GGUF | ONNX |
| **Embedding endpoint** | `/api/embeddings` | `/v1/embeddings` | Local inference |
| **Chat endpoint** | `/api/chat` | `/v1/chat/completions** | N/A |
| **Auto-download** | `ollama pull` | Manual wget | `huggingface-cli` |
| **Best for** | Simple setup | Max GPU performance | Minimal RAM/CPU |

---

## ONNX — When to Use

### Use ONNX when:

- **No GPU available** (Raspberry Pi, old hardware)
- **Minimal RAM** (ONNX INT8 uses ~100MB vs ~500MB for Ollama)
- **Docker/edge deployment** (smaller container size)
- **You need deterministic inference** (same output every time)

### Don't use ONNX when:

- You want chat/LLM capabilities (ONNX is embedding-only)
- You need GPU acceleration (ONNX CPU is slower)
- You want easy model switching (ONNX requires re-export)

### Setup:

```bash
# Download model
./scripts/download_model.sh onnx medium

# Start
./start.sh onnx cpu
```

### Model: Octen-Embedding-0.6B-ONNX-INT8

| Property | Value |
|----------|-------|
| Dimensions | 1024 |
| Size | ~600MB |
| Speed (RPi 4) | ~150ms |
| Speed (Jetson) | ~30ms |
| Quality | ★★★★☆ |
| RAM | ~100MB |

---

## Ollama — When to Use

### Use Ollama when:

- **You want simplicity** (one command setup)
- **You need both embedding + chat** (brain proxy needs chat model)
- **You have decent RAM** (2GB+ for embedding model)
- **You want auto model management** (pull, list, run)

### Don't use Ollama when:

- RAM is very limited (<1GB)
- You need maximum GPU performance
- You're in a restricted environment (no internet for download)

### Setup:

```bash
# Install + download
./scripts/download_model.sh ollama medium

# Start
./start.sh ollama cpu
```

### Models:

| Model | Dims | Size | Speed (RPi 4) | Speed (Jetson) | RAM |
|-------|------|------|---------------|----------------|-----|
| nomic-embed-text | 768 | ~274MB | ~100ms | ~20ms | ~300MB |
| mxbai-embed-large | 1024 | ~670MB | ~200ms | ~40ms | ~600MB |
| llama3.2:1b (chat) | — | ~1.3GB | ~500ms | ~100ms | ~1.5GB |
| llama3.2:3b (chat) | — | ~2.0GB | ~1s | ~200ms | ~2.5GB |

---

## llama.cpp — When to Use

### Use llama.cpp when:

- **You want maximum GPU performance** (TensorRT, CUDA)
- **You have GGUF models** (quantized, smaller)
- **You want fine-grained control** (context size, layers, threads)
- **You're on Jetson** (CUDA acceleration)

### Don't use llama.cpp when:

- You want simplicity (build from source required)
- You need easy setup (no `pip install` equivalent)
- You're on Raspberry Pi (CPU-only llama.cpp is slow)

### Setup:

```bash
# Build (Raspberry Pi)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_NATIVE=ON
cmake --build build -j4

# Build (Jetson)
cmake -B build -DLLAMA_CUDA=ON
cmake --build build -j$(nproc)

# Download model
./scripts/download_model.sh llamacpp medium

# Start server
./build/bin/llama-server -m models/llama-3.2-3b.gguf --port 8080

# Start MATHIR
./start.sh llamacpp gpu
```

### Performance:

| Device | Model | Speed | Notes |
|--------|-------|-------|-------|
| RPi 4 (CPU) | llama3.2:1b Q4 | ~500ms | Slow, but works |
| RPi 5 (CPU) | llama3.2:3b Q4 | ~200ms | Better |
| Jetson Nano | llama3.2:3b Q4 CUDA | ~80ms | GPU accelerated |
| Jetson Orin | llama3.1:8b Q4 CUDA | ~30ms | Fast |

---

## Brain Proxy — Connecting to LLM

The brain architecture needs a CHAT model (not just embedding) to inject memories into LLM calls.

### With Ollama:

```json
{
  "brain": {
    "enabled": true,
    "proxy_port": 8182,
    "llm_port": 8181,
    "provider": "ollama",
    "chat_model": "llama3.2:3b"
  }
}
```

### With llama.cpp:

```json
{
  "brain": {
    "enabled": true,
    "proxy_port": 8182,
    "llm_port": 8181,
    "provider": "llamacpp",
    "chat_model": "local"
  }
}
```

### With ONNX:

ONNX is embedding-only. For brain proxy, you need a separate chat model (Ollama or llama.cpp).

---

## Recommendations by Device

### Raspberry Pi 4 (2GB RAM)

```bash
# Best: Ollama + nomic-embed-text
./scripts/download_model.sh ollama small
./start.sh ollama cpu
```

### Raspberry Pi 4 (4GB RAM)

```bash
# Better: Ollama + nomic-embed-text + llama3.2:1b (for brain)
./scripts/download_model.sh ollama medium
./start.sh ollama cpu
```

### Raspberry Pi 5 (8GB RAM)

```bash
# Great: Ollama + mxbai-embed-large + llama3.2:3b
./scripts/download_model.sh ollama large
./start.sh ollama cpu
```

### Jetson Nano (4GB)

```bash
# Good: llama.cpp + CUDA
./scripts/download_model.sh llamacpp medium
./start.sh llamacpp gpu
```

### Jetson Orin Nano (8GB)

```bash
# Great: Ollama + GPU
./scripts/download_model.sh ollama large
./start.sh ollama gpu
```

### Desktop (NVIDIA GPU)

```bash
# Best: Ollama + GPU
./scripts/download_model.sh ollama large
./start.sh ollama gpu
```

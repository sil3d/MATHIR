# GPU vs CPU — Performance Guide

**When to use GPU, when to use CPU, and how to benchmark.**

---

## Quick Answer

| Device | CPU Speed | GPU Speed | Use GPU? |
|--------|-----------|-----------|----------|
| Raspberry Pi 4 | ~100ms | N/A | No GPU |
| Raspberry Pi 5 | ~50ms | N/A | No GPU |
| Jetson Nano | ~80ms | ~30ms | YES |
| Jetson Orin Nano | ~40ms | ~10ms | YES |
| Desktop (RTX) | ~50ms | ~3ms | YES |

**Rule of thumb:** If GPU is available and you do >100 embeddings/day, use GPU.

---

## CPU Mode

### When to use CPU:

- Raspberry Pi (no GPU)
- Docker containers (no GPU passthrough)
- Low-power devices
- Development/testing

### How to start:

```bash
./start.sh ollama cpu
```

### Performance:

| Device | Model | Embeddings/sec | Latency |
|--------|-------|----------------|---------|
| RPi 4 (2GB) | nomic-embed-text | ~10 | ~100ms |
| RPi 4 (4GB) | nomic-embed-text | ~10 | ~100ms |
| RPi 5 (8GB) | nomic-embed-text | ~20 | ~50ms |
| Desktop CPU | nomic-embed-text | ~20 | ~50ms |

### CPU Optimization:

```bash
# Set thread count
export OMP_NUM_THREADS=4  # RPi 4
export OMP_NUM_THREADS=8  # RPi 5

# Use ONNX (faster than PyTorch on CPU)
pip3 install onnxruntime
# Configure: "provider": {"type": "onnx", "provider": "CPUExecutionProvider"}
```

---

## GPU Mode

### When to use GPU:

- Jetson (CUDA available)
- Desktop with NVIDIA GPU
- Production with high throughput
- Brain proxy (needs fast LLM)

### How to start:

```bash
./start.sh ollama gpu
# or
./start.sh llamacpp gpu
```

### Performance:

| Device | Model | Embeddings/sec | Latency |
|--------|-------|----------------|---------|
| Jetson Nano | nomic-embed-text | ~30 | ~30ms |
| Jetson Orin | nomic-embed-text | ~100 | ~10ms |
| RTX 4060 | nomic-embed-text | ~300 | ~3ms |
| RTX 4090 | nomic-embed-text | ~500 | ~2ms |

### GPU Setup:

```bash
# Jetson (pre-installed CUDA)
pip3 install torch torchvision

# Desktop
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Benchmark Your Device

Run the benchmark script to find optimal settings:

```bash
cd benchmarks
python3 benchmark_all.py --device [cpu|gpu] --provider [ollama|onnx|llamacpp]
```

### What it measures:

1. **Cold start** — First embedding (model loading)
2. **Warm embedding** — Subsequent embeddings
3. **Batch embedding** — Multiple texts at once
4. **Recall latency** — Vector search speed
5. **Memory usage** — RAM/VRAM consumption

### Example output:

```
=== Benchmark Results ===
Device: Raspberry Pi 5 (CPU)
Provider: Ollama (nomic-embed-text)

Cold start:     2.3s
Warm embedding: 48ms
Batch (10):     320ms (32ms/text)
Recall (k=5):   15ms
RAM:            342MB

Recommendation: CPU mode is optimal for this device.
```

---

## Brain Proxy Performance

The brain proxy injects memories into every LLM call. This requires a CHAT model.

### Latency impact:

| Model | Location | Latency | Notes |
|-------|----------|---------|-------|
| llama3.2:1b | Ollama CPU | ~500ms | Acceptable |
| llama3.2:3b | Ollama CPU | ~1s | Slow |
| llama3.2:3b | Ollama GPU | ~200ms | Good |
| llama3.1:8b | Ollama GPU | ~300ms | Better quality |
| llama3.2:3b | llama.cpp CUDA | ~80ms | Fast |

### Recommendation:

- **RPi 4/5**: Ollama + llama3.2:1b (CPU, ~500ms)
- **Jetson**: Ollama + llama3.2:3b (GPU, ~200ms)
- **Desktop**: Ollama + llama3.1:8b (GPU, ~300ms)

---

## Troubleshooting

### GPU not detected:

```bash
# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Check Jetson
cat /etc/nv_tegra_release

# Install CUDA support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Slow on CPU:

```bash
# Use ONNX instead of PyTorch
pip3 install onnxruntime

# Reduce model size
ollama pull nomic-embed-text  # Instead of mxbai-embed-large

# Increase threads
export OMP_NUM_THREADS=$(nproc)
```

### Out of memory:

```bash
# Check RAM
free -h

# Use smaller model
ollama pull llama3.2:1b  # Instead of 3b

# Use ONNX INT8 (smallest)
# Config: "provider": {"type": "onnx", "provider": "CPUExecutionProvider"}
```

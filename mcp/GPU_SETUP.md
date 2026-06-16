# GPU Acceleration Guide

## Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| CUDA Toolkit | 12.x | GPU compute backend |
| cuDNN | 8.9+ | Neural network primitives |
| PyTorch | 2.0+ (CUDA build) | Tensor operations |
| onnxruntime-gpu | 1.26.0 | ONNX inference on GPU |

## Verify GPU Is Working

```python
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # "NVIDIA GeForce RTX 4060"
print(torch.cuda.memory_allocated(0) / 1024**2, "MB")
```

```bash
# Check CUDA from terminal
nvidia-smi
# Should show GPU name, VRAM, CUDA version
```

## Install Steps

### 1. CUDA Toolkit

```bash
# Verify installed
nvcc --version
# Should show CUDA 12.x
```

If not installed: https://developer.nvidia.com/cuda-downloads

### 2. PyTorch with CUDA

```bash
# For CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. onnxruntime-gpu

```bash
pip install onnxruntime-gpu==1.26.0 --no-cache-dir

# Verify
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should include: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Known Issues

### INT8 ONNX Models + CUDA EP

**Problem**: INT8 quantized ONNX models fail with `CUDAExecutionProvider`:
```
[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation 
for the node. Supports only the following opsets: 21 and below.
```

**Solution**: Use FP32 ONNX models with CUDA, or INT8 with CPU only.

```python
# INT8 models — force CPU
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# FP32 models — use CUDA
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
```

### DML (DirectML) Reshape Error

**Problem**: DirectML EP fails on ONNX models with dynamic shapes:
```
DmlFallbackManager.cpp:126 DmlFallbackToCpuIfRequired ... DML EP does not 
support Reshape with an empty shape tensor
```

**Solution**: Use CUDA EP instead of DML, or set `ORT_DML_DISABLE_GRAPH_FUSION=1`.

### CUDA OOM (Out of Memory)

**Problem**: Large models exceed GPU VRAM.

**Solution**: Use CPU fallback or quantized models:
```python
# Check available VRAM
import torch
free, total = torch.cuda.mem_get_info()
print(f"Free: {free/1024**3:.1f} GB / Total: {total/1024**3:.1f} GB")
```

Model VRAM requirements:
- bge-large (FP32): ~700 MB ← MATHIR default
- nomic (FP32): ~300 MB
- MiniLM (FP32): ~150 MB
- Qwen2.5-7B (FP32): ~14 GB (requires 16GB+ GPU)

## Daemon Architecture with GPU

```
┌─────────────────────────────────────────┐
│            mathir_daemon.py             │
│         (persistent process)            │
├─────────────────────────────────────────┤
│  TCP Socket (127.0.0.1:7338)           │
│  JSON-RPC protocol                     │
├─────────────────────────────────────────┤
│  SentenceTransformer (CUDA)            │
│  ├── BAAI/bge-large-en-v1.5 (default)  │
│  └── CPU fallback available             │
├─────────────────────────────────────────┤
│  Model loaded in VRAM                  │
│  (no cold start per request)           │
├─────────────────────────────────────────┤
│  HybridSearch auto-scaling             │
│  ├── numpy brute-force (N < 5K)        │
│  └── USearch HNSW mmap (N >= 5K)       │
└─────────────────────────────────────────┘
```

### Why Persistent Daemon?

Without daemon: each request loads model (~2-5s cold start)
With daemon: model stays loaded, first request ~22ms, subsequent ~10ms

```bash
# Start daemon (model loads once, stays in VRAM)
python bin/mathir_daemon.py

# Client requests are instant — model already loaded
python bin/mathir_client.py ping  # <1ms
python bin/mathir_client.py save "test content"  # ~22ms
```

## GPU vs CPU Decision

| Factor | CPU | GPU |
|--------|-----|-----|
| Setup complexity | Low | Medium |
| RAM requirement | Model size only | Model + VRAM |
| Latency (1024d) | ~25ms | ~3ms |
| Throughput | 40 req/s | 200+ req/s |
| Multi-model | Limited by RAM | Limited by VRAM |
| Cost | Free | GPU hardware |

**When GPU matters:**
- >100 concurrent agents
- Batch embedding (>1000 chunks)
- Large models (Qwen2.5-7B)

**When CPU is fine:**
- Single agent
- <100 embeddings/minute
- Small models (MiniLM, nomic)

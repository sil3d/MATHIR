# CUDA Setup — MATHIR Embeddings & GPU Acceleration

## Current Working Stack

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11 | |
| PyTorch | 2.6.0+cu124 | CUDA 12.4 support |
| onnxruntime-gpu | 1.26.0 | CUDAExecutionProvider + TensorrtExecutionProvider |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU | Compute capability 8.9 |

---

## Quick Verification

Run these commands to confirm your setup is correct:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:
```
PyTorch: 2.6.0+cu124
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

### ONNX Runtime GPU Check

```python
import onnxruntime as ort
print("Available providers:")
for p in ort.get_available_providers():
    print(f"  - {p}")
```

Expected:
```
Available providers:
  - TensorrtExecutionProvider
  - CUDAExecutionProvider
  - CPUExecutionProvider
```

---

## Recommended Embedding Setup

**Use sentence-transformers with PyTorch CUDA** (not ONNX backend).

### Default Model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
# 1024d embeddings, high quality
```

### Alternative Models

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `BAAI/bge-large-en-v1.5` | 1024 | Default, high quality |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Matryoshka (variable dims) |
| `all-MiniLM-L6-v2` | 384 | Fast, lightweight |

### Why NOT to Use ONNX Backend

The `backend="onnx"` option in SentenceTransformer causes **silent CPU fallback**. Even with onnxruntime-gpu installed, embeddings will run on CPU without any warning. Always use the default PyTorch backend with CUDA:

```python
# CORRECT — uses CUDA via PyTorch
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# WRONG — silently falls back to CPU
model = SentenceTransformer("BAAI/bge-large-en-v1.5", backend="onnx")
```

---

## Known Issues

### 1. INT8 ONNX Models Are Not GPU-Compatible

INT8 quantized ONNX models (e.g., Octen INT8 variants) are **not compatible** with CUDAExecutionProvider. They silently fall back to CPU. There is no workaround — use full-precision models if you need GPU acceleration.

### 2. DirectML EP Reshape Errors

DirectMLExecutionProvider causes `Reshape` errors with some ONNX model architectures. If you see reshape-related errors, this is the cause. Use CUDAExecutionProvider instead (which works correctly).

### 3. onnxruntime-gpu Install Timeout

The `onnxruntime-gpu` package is large (~2GB). If pip times out during download:

```bash
pip install onnxruntime-gpu==1.26.0 --no-cache-dir
```

The `--no-cache-dir` flag prevents disk space issues and can resolve timeout problems.

### 4. SentenceTransformer ONNX Silent CPU Fallback

As noted above, `backend="onnx"` does not use GPU even when onnxruntime-gpu is installed. This is by design in the sentence-transformers library. Always use PyTorch backend.

---

## Manual Installation

If you need to set up from scratch:

```bash
conda create -n mathir_cuda python=3.11 -y
conda activate mathir_cuda

# PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Embeddings
pip install sentence-transformers

# MATHIR dependencies
pip install streamlit plotly pandas numpy

# ONNX Runtime GPU (optional, for ONNX models)
pip install onnxruntime-gpu==1.26.0 --no-cache-dir
```

---

## VSCode Configuration

1. **Ctrl + Shift + P** → `Python: Select Interpreter`
2. Select `mathir_cuda` environment
3. Verify bottom right shows: `Python 3.11.x ('mathir_cuda')`

---

## Launch MATHIR

```bash
conda activate mathir_cuda
streamlit run app_streamlit.py
```

---

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `CUDA: False` | Wrong Python interpreter | `conda activate mathir_cuda` |
| `CUDA out of memory` | Insufficient VRAM | Close GPU apps, reduce batch size |
| `CUDAExecutionProvider not found` | onnxruntime-gpu not installed | `pip install onnxruntime-gpu==1.26.0 --no-cache-dir` |
| Reshape error in ONNX | DirectML EP issue | Use CUDAExecutionProvider, not DirectML |
| Embeddings on CPU | `backend="onnx"` used | Remove `backend="onnx"`, use PyTorch |

---

## Performance Comparison

| Task | CPU | GPU (RTX 4060) |
|------|-----|-----------------|
| Embed 1000 documents | ~60s | ~8s |
| Sentence similarity | ~50ms | <10ms |
| Batch encoding (512 docs) | ~30s | ~4s |

---

<div align="center">

**GPU acceleration is active and working!**

</div>

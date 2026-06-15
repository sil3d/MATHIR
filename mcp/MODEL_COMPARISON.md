# Model Comparison

## Benchmark Table

| Model | Dims | Size | CPU Save | CPU Recall | GPU Save | MTEB Avg | License |
|-------|------|------|----------|-----------|----------|----------|---------|
| MiniLM-L6-v2 | 384 | 80 MB | 22ms | 53ms | — | 56.26 | Apache-2.0 |
| nomic-embed-text-v1.5 | 768 | 137 MB | 21ms | 27ms | ~12ms | 62.38 | Apache-2.0 |
| bge-large-en-v1.5 | 1024 | 335 MB | 43ms | 25ms | ~15ms | 64.23 | MIT |
| e5-large-v2 | 1024 | 1.3 GB | 68ms | 45ms | ~18ms | 63.13 | MIT |
| Octen-MiniLM-L6-INT8 | 384 | 22 MB | 8ms | 18ms | — | ~55 | Apache-2.0 |
| Qwen2.5-7B-emb | 3584 | 4.7 GB | — | — | ~30ms | 71.5 | Apache-2.0 |

> Latencies are approximate for encoding a 128-token chunk on a mid-range CPU (Ryzen 7 / i7-12700).
> GPU times assume RTX 3060+ with onnxruntime-gpu.

## Model Profiles

### MiniLM-L6-v2 (384d)
- **Best for**: Fast agent memory, real-time recall
- **Pros**: Tiny, fastest on CPU, 80MB RAM
- **Cons**: Lowest quality, limited nuance
- **Install**: `pip install sentence-transformers`
- **ONNX**: Built-in via Optimum

### nomic-embed-text-v1.5 (768d)
- **Best for**: Default balanced choice, most projects
- **Pros**: Best speed/quality ratio, Matryoshka support, Apache-2.0
- **Cons**: Requires Optimum for ONNX export
- **Install**: `pip install optimum`
- **ONNX**: Export via `optimum-cli export onnx`

### bge-large-en-v1.5 (1024d)
- **Best for**: High-quality retrieval, production systems
- **Pros**: Highest MTEB under 1024d, Matryoshka support
- **Cons**: 335MB, slower on CPU
- **Install**: `pip install sentence-transformers`
- **ONNX**: Via Optimum

### e5-large-v2 (1024d)
- **Best for**: Research, maximum MTEB score
- **Pros**: Strong retrieval performance
- **Cons**: 1.3GB, slowest on CPU, no INT8 quantization
- **Install**: `pip install sentence-transformers`

### Octen-MiniLM-L6-INT8 (384d)
- **Best for**: Edge deployment, minimal resources
- **Pros**: 22MB, fastest inference, INT8 quantized
- **Cons**: Lowest quality, INT8 incompatible with CUDA EP
- **Install**: Pre-quantized ONNX from OctoAI

### Qwen2.5-7B-emb (3584d)
- **Best for**: Maximum quality, research, GPU servers
- **Pros**: Highest MTEB, 3584 dimensions, best accuracy
- **Cons**: 4.7GB, GPU required, high VRAM usage
- **Install**: `pip install transformers accelerate`
- **ONNX**: Via Optimum (GPU only)

## Recommendation Matrix

| Scenario | Model | Why |
|----------|-------|-----|
| Local agent, speed matters | MiniLM-L6-v2 | 22ms save, 80MB |
| Default for most projects | nomic-embed-text-v1.5 | Best balance, Matryoshka |
| Production with quality SLA | bge-large-en-v1.5 | 64.23 MTEB, 1024d |
| Edge / IoT device | Octen-INT8 | 22MB, 8ms |
| GPU server, max quality | Qwen2.5-7B-emb | 71.5 MTEB, 3584d |
| Research benchmarks | e5-large-v2 | Strong MTEB |

## MTEB Scores (Retrieval)

| Model | ndcg@10 | Precision@10 | Recall@10 |
|-------|---------|-------------|-----------|
| Qwen2.5-7B-emb | 71.5 | 72.8 | 88.2 |
| bge-large-en-v1.5 | 64.2 | 65.1 | 84.7 |
| e5-large-v2 | 63.1 | 64.3 | 83.9 |
| nomic-embed-text-v1.5 | 62.4 | 63.5 | 82.1 |
| MiniLM-L6-v2 | 56.3 | 57.2 | 76.4 |

## Vector Storage Cost

```
Model               1K memories    10K memories   100K memories
────────────────────────────────────────────────────────────────
MiniLM (384d)       2 MB           20 MB          200 MB
nomic (768d)        4 MB           40 MB          400 MB
bge-large (1024d)   5 MB           50 MB          500 MB
Qwen2.5 (3584d)     18 MB          180 MB         1.8 GB
```

Excludes HNSW index overhead (~1.5x multiplier).

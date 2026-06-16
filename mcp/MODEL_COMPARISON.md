# Model Comparison

## Benchmark Table

| Model | Dims | Size | CPU Save | CPU Recall | GPU Save | GPU Recall | MTEB Avg | License |
|-------|------|------|----------|-----------|----------|------------|----------|---------|
| MiniLM-L6-v2 | 384 | 80 MB | 22ms | 53ms | — | — | 56.26 | Apache-2.0 |
| nomic-embed-text-v1.5 | 768 | 137 MB | 21ms | 27ms | ~12ms | ~10ms | 62.38 | Apache-2.0 |
| bge-large-en-v1.5 | 1024 | 335 MB | 43ms | 25ms | **3ms** | **3ms** | 64.23 | MIT |
| e5-large-v2 | 1024 | 1.3 GB | 68ms | 45ms | ~18ms | ~15ms | 63.13 | MIT |
| Octen-MiniLM-L6-INT8 | 384 | 22 MB | 8ms | 18ms | — | — | ~55 | Apache-2.0 |
| Qwen2.5-7B-emb | 3584 | 4.7 GB | — | — | ~30ms | ~25ms | 71.5 | Apache-2.0 |

> GPU times: RTX 4060 Laptop GPU, CUDA 12.4, torch 2.6.0+cu124.
> CPU times: mid-range CPU (Ryzen 7 / i7-12700).

## Model Profiles

### bge-large-en-v1.5 (1024d) — MATHIR DEFAULT
- **Best for**: Default choice, production systems
- **Pros**: Best quality under 1024d, CUDA ~3ms/text, Matryoshka support
- **Cons**: 335MB, slower on CPU (43ms)
- **Install**: `pip install sentence-transformers`
- **GPU**: CUDA via SentenceTransformer (no ONNX needed)

### nomic-embed-text-v1.5 (768d)
- **Best for**: Balanced alternative, most projects
- **Pros**: Best speed/quality ratio, Matryoshka support, Apache-2.0
- **Cons**: Requires Optimum for ONNX export
- **Install**: `pip install optimum`
- **ONNX**: Export via `optimum-cli export onnx`

### MiniLM-L6-v2 (384d)
- **Best for**: Edge deployment, minimal resources
- **Pros**: Tiny, 80MB RAM, fastest on CPU
- **Cons**: Lowest quality, limited nuance
- **Install**: `pip install sentence-transformers`

### e5-large-v2 (1024d)
- **Best for**: Research, alternative to bge-large
- **Pros**: Strong retrieval performance
- **Cons**: 1.3GB, slowest on CPU
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
| Default for MATHIR | bge-large-en-v1.5 | 1024d, CUDA 3ms, 64.23 MTEB |
| Alternative balanced | nomic-embed-text-v1.5 | 768d, best speed/quality ratio |
| Edge / IoT device | Octen-INT8 | 22MB, 8ms CPU |
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
bge-large (1024d)   5 MB           50 MB          500 MB  ← MATHIR default
Qwen2.5 (3584d)     18 MB          180 MB         1.8 GB
```

Excludes HNSW index overhead (~1.5x multiplier).

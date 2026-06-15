# Embedding Dimensions Guide

## What Are Embedding Dimensions?

Embedding dimensions define the vector size representing each text chunk. Higher dimensions capture more nuance but cost more in speed, storage, and RAM.

| Dimensions | Example Model | Vector Size (bytes) | SQLite Index (1K memories) |
|-----------|---------------|---------------------|---------------------------|
| 384 | MiniLM-L6-v2 | 1,536 | ~2 MB |
| 768 | nomic-embed-text-v1.5 | 3,072 | ~4 MB |
| 1024 | bge-large-en-v1.5 | 4,096 | ~5 MB |
| 3584 | Qwen2.5-7B | 14,336 | ~18 MB |

## Why Dimensions Matter

### Quality vs Speed Trade-off

```
Quality:  384d ████████░░░░░░░ 60%
          768d ██████████░░░░░ 80%
          1024d ███████████░░░ 90%
          3584d ██████████████ 100%

Speed:    384d ██████████████ 100% (fastest)
          768d ████████████░░ 85%
          1024d ██████████░░░ 70%
          3584d █████░░░░░░░░ 35%

Storage:  384d ██░░░░░░░░░░░ 15%
          768d ████░░░░░░░░░ 30%
          1024d ██████░░░░░░ 40%
          3584d █████████████ 100%
```

### Recommendation by Use Case

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Fast agent memory | 384d (MiniLM) | Lowest latency, good enough for recall |
| Balanced (default) | 768d (nomic) | Best speed/quality ratio, Matryoshka support |
| High-quality retrieval | 1024d (bge-large) | Best MTEB score under 1024d |
| Research / maximum quality | 3584d (Qwen2.5-7B) | Highest accuracy, needs GPU |

### Speed Benchmarks

| Model | Dims | Save Latency | Recall Latency (k=10) |
|-------|------|-------------|----------------------|
| MiniLM-L6-v2 | 384 | 22ms | 53ms |
| nomic-embed-text-v1.5 | 768 | 21ms | 27ms |
| bge-large-en-v1.5 | 1024 | 43ms | 25ms |
| Qwen2.5-7B | 3584 | ~30ms (GPU) | ~40ms (GPU) |

> Note: nomic-embed saves faster than MiniLM despite higher dims because ONNX optimization is better.

## Matryoshka Embedding

Some models (nomic, bge) support **Matryoshka representation learning (MRL)** — you can truncate embeddings without re-embedding:

```python
# nomic-embed outputs 768d, but you can use first 384d
embedding = model.encode("text")  # shape: (768,)
truncated = embedding[:384]       # shape: (384,) — still valid!
```

This lets you:
1. Start at 768d for quality
2. Drop to 384d if speed matters later
3. No re-embedding of existing memories

## Storage Implications

SQLite-vec creates an index per dimension. Storage grows linearly:

```
1,000 memories × 768d × 4 bytes = 3 MB
10,000 memories × 768d × 4 bytes = 30 MB
100,000 memories × 768d × 4 bytes = 300 MB
```

With HNSW index overhead (~1.5x), multiply by 1.5.

## Dimension Change Handling

MATHIR auto-detects dimension mismatches:

1. On startup, reads first embedding from DB
2. Compares dimensions against loaded model
3. If mismatch → drops old `vec0` table
4. Rebuilds vec0 with correct dimensions
5. All memories preserved (content + metadata intact)

**Warning**: vec0 rebuild can take minutes for large databases. Existing memories are not lost — only the vector index is recreated.

```python
# Auto-detection in mathir_daemon.py
existing_dim = db.execute("SELECT vec_length(embedding) FROM memory LIMIT 1").fetchone()
if existing_dim and existing_dim[0] != model_dim:
    db.execute("DROP TABLE IF EXISTS vec0")
    db.execute(create_vec0_sql(model_dim))  # Recreate with correct dims
```

## Choosing Your Dimension

| Priority | Recommendation |
|----------|---------------|
| Speed first | 384d MiniLM |
| Balance first | 768d nomic (default) |
| Quality first | 1024d bge-large |
| Accuracy first | 3584d Qwen2.5-7B (GPU required) |
| Unknown | 768d nomic (safe default) |

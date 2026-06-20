# Embedding Dimensions Guide

**Current model: paraphrase-multilingual-MiniLM-L12-v2 (384d, 239MB VRAM)**

---

## Why Change the Model?

The default model (384d) is optimized for **speed and low VRAM**. But you might want to change for:

| Reason | What to do |
|--------|-----------|
| Better recall quality | Upgrade to 768d or 1024d |
| English-only project | Use MiniLM-L6-v2 (smaller, faster) |
| Multilingual priority | Stay with paraphrase-multilingual (50+ langs) |
| Maximum quality | Use Qwen2.5-7B (3584d, GPU required) |
| Edge / minimal RAM | Use Octen-MiniLM-L6-INT8 (22MB) |

---

## How to Change Model (Step by Step)

### Step 1: Choose your model

| Model | Dims | VRAM | Speed | Quality | Languages |
|-------|------|------|-------|---------|-----------|
| **paraphrase-multilingual-MiniLM-L12-v2** | 384 | 239MB | ~104ms | ★★★☆☆ | 50+ |
| MiniLM-L6-v2 | 384 | 80MB | ~22ms | ★★★☆☆ | English |
| nomic-embed-text-v1.5 | 768 | ~500MB | ~21ms | ★★★★☆ | 100+ |
| BAAI/bge-large-en-v1.5 | 1024 | ~1.5GB | ~3ms | ★★★★☆ | English |
| e5-large-v2 | 1024 | ~1.3GB | ~3ms | ★★★★☆ | English |
| Qwen2.5-7B-emb | 3584 | ~4.7GB | ~30ms | ★★★★★ | 100+ |

### Step 2: Install the model

```bash
# For paraphrase-multilingual (default) — already installed
pip install sentence-transformers

# For nomic
pip install sentence-transformers
# Model downloads automatically on first use

# For bge-large
pip install sentence-transformers
# Model downloads automatically on first use

# For Qwen2.5-7B (requires GPU)
pip install sentence-transformers
# Model downloads automatically (~4.7GB)
```

### Step 3: Update config

Edit `~/.config/MATHIR/config/mathir.json`:

```json
{
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "device": "cuda",
  "embedding_dim": 384,
  "port": 7338,
  "db_path": ".mathir/mathir.db"
}
```

**Change to nomic (768d):**
```json
{
  "model": "sentence-transformers/nomic-embed-text-v1.5",
  "device": "cuda",
  "embedding_dim": 768,
  "port": 7338,
  "db_path": ".mathir/mathir.db"
}
```

**Change to bge-large (1024d):**
```json
{
  "model": "sentence-transformers/BAAI/bge-large-en-v1.5",
  "device": "cuda",
  "embedding_dim": 1024,
  "port": 7338,
  "db_path": ".mathir/mathir.db"
}
```

### Step 4: Migrate existing database (IMPORTANT!)

If you have existing memories, you MUST migrate them to the new dimensions:

```bash
# Backup first!
cp .mathir/mathir.db .mathir/mathir.db.backup

# Migrate to new dimensions
python ~/.config/MATHIR/dev/migrate_db.py --db .mathir/mathir.db --new-dim 768

# Or for 1024d
python ~/.config/MATHIR/dev/migrate_db.py --db .mathir/mathir.db --new-dim 1024
```

**Without migration:** Existing memories become unusable (dimension mismatch). The daemon will auto-rebuild the vec0 table, but all vector data is lost.

### Step 5: Restart daemon

```bash
# Kill existing daemon
python ~/.config/MATHIR/mathir_lib/mathir_client.py stop

# Start new daemon
python ~/.config/MATHIR/mathir_lib/mathir_daemon.py
```

### Step 6: Verify

```bash
# Check model loaded correctly
python ~/.config/MATHIR/mathir_lib/mathir_client.py ping

# Test save + recall
python ~/.config/MATHIR/mathir_lib/mathir_client.py save "test memory" -a test -t semantic -l test
python ~/.config/MATHIR/mathir_lib/mathir_client.py recall "test" -k 1
```

---

## Quick Reference: Common Upgrades

### Upgrade 384d → 768d (nomic)

```bash
# 1. Update config
# Edit ~/.config/MATHIR/config/mathir.json:
# "model": "sentence-transformers/nomic-embed-text-v1.5"
# "embedding_dim": 768

# 2. Migrate DB
python ~/.config/MATHIR/dev/migrate_db.py --db .mathir/mathir.db --new-dim 768

# 3. Restart daemon
python ~/.config/MATHIR/mathir_lib/mathir_client.py stop
python ~/.config/MATHIR/mathir_lib/mathir_daemon.py
```

### Upgrade 384d → 1024d (bge-large)

```bash
# 1. Update config
# Edit ~/.config/MATHIR/config/mathir.json:
# "model": "sentence-transformers/BAAI/bge-large-en-v1.5"
# "embedding_dim": 1024

# 2. Migrate DB
python ~/.config/MATHIR/dev/migrate_db.py --db .mathir/mathir.db --new-dim 1024

# 3. Restart daemon
python ~/.config/MATHIR/mathir_lib/mathir_client.py stop
python ~/.config/MATHIR/mathir_lib/mathir_daemon.py
```

### Downgrade 1024d → 384d (if VRAM limited)

```bash
# 1. Update config
# Edit ~/.config/MATHIR/config/mathir.json:
# "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# "embedding_dim": 384

# 2. Migrate DB
python ~/.config/MATHIR/dev/migrate_db.py --db .mathir/mathir.db --new-dim 384

# 3. Restart daemon
python ~/.config/MATHIR/mathir_lib/mathir_client.py stop
python ~/.config/MATHIR/mathir_lib/mathir_daemon.py
```

---

## What Are Embedding Dimensions?

Embedding dimensions define the vector size representing each text chunk. Higher dimensions capture more nuance but cost more in speed, storage, and RAM.

**MATHIR default: 384d** — best balance of quality, speed, and memory usage.

| Dimensions | Example Model | Vector Size (bytes) | SQLite Index (1K memories) |
|-----------|---------------|---------------------|---------------------------|
| **384** | **paraphrase-multilingual-MiniLM-L12-v2** | **1,536** | **~2 MB** |
| 768 | nomic-embed-text-v1.5 | 3,072 | ~4 MB |
| 1024 | BAAI/bge-large-en-v1.5 | 4,096 | ~5 MB |
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
| Default (MATHIR) | 384d (paraphrase-multilingual) | Best VRAM ratio, 50+ langs, ~240MB GPU |
| Balanced alternative | 768d (nomic) | Good speed/quality ratio |
| Maximum quality | 1024d (BAAI/bge-large-en-v1.5) | High quality, needs 1.5GB GPU |
| Edge / minimal | 384d (MiniLM-L6-v2) | Smallest, English-only |

### Speed Benchmarks (RTX 4060, CUDA)

| Model | Dims | Save Latency | Recall Latency (k=10) |
|-------|------|-------------|----------------------|
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | ~104ms/sent | ~140ms (k=3) |
| nomic-embed-text-v1.5 | 768 | 21ms | 27ms |
| BAAI/bge-large-en-v1.5 | 1024 | 3ms (CUDA) | 3ms (CUDA) |
| e5-large-v2 | 1024 | 2.9ms (CUDA) | 2.9ms (CUDA) |
| Qwen2.5-7B | 3584 | ~30ms (GPU) | ~40ms (GPU) |

> paraphrase-multilingual-MiniLM-L12-v2: 384d, 50+ languages, ~240MB VRAM, 0.929 cosine sim FR↔EN verified.

## Matryoshka Embedding

Some models (nomic, bge) support **Matryoshka representation learning (MRL)** — you can truncate embeddings without re-embedding:

```python
# bge-large outputs 1024d, but you can use first 768d or 384d
embedding = model.encode("text")  # shape: (1024,)
truncated = embedding[:768]       # shape: (768,) — still valid!
truncated = embedding[:384]       # shape: (384,) — still valid!
```

This lets you:
1. Start at 1024d for quality
2. Drop to 768d or 384d if speed matters later
3. No re-embedding of existing memories

## Storage Implications

SQLite-vec creates an index per dimension. Storage grows linearly:

```
1,000 memories × 1024d × 4 bytes = 4 MB
10,000 memories × 1024d × 4 bytes = 40 MB
100,000 memories × 1024d × 4 bytes = 400 MB
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
| Default (MATHIR) | 384d paraphrase-multilingual (50+ langs, low VRAM) |
| Balance first | 768d nomic |
| Quality first | 3584d Qwen2.5-7B (GPU required) |
| Edge / minimal | 384d MiniLM (CPU only) |

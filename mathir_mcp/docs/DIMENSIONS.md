# Embedding Dimensions Guide (v8.5.0)

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

## Real, Independently-Verified Alternative (Same 384d Footprint)

Unlike the MTEB numbers quoted elsewhere in this doc (borrowed from public
leaderboards, not run by this project), the comparison below was actually
executed against this project's own real BEIR corpora (scifact, nfcorpus,
arguana) using the real `beir.retrieval.evaluation.EvaluateRetrieval`
methodology, on 2026-06-30 (see
`benchmarks/06_results/current/embedding_model_comparison.json` for the raw
output and `benchmarks/07_utilities/compare_embedding_models.py` to
reproduce it):

| Dataset | Default (paraphrase-multilingual-MiniLM-L12-v2) | `intfloat/multilingual-e5-small` | Encoding speed |
|---|---|---|---|
| scifact | nDCG@10 0.4837 | **nDCG@10 0.6770** (+40%) | e5-small ~5x slower |
| nfcorpus | nDCG@10 0.2345 | **nDCG@10 0.3100** (+32%) | e5-small ~5x slower |
| arguana | **nDCG@10 0.4488** | nDCG@10 0.3908 (-13%) | e5-small ~1.8x slower |

Both models are the same size class (384d, ~118M params, similar
edge/low-VRAM footprint) — this is a same-footprint quality/speed
trade-off, not an upgrade to a bigger model. `multilingual-e5-small` is
trained specifically for retrieval (requires `"query: "` / `"passage: "`
text prefixes — see the comparison script for the exact usage) and wins
clearly on factual/QA-style retrieval (scifact, nfcorpus — closer to
MATHIR's typical "find a fact in my memories" use case), but loses on
argument-similarity-style retrieval (arguana) where the current default's
paraphrase-training is actually a better fit. It is also meaningfully
slower to encode, which matters for edge/resource-constrained deployments
(MATHIR explicitly targets running locally without cloud dependency).

**MATHIR's default was deliberately NOT changed** based on this result —
the current 384d paraphrase-multilingual model remains the default
specifically to preserve edge-device speed and because the trade-off is
task-dependent, not a clean win. If your use case is factual/QA-style
retrieval and you can accept slower encoding, set in `mathir.json`:

```json
{
  "model": "intfloat/multilingual-e5-small",
  "embedding_dim": 384
}
```

then follow the migration steps below (dimension is unchanged at 384d, so
no vec0 rebuild is needed — but re-embedding existing content is still
recommended for consistent similarity scores across old and new memories,
since the two models produce different embedding spaces at the same
dimensionality).

## Where MATHIR's Retrieval Quality Gap Actually Comes From (Investigation Notes)

A live benchmark (`benchmarks/09_mathir_vs_faiss_stress/`) showed MATHIR's
`hybrid_search` trailing a stronger-embedder FAISS baseline on real BEIR
data. Two follow-up investigations (2026-07-01,
`benchmarks/07_utilities/isolate_mathir_retrieval_bug.py` and
`test_rrf_weights.py`) isolated exactly why, so this doesn't get
misdiagnosed later:

1. **MATHIR's vector search mechanism itself is not the problem.** Holding
   the embedder fixed and comparing raw FAISS `IndexFlatIP` against
   `VecMemory.search()`'s real code path on the exact same embeddings
   (nfcorpus) produced an **identical** nDCG@10 (0.2345 = 0.2345) — sqlite-vec's
   exact brute-force cosine search is mathematically equivalent to FAISS at
   these corpus sizes. (It is ~500x slower per-query in this specific
   unindexed comparison, a separate performance question, not a quality one.)
2. **The RRF fusion default weights (vector_weight=1.0, bm25_weight=1.0)
   are already near-optimal for the current embedder** — sweeping
   vector_weight from 1 to 10 (favoring the vector signal more) made
   nfcorpus nDCG@10 *worse* (0.3056 → 0.2583), not better. With a weaker
   embedder, BM25's lexical signal compensates for semantic weakness rather
   than diluting it; this only flips (BM25 hurting) once the embedder is
   already strong, per the `multi_dataset_efficient.py` results using
   bge-base-en-v1.5.
3. **Conclusion: the quality gap is the embedding model's retrieval-specific
   training, full stop** — not a search bug, not a fusion-weight
   misconfiguration. The trade-off is real and belongs to whoever picks the
   model (see the table above), not something further code changes here can
   fix without changing that choice.

### Rejected idea: confidence-gated adaptive fusion

A natural next architecture idea, given point 2 above: gate BM25 fusion
per-query on the dense ranking's own confidence (e.g. the top1/top2 score
margin) — fuse only when the dense ranking looks ambiguous, trust dense-only
otherwise. This was tested empirically
(`benchmarks/07_utilities/test_adaptive_fusion_hypothesis.py`, 2026-07-01,
nfcorpus, median-split by margin) **before** writing any server code, and
the hypothesis did not hold: hybrid fusion improved nDCG@10 in *both* the
low-confidence bucket (0.1946 → 0.2847, +0.09) *and* the high-confidence
bucket (0.2746 → 0.3267, +0.05) with the current embedder. There is no
per-query signal here to gate on — the real, validated factor is the
*embedder's overall strength* (point 2), not per-query ambiguity. Do not
build a confidence-gated fusion mechanism on this premise; it would add
real complexity for no measured benefit. If a stronger embedder is adopted
later, this same test should be re-run with that embedder before deciding
whether hybrid should be disabled outright versus kept adaptive — the two
outcomes are indistinguishable without re-testing.

### Rejected idea: embedding-space pseudo-relevance feedback (PRF)

A genuinely novel (self-written, not FAISS/BM25/RRF/CE-based) two-pass
retrieval idea: refine the query embedding using a score-weighted centroid
of the first pass's own top-m results before searching again (Rocchio-style,
implemented from scratch in
`benchmarks/07_utilities/novel_algo_embedding_prf.py`, 2026-07-01). Swept
m ∈ {3,5,10} and blend weight β ∈ {0.1,0.25,0.5,1.0} on nfcorpus and scifact.

Result: small, inconsistent, dataset-dependent. Best config improved
nfcorpus nDCG@10 by +0.0063 (0.2345→0.2408) but the best config on scifact
still *lost* -0.0049 (0.4837→0.4787) relative to plain single-pass dense
search, and every other tested (m, β) combination on scifact lost more.
This is the classic PRF "query drift" failure mode: when the first pass's
top-m already contains false positives (more likely when the baseline
dense ranking is already fairly good, as on scifact), blending them back
into the query moves it toward noise rather than signal.

This is now the **third** independent technique (after hybrid BM25 fusion
and cross-encoder reranking) that shows the exact same pattern: it helps
when the baseline dense signal is weak and hurts when the baseline dense
signal is already strong. That's a real, generalizable finding in its own
right — any secondary/augmentation signal added on top of this embedder's
dense search inherits this same trade-off. Not adopted as a default; the
script is kept for reproducibility and to save whoever revisits this idea
from re-discovering the same drift problem from scratch.

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
python -m mathir_lib.mathir_client stop

# Start new daemon
python -m mathir_mcp
```

### Step 6: Verify

```bash
# Check model loaded correctly
python -m mathir_lib.mathir_client ping

# Test save + recall
python -m mathir_lib.mathir_client save "test memory" -a test -t semantic -l test
python -m mathir_lib.mathir_client recall "test" -k 1
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
python -m mathir_lib.mathir_client stop
python -m mathir_mcp
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
python -m mathir_lib.mathir_client stop
python -m mathir_mcp
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
python -m mathir_lib.mathir_client stop
python -m mathir_mcp
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
# Auto-detection in mathir_server.py
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

---

## Model Benchmarks

*(merged from MODEL_COMPARISON.md — v8.3+)*

### Benchmark Table

| Model | Dims | Size | CPU Save | CPU Recall | GPU Save | GPU Recall | MTEB Avg | License |
|-------|------|------|----------|-----------|----------|------------|----------|---------|
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 471 MB / 239 MB fp16 | ~104ms/sent | ~140ms | ~104ms/sent | ~140ms | ~49.7 (Eng) / 49.4 (Multi) | Apache-2.0 |
| MiniLM-L6-v2 | 384 | 80 MB | 22ms | 53ms | — | — | 56.26 | Apache-2.0 |
| nomic-embed-text-v1.5 | 768 | 137 MB | 21ms | 27ms | ~12ms | ~10ms | 62.38 | Apache-2.0 |
| bge-large-en-v1.5 | 1024 | 335 MB | 43ms | 25ms | **3ms** | **3ms** | 64.23 | MIT |
| e5-large-v2 | 1024 | 1.3 GB | 68ms | 45ms | ~18ms | ~15ms | 63.13 | MIT |
| Octen-MiniLM-L6-INT8 | 384 | 22 MB | 8ms | 18ms | — | — | ~55 | Apache-2.0 |
| Qwen2.5-7B-emb | 3584 | 4.7 GB | — | — | ~30ms | ~25ms | 71.5 | Apache-2.0 |

> GPU times: RTX 4060 Laptop GPU, CUDA 12.4, torch 2.6.0+cu124.
> CPU times: mid-range CPU (Ryzen 7 / i7-12700).

### Model Profiles

#### paraphrase-multilingual-MiniLM-L12-v2 (384d) — MATHIR DEFAULT
- **Best for**: Multilingual projects (FR/EN/DE/ES/JA/ZH), low VRAM
- **Pros**: 50+ languages, Apache-2.0, 43.8M downloads, 471MB CPU / 239MB fp16 GPU
- **Cons**: Lower MTEB English (~49.7 vs bge-large 64.2), 128 token max (chunking needed)
- **Install**: pip install sentence-transformers
- **GPU**: CUDA fp16 via SentenceTransformer
- **Verified**: 0.929 cosine sim "Bonjour le monde" ↔ "Hello world" (cross-lingual)

#### nomic-embed-text-v1.5 (768d)
- **Best for**: Balanced alternative, most projects
- **Pros**: Best speed/quality ratio, Matryoshka support, Apache-2.0
- **Cons**: Requires Optimum for ONNX export
- **Install**: pip install optimum
- **ONNX**: Export via optimum-cli export onnx

#### MiniLM-L6-v2 (384d)
- **Best for**: Edge deployment, minimal resources
- **Pros**: Tiny, 80MB RAM, fastest on CPU
- **Cons**: Lowest quality, limited nuance
- **Install**: pip install sentence-transformers

#### e5-large-v2 (1024d)
- **Best for**: Research, alternative to bge-large
- **Pros**: Strong retrieval performance
- **Cons**: 1.3GB, slowest on CPU
- **Install**: pip install sentence-transformers

#### Octen-MiniLM-L6-INT8 (384d)
- **Best for**: Edge deployment, minimal resources
- **Pros**: 22MB, fastest inference, INT8 quantized
- **Cons**: Lowest quality, INT8 incompatible with CUDA EP
- **Install**: Pre-quantized ONNX from OctoAI

#### Qwen2.5-7B-emb (3584d)
- **Best for**: Maximum quality, research, GPU servers
- **Pros**: Highest MTEB, 3584 dimensions, best accuracy
- **Cons**: 4.7GB, GPU required, high VRAM usage
- **Install**: pip install transformers accelerate
- **ONNX**: Via Optimum (GPU only)

### Recommendation Matrix

| Scenario | Model | Why |
|----------|-------|-----|
| Default for MATHIR | paraphrase-multilingual-MiniLM-L12-v2 | 384d, 50+ languages, low VRAM (239MB fp16) |
| Alternative balanced | nomic-embed-text-v1.5 | 768d, best speed/quality ratio, Apache-2.0 |
| Edge / IoT device | Octen-INT8 | 22MB, 8ms CPU |
| GPU server, max quality | Qwen2.5-7B-emb | 71.5 MTEB, 3584d |
| Research benchmarks | e5-large-v2 | Strong MTEB |
| High-quality English (previous default) | bge-large-en-v1.5 | 1024d, CUDA 3ms, 64.23 MTEB |

### MTEB Scores (Retrieval)

| Model | ndcg@10 | Precision@10 | Recall@10 |
|-------|---------|-------------|-----------|
| Qwen2.5-7B-emb | 71.5 | 72.8 | 88.2 |
| bge-large-en-v1.5 | 64.2 | 65.1 | 84.7 |
| e5-large-v2 | 63.1 | 64.3 | 83.9 |
| nomic-embed-text-v1.5 | 62.4 | 63.5 | 82.1 |
| MiniLM-L6-v2 | 56.3 | 57.2 | 76.4 |

### Vector Storage Cost

`
Model               1K memories    10K memories   100K memories
────────────────────────────────────────────────────────────────
MiniLM (384d)       2 MB           20 MB          200 MB
paraphrase-multilingual (384d) 2 MB   20 MB          200 MB  ← MATHIR default
nomic (768d)        4 MB           40 MB          400 MB
bge-large (1024d)   5 MB           50 MB          500 MB
Qwen2.5 (3584d)     18 MB          180 MB         1.8 GB
`

Excludes HNSW index overhead (~1.5x multiplier).

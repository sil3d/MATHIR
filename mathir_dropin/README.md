# MATHIR Drop-in

A minimal, self-contained memory plugin you can copy into any project
and use in **5 minutes**.

> **What is MATHIR?** A 4-tier memory system (working, episodic,
> semantic, immune) for AI agents. You feed it embeddings, it stores
> them, blends them through a learned router, and lets you recall
> similar ones later — with the same dict format regardless of whether
> the input is text, image, audio, or video.

This package is a **strict subset** of the full `mathir_lib` research
codebase. It re-implements the canonical 4-tier model in 7 files
(~1500 lines) with one obvious storage format and zero hidden
dependencies.

---

## 🚀 5-Minute Quickstart

### 1. Install (no PyPI, no git — just copy)

```bash
# Copy the mathir_dropin/ folder into your project
cp -r mathir_dropin /your/project/

# Add it to your PYTHONPATH (or just `cd` into the parent)
export PYTHONPATH="/your/project:$PYTHONPATH"
```

**Dependencies**: `torch` and `numpy`. Nothing else.

```bash
pip install torch numpy
```

### 2. Use it (3 lines)

```python
from mathir_dropin import MATHIRMemory
import torch

memory = MATHIRMemory(embedding_dim=384, db_path="agent.db")
memory_id = memory.store(torch.randn(1, 384), {"text": "hello world"})
hits = memory.recall(torch.randn(1, 384), k=3)
print(memory.get_stats())
```

That's it. You now have a 4-tier memory that persists to `agent.db`.

### 3. Try the demo

```bash
python -m mathir_dropin._demo
```

It creates a temp DB, stores 10 memories, runs embedding + FTS5 search,
reopens the DB to prove persistence, and prints the file size.

---

## 🔄 Cross-Provider Portability

MATHIR stores embeddings from multiple providers simultaneously, so you never need to re-embed when switching LLMs.

```python
from mathir_dropin import MATHIRMemory
import torch

memory = MATHIRMemory(embedding_dim=384, db_path="agent.db")

# Store with ALL providers you might use
emb_primary = torch.randn(1, 384)
emb_cohere = torch.randn(1, 1024)
emb_voyage = torch.randn(1, 1024)

mid = memory.store(emb_primary, {"text": "thermodynamics"}, extra_providers={
    "cohere": (emb_cohere, "embed-english-v3.0"),
    "voyage": (emb_voyage, "voyage-4")
})

# Query with ANY provider — MATHIR uses the right embeddings
results = memory.recall(emb_cohere, k=5, provider="cohere")
```

**How it works:**
- `memory_embeddings` table stores one embedding per provider per memory
- `recall(..., provider="cohere")` uses Cohere's embeddings for similarity
- No re-embedding needed when switching LLMs
- Storage: N × 4KB per memory per provider

**Providers supported:** openai, cohere, voyage, ollama, huggingface, direct, or any custom name

---

## 📦 What's in the box

```
mathir_dropin/
├── README.md             ← you are here
├── __init__.py           ← public API: MATHIRMemory, configure, save, load
├── memory.py             ← main class (4 tiers + KL router)
├── store.py              ← SQLite + FTS5 storage
├── config.py             ← default config + validate_config()
├── exceptions.py         ← MATHIRError and 3 specific subclasses
├── _demo.py              ← runnable end-to-end demo
└── tests/
    └── test_memory.py    ← 10 critical tests
```

**Total: ~1500 lines** across 7 files. The full `mathir_lib` has 18+
files and 5000+ lines.

---

## ❓ "Where is the data? What format? Can I inspect it?"

**Yes — and here's exactly where.**

By default, everything lives in **a single SQLite database file** at
the path you passed to `db_path`.

```python
memory = MATHIRMemory(embedding_dim=384, db_path="my_memory.db")
```

The file `my_memory.db` is now on disk. Open it with anything:

```bash
# Command line
sqlite3 my_memory.db "SELECT memory_id, modality, tier, timestamp FROM memories LIMIT 5;"

# Or just
sqlite3 my_memory.db
sqlite> .schema
sqlite> SELECT * FROM memories WHERE modality = 'text';
```

Or open it in [DB Browser for SQLite](https://sqlitebrowser.org/),
DataGrip, TablePlus, or any tool that speaks SQLite.

### Schema (one table, two indices, one FTS5 index)

```sql
CREATE TABLE memories (
    memory_id     TEXT PRIMARY KEY,    -- e.g. "mem_a1b2c3d4"
    modality      TEXT NOT NULL,       -- 'text' | 'image' | 'audio' | 'video' | 'other'
    embedding     BLOB,                -- float32 tensor, header + raw bytes
    embedding_dim INTEGER,             -- for sanity checks on load
    metadata      TEXT,                -- user-provided dict, stored as JSON
    modality_text TEXT,                -- short text used for FTS5 search
    timestamp     REAL,                -- unix seconds
    tier          TEXT,                -- 'working' | 'episodic' | 'semantic' | 'immune'
    stability     REAL DEFAULT 1.0,    -- Ebbinghaus spaced-repetition weight
    recall_count  INTEGER DEFAULT 0    -- # of times this memory was recalled
);

CREATE INDEX idx_memories_modality  ON memories(modality);
CREATE INDEX idx_memories_timestamp ON memories(timestamp);
CREATE INDEX idx_memories_tier      ON memories(tier);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    memory_id UNINDEXED,
    modality_text,
    tokenize = 'porter unicode61'
);
```

So when you ask "where is my data?":

| What           | Where                                |
|----------------|--------------------------------------|
| Embeddings     | `memories.embedding` (BLOB, float32) |
| User metadata  | `memories.metadata` (JSON)           |
| Searchable text| `memories.modality_text` + FTS5      |
| Timestamps     | `memories.timestamp`                 |
| Tier routing   | `memories.tier`                      |
| Recency / use  | `memories.recall_count`, `stability` |
| Provider info  | `memories.provider`, `memories.model`|
| Multi-provider embeddings | `memory_embeddings` table |

> **Provider/Model Tracking:** Each memory stores its `provider` and `model`
> fields. Additional embeddings from other providers are stored in the
> `memory_embeddings` table with their own provider/model tracking for
> cross-provider recall.

### "I don't want a file"

Pass `db_path=None` (or `":memory:"` for an in-memory SQLite):

```python
memory = MATHIRMemory(embedding_dim=384, db_path=None)  # pure RAM
memory = MATHIRMemory(embedding_dim=384, db_path=":memory:")  # RAM + FTS5
```

---

## 📖 API Reference

### `MATHIRMemory(embedding_dim, config=None, db_path=None)`

The main class. Single `nn.Module` containing the 4 tiers + router.

| Argument       | Type             | Default        | Description                              |
|----------------|------------------|----------------|------------------------------------------|
| `embedding_dim`| `int`            | *required*     | Dimension of incoming embeddings         |
| `config`       | `dict`           | `None`         | Override defaults (use `configure({})`)  |
| `db_path`      | `str` \| `None`  | `"mathir.db"`  | SQLite path, or `None` for RAM-only      |

### `memory.perceive(embedding, metadata=None) → dict`

Run an embedding through the 4 tiers and the router. Returns:

```python
{
    "enhanced_embedding": torch.Tensor,  # [B, embedding_dim]
    "modality":          "text",         # inferred or from metadata
    "router_weights":    torch.Tensor,   # [B, 4] — working/episodic/semantic/immune
    "anomaly_score":     torch.Tensor,   # [B] — continuous immune score
    "memory_id":         None,           # (perceive doesn't store)
}
```

### `memory.store(embedding, metadata=None, tier="episodic") → str`

Persist an embedding. Returns the new `memory_id` (e.g. `"mem_a1b2c3d4"`).

```python
mid = memory.store(
    torch.randn(1, 384),
    {"text": "the user said hello", "user": "alice"},
)
```

### `memory.recall(query_embedding, k=5, modality=None) → list[dict]`

Top-k most similar memories by cosine similarity. Each dict has:

```python
{
    "memory_id":  "mem_xyz",
    "similarity": 0.847,
    "metadata":   {"text": "...", "user": "alice"},
    "embedding":  torch.Tensor,  # original 1-D embedding
    "tier":       "episodic",
    "modality":   "text",
}
```

### `memory.recall_text(query_text, k=5, modality=None) → list[dict]`

BM25 text search via SQLite FTS5. Same return shape as `recall()`.

```python
hits = memory.recall_text("transformer attention", k=5)
```

### `memory.forget(threshold=0.1) → int`

Prune low-utility episodic memories. Returns the number dropped.

### `memory.get_stats() → dict`

```python
{
    "embedding_dim":  384,
    "internal_dim":   272,
    "tier_working":   {"usage": 12, "capacity": 64},
    "tier_episodic":  {"usage": 100, "capacity": 1000},
    "tier_semantic":  {"num_prototypes": 256, "used_prototypes": 89, ...},
    "tier_immune":    {"usage": 50, "capacity": 100, "threshold": 2.0},
    "router":         {"type": "kl_constrained", "kl_coefficient": 0.01},
    "storage":        {"type": "sqlite", "db_path": "mathir.db", "row_count": 100},
}
```

### `memory.save()` / `memory.load()`

Manual flush / rehydrate. `auto_save=True` (the default) makes these
unnecessary for normal use, but they are useful for batch workloads or
checkpointing.

### `memory.reset()`

Wipe every tier in memory. **Does not touch SQLite.** If you want to
wipe SQLite too:

```python
memory._store.drop_all()
```

---

## ⚙️ Configuration

The default config is:

```python
DEFAULT_CONFIG = {
    "memory": {
        "embedding_dim": 384,
        "internal_dim": 272,
        "working_capacity": 64,
        "episodic_capacity": 1000,
        "semantic_prototypes": 256,
        "immunological_capacity": 100,
        "kl_coefficient": 0.01,
        "anomaly_threshold": 2.0,
        "decay_rate": 0.95,
    },
    "router": {
        "type": "kl_constrained",   # or "uniform"
        "kl_coefficient": 0.01,
        "hidden_dim": 128,
    },
    "storage": {
        "type": "sqlite",           # or "memory"
        "db_path": "mathir.db",
        "auto_save": True,
    },
    "perception": {
        "use_residual": True,
        "use_layer_norm": True,
    },
}
```

Override what you need:

```python
from mathir_dropin import configure, MATHIRMemory

cfg = configure({
    "memory": {
        "embedding_dim": 768,           # I'm using BERT-base
        "episodic_capacity": 5000,      # I have a lot of memory
    },
    "storage": {
        "db_path": "/var/data/agent.db",
    },
})

memory = MATHIRMemory(embedding_dim=768, config=cfg)
```

---

## 🧪 Tests

```bash
cd mathir_dropin
python -m pytest tests/ -v
```

10 tests covering: init defaults, init with DB, store+recall, modality
filter, persistence, dimension errors, forget, stats, concurrent
stores, FTS5 text search.

---

## 🔄 Upgrading from V6

The full `mathir_lib` V6 stored memories in PyTorch buffers with custom
serialization. The drop-in uses plain SQLite instead, which is:

1. **Inspectable** — `sqlite3 mathir.db` and you see your data.
2. **Portable** — one file you can `scp`, `cp`, or attach to an email.
3. **Queryable** — use any SQL tool, BI dashboard, or notebook.

If you have V6 `.pt` checkpoints you want to migrate, the on-disk
shape is just:

```python
# V6: state_dict()
v6_state = torch.load("agent_v6.pt")

# Drop-in: import + reinsert
from mathir_dropin import MATHIRMemory
mem = MATHIRMemory(embedding_dim=v6_state["embedding_dim"], db_path="agent_v7.db")
for embedding, metadata in v6_state["episodes"]:
    mem.store(embedding, metadata)
```

---

## 🆚 Drop-in vs. Full Library

| Feature                     | Drop-in (this)         | Full `mathir_lib`      |
|-----------------------------|------------------------|------------------------|
| 4-tier memory               | ✅                      | ✅                      |
| KL-constrained router       | ✅                      | ✅                      |
| SQLite + FTS5 persistence   | ✅                      | ❌ (custom)             |
| Ebbinghaus spaced repetition| ✅ (stability boost)   | ✅ (full curve + evict) |
| Variational memory tier     | ❌                      | ✅                      |
| Sparse coding tier          | ❌                      | ✅                      |
| Cross-attention tier        | ❌                      | ✅                      |
| Neural ODE tier             | ❌                      | ✅                      |
| Hyperbolic semantic         | ❌                      | ✅                      |
| InfoNCE self-supervised     | ❌                      | ✅                      |
| FAISS backend (>1M items)   | ❌                      | ✅                      |
| Hybrid BM25 + CE retrieval  | ❌                      | ✅                      |
| Manifold-Constrained Hyper  | ❌                      | ✅                      |
| TurboQuant compression      | ❌                      | ✅                      |
| ONNX export                 | ❌                      | ✅                      |
| **Lines of code**           | **~1500**              | **5000+**              |
| **Files**                   | **7**                  | **18+**                |

**Rule of thumb:**

- 🚀 **Shipping a feature?** Use this drop-in.
- 🔬 **Researching / ablations?** Use `mathir_lib`.

---

## 📜 License

Same as the parent MATHIR project (see `../LICENSE`).

---

## 🐛 Troubleshooting

### "DimensionMismatchError"

You're passing a vector of the wrong size. Check:

```python
print(memory.embedding_dim)        # the configured size
print(your_embedding.shape[-1])    # what you're feeding in
```

### "Database is locked"

Another process has the file open. Either close it, or use
`db_path=":memory:"` for tests, or use a unique path per process.

### "FTS5: no such module"

Your SQLite was compiled without FTS5. Very rare on modern systems,
but if it happens, you can still use `recall()` (embedding search) —
only `recall_text()` requires FTS5.

### "I want to do <X> that the drop-in doesn't support"

Use the full `mathir_lib` package. The drop-in is intentionally a
strict subset. If you think `<X>` should be in the drop-in, open an
issue.

# MATHIR — How to Ship (Production Deployment Guide)

**Stop being confused. Here's the answer.**

---

## ❓ "I have 100+ files. What do I actually ship?"

**Answer: 1 folder (`mathir_dropin/`). That's it.**

```
your-app/
├── your_app.py            # your code
├── requirements.txt        # just torch + numpy (already in your stack)
├── mathir.db              # SQLite database (auto-created)
└── mathir_dropin/         # ← copy this single folder from MATHIR
    ├── __init__.py         # exports: MATHIRMemory
    ├── memory.py           # the main class (~400 lines)
    ├── store.py            # SQLite persistence (~300 lines)
    ├── config.py           # defaults (~150 lines)
    ├── exceptions.py       # errors
    └── README.md           # 5-min quickstart
```

**5 files, ~1500 lines total.** No need to navigate 18+ files in `mathir_lib/`.

---

## ❓ "What format is the data?"

**Answer: SQLite database file.** You can open it with any SQLite browser.

```bash
$ sqlite3 my_memory.db
SQLite version 3.40.0
Enter ".help" for usage hints.

sqlite> .schema
CREATE TABLE memories (
    memory_id TEXT PRIMARY KEY,    -- e.g., "mem_abc123"
    modality TEXT,                 -- 'text' | 'image' | 'audio' | 'video'
    embedding BLOB,                -- serialized torch.Tensor
    embedding_dim INTEGER,
    metadata JSON,                 -- your custom dict
    modality_text TEXT,            -- for text search (BM25/FTS5)
    timestamp TIMESTAMP,
    tier TEXT,                     -- 'working' | 'episodic' | 'semantic' | 'immune'
    stability REAL DEFAULT 1.0,
    recall_count INTEGER DEFAULT 0
);

sqlite> SELECT modality, count(*) FROM memories GROUP BY modality;
text|143
image|27
audio|8

sqlite> SELECT memory_id, modality, metadata FROM memories LIMIT 3;
mem_001|text|{"user": "alice", "turn": 5}
mem_002|image|{"path": "cat.jpg"}
mem_003|audio|{"duration_s": 3.2}
```

**You can:**
- Open with DB Browser for SQLite (GUI)
- Query with SQL
- Backup by copying one file
- Migrate with standard SQLite tools

---

## ❓ "How do I integrate in 3 lines of code?"

**Answer:**

```python
from mathir_dropin import MATHIRMemory
import torch

# Line 1: Create memory
memory = MATHIRMemory(embedding_dim=384, db_path="my_memory.db")

# Line 2: Store
memory.store(torch.randn(1, 384), metadata={"user": "alice"})

# Line 3: Recall
results = memory.recall(torch.randn(1, 384), k=5)
```

**That's it.** No setup, no config files (unless you want them).

---

## ❓ "What's the difference between this and `mathir_lib`?"

| | `mathir_dropin/` (new) | `mathir_lib/` (research) |
|---|---|---|
| **Files** | 5 | 18+ |
| **Lines of code** | ~1500 | ~10,000+ |
| **Purpose** | Production use | Research / doctoral work |
| **V7 features** | Essential 4-tier | All 8 algorithms + 6 theorems |
| **Storage** | SQLite (default) | In-memory only |
| **Persistence** | ✅ Yes | ❌ No |
| **Multimodal** | ✅ Yes | ✅ Yes |
| **Tests** | 10 critical | 130+ comprehensive |
| **Use when** | Shipping to prod | Research / paper writing |

**Rule of thumb**: Use `mathir_dropin/` for production. Use `mathir_lib/` only if you're writing a paper.

---

## 📦 Step-by-Step: How to Integrate in 5 minutes

### Step 1: Copy the package
```bash
# From your app directory
cp -r /path/to/MATHIR/mathir_dropin ./
```

### Step 2: Install dependencies
```bash
pip install torch numpy  # you probably already have these
```

### Step 3: Use it
```python
# your_app.py
import torch
from mathir_dropin import MATHIRMemory

class YourChatBot:
    def __init__(self):
        # One line: persistent memory
        self.memory = MATHIRMemory(embedding_dim=1536, db_path="chat_memory.db")

    def remember(self, user_id: str, message: str, embedding: torch.Tensor):
        self.memory.store(
            embedding,
            metadata={"user": user_id, "message": message, "ts": time.time()}
        )

    def recall(self, query_embedding: torch.Tensor, user_id: str = None, k: int = 5):
        results = self.memory.recall(query_embedding, k=k)
        if user_id:
            results = [r for r in results if r["metadata"].get("user") == user_id]
        return results
```

### Step 4: Inspect your data
```bash
# Use any SQLite browser to see your memories
# CLI:
sqlite3 chat_memory.db "SELECT * FROM memories LIMIT 10"
# GUI: download "DB Browser for SQLite"
```

---

## 🔍 Storage Format Details

### Default: SQLite (recommended for production)

```sql
-- One table, indexed on common queries
CREATE TABLE memories (
    memory_id TEXT PRIMARY KEY,    -- unique ID (UUID-style)
    modality TEXT,                 -- 'text' | 'image' | 'audio' | 'video' | 'multimodal'
    embedding BLOB,                -- pickle/numpy/torch tensor serialization
    embedding_dim INTEGER,          -- 384 | 512 | 1536 | 4096
    metadata JSON,                 -- your custom data
    modality_text TEXT,            -- text version for hybrid search (BM25)
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tier TEXT DEFAULT 'episodic',  -- 'working' | 'episodic' | 'semantic' | 'immune'
    stability REAL DEFAULT 1.0,     -- Ebbinghaus (V7)
    recall_count INTEGER DEFAULT 0
);

CREATE INDEX idx_modality ON memories(modality);
CREATE INDEX idx_timestamp ON memories(timestamp DESC);
CREATE INDEX idx_tier ON memories(tier);
CREATE INDEX idx_text ON memories(modality_text);  -- For LIKE queries

-- Optional: FTS5 for full-text search
CREATE VIRTUAL TABLE memories_fts USING fts5(
    modality_text, metadata, content='memories', content_rowid='rowid'
);
```

### Optional: In-Memory (for tests / short-lived)

```python
memory = MATHIRMemory(embedding_dim=384)  # No db_path = in-memory
# Faster, but no persistence
```

### Optional: Other backends (advanced)

The dropin package can be extended to:
- **Redis**: for distributed memory across servers
- **PostgreSQL**: for large-scale production (with `pgvector` extension)
- **MongoDB**: for document-style memory

But SQLite is the default and covers 95% of use cases.

---

## 🔄 How to Migrate from V6 / VectorDB

### From a Vector Database (FAISS, Pinecone, etc.)

```python
# BEFORE (Pinecone)
import pinecone
pinecone.init(api_key="...")
index = pinecone.Index("my-memories")
index.upsert(vectors=[(id, embedding, metadata)])
results = index.query(vector=query, top_k=5)

# AFTER (MATHIR dropin)
from mathir_dropin import MATHIRMemory
memory = MATHIRMemory(embedding_dim=384, db_path="my_memory.db")
memory.store(torch.from_numpy(embedding), metadata=metadata)
results = memory.recall(torch.from_numpy(query), k=5)
```

**What you GAIN:**
- Online learning (memory adapts as you use it)
- Anomaly detection (NP-optimal Mahalanobis)
- Hierarchical memory (4 temporal tiers)
- Spaced repetition forgetting (Ebbinghaus)
- Hybrid retrieval (BM25 + dense + cross-encoder)

**What you LOSE (vs pure vector DB):**
- ~1ms additional latency for hybrid retrieval (vs 0.05ms for FAISS)
- Slightly more complex setup

### From MATHIR V6 (`mathir_lib`)

```python
# BEFORE (V6)
from mathir_lib import MATHIRPlugin
plugin = MATHIRPlugin(embedding_dim=4096)
out = plugin.perceive(emb)
plugin.store({"embedding": emb, "user": "alice"})

# AFTER (V7.2 dropin)
from mathir_dropin import MATHIRMemory
memory = MATHIRMemory(embedding_dim=4096, db_path="memory.db")
out = memory.perceive(emb, metadata={"user": "alice"})
memory_id = memory.store(emb, metadata={"user": "alice"})  # returns ID
```

---

## 🏗️ Architecture (What Goes Where)

```
┌─────────────────────────────────────────────────────┐
│                 YOUR APPLICATION                     │
│                                                       │
│   ┌─────────────┐         ┌──────────────────────┐  │
│   │  Your LLM   │ ←─────→ │   mathir_dropin/      │  │
│   │ (GPT-4,     │         │   ┌──────────────┐    │  │
│   │  Claude,    │         │   │ MATHIRMemory │    │  │
│   │  Qwen, etc) │         │   │  (memory.py) │    │  │
│   └─────────────┘         │   └──────┬───────┘    │  │
│                           │          │             │  │
│                           │   ┌──────▼───────┐    │  │
│                           │   │  SQLiteStore  │    │  │
│                           │   │  (store.py)   │    │  │
│                           │   └──────┬───────┘    │  │
│                           └──────────┼─────────────┘  │
│                                      │              │
│                                      ▼              │
│                              ┌──────────────┐       │
│                              │  my_memory.db │       │
│                              │  (SQLite)      │       │
│                              └──────────────┘       │
└─────────────────────────────────────────────────────┘
```

**You only need to know about `MATHIRMemory`.** The rest is internal.

---

## ❓ FAQ

### Q: Can I see my data?
**A:** YES. `sqlite3 my_memory.db` opens the database. Use DB Browser for SQLite (free GUI).

### Q: Can I back up my data?
**A:** YES. Just copy the `.db` file. SQLite is a single-file database.

### Q: Can I migrate my data?
**A:** YES. SQLite has standard export/import tools. Or write a Python script that reads from the source and inserts via `memory.store()`.

### Q: What if my embeddings change dimension?
**A:** Create a new `MATHIRMemory` with the new dim. Old data is in the old DB. You can manually migrate or run two in parallel.

### Q: Can I scale beyond SQLite?
**A:** YES. The dropin is designed to be backend-agnostic. Replace `store.py` with a Redis/Postgres/MongoDB version. The `memory.py` interface stays the same.

### Q: Is this thread-safe?
**A:** YES. SQLite supports concurrent reads. Writes are serialized but fast (<1ms).

### Q: Is this async-compatible?
**A:** YES. The `store()` and `recall()` methods are async-friendly. Wrap in `asyncio.to_thread()` if you need non-blocking I/O.

### Q: How do I deploy to Docker?
**A:** Add `mathir_dropin/` to your Docker image. Mount the SQLite file as a volume for persistence:
```dockerfile
COPY mathir_dropin /app/mathir_dropin
VOLUME /data
CMD python your_app.py  # writes to /data/my_memory.db
```

### Q: How do I backup in production?
**A:** Use SQLite's online backup API, or just `cp my_memory.db backup-$(date).db` periodically. SQLite is designed for this.

---

## 🎯 The Take-Home

1. **One folder** (`mathir_dropin/`) is all you need
2. **One file** (`my_memory.db`) is your data
3. **One class** (`MATHIRMemory`) is your API
4. **Three lines** of code is your integration

**You can stop being confused. Just copy the folder, use the class, ship the .db file.**

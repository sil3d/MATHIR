# Daemon Architecture (v8.5.0)

## What the Daemon Does

The MATHIR daemon is a **persistent background process** that:

1. Loads the embedding model once at startup (SentenceTransformer + CUDA)
2. Keeps the model in RAM/VRAM for instant access
3. Serves requests via TCP socket (JSON-RPC)
4. Manages the SQLite database with vec0 vector index
5. Handles 5-tier cognitive memory routing

Without daemon: each embedding request loads the model (~2-5s)
With daemon: model stays loaded, requests complete in ~20ms

## Protocol

**Transport**: TCP socket on `127.0.0.1:7338`
**Format**: JSON-RPC 2.0 (newline-delimited)

### Request Format

```json
{"jsonrpc": "2.0", "method": "memory_save", "params": {...}, "id": 1}
```

### Response Format

```json
{"jsonrpc": "2.0", "result": {...}, "id": 1}
```

### Error Format

```json
{"jsonrpc": "2.0", "error": {"code": -1, "message": "error description"}, "id": 1}
```

## Methods

### `ping`

Check if daemon is alive and model is loaded.

**Params**: None
**Returns**:
```json
{
  "status": "ok",
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "dims": 384,
  "uptime_seconds": 3600
}
```

### `memory_save`

Save a memory block with embedding.

**Params**:
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| content | string | yes | Text content to store |
| agent | string | yes | Agent identifier |
| block_type | string | yes | working_memory / episodic / semantic / procedural / immunological |
| label | string | yes | Unique label for this memory |
| priority | int | no | 1-10, default 5 |
| metadata | dict | no | Additional key-value pairs |

**Returns**:
```json
{
  "memory_id": "42",
  "dims": 1024,
  "tier": "semantic"
}
```

### `memory_recall`

Search memories by semantic similarity.

**Params**:
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | yes | Search query text |
| agent | string | no | Filter by agent |
| k | int | no | Number of results (default 5) |
| block_type | string | no | Filter by memory type |

**Returns**:
```json
{
  "results": [
    {
      "memory_id": "42",
      "content": "This project uses React + TypeScript",
      "label": "tech-stack",
      "score": 0.89,
      "block_type": "semantic"
    }
  ],
  "query_time_ms": 27
}
```

### `memory_stats`

Get memory statistics.

**Params**: None
**Returns**:
```json
{
  "total_memories": 1247,
  "by_tier": {
    "working_memory": 23,
    "episodic": 456,
    "semantic": 678,
    "procedural": 90,
    "immunological": 12
  },
  "by_agent": {
    "coder": 512,
    "debugger": 389,
    "swarm": 346
  },
  "db_size_mb": 45.2
}
```

### `memory_delete`

Soft-delete (archive) a memory.

**Params**:
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| memory_id | string | yes | ID of memory to delete |
| reason | string | yes | Why it's being deleted |

**Returns**:
```json
{"deleted": true, "memory_id": "42"}
```

### `memory_hybrid_search`

Hybrid search: vector cosine + BM25 lexical + RRF fusion. Better than either alone.

**Params**:
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | yes | Search query text |
| k | int | no | Number of results (default 5, max 100) |
| agent | string | no | Filter by agent |
| vector_weight | float | no | Vector result weight for RRF (default 1.0) |
| bm25_weight | float | no | BM25 result weight for RRF (default 1.0) |

**Returns**:
```json
{
  "results": [
    {
      "memory_id": "mem_abc123",
      "content": "Fixed auth bug: token refresh was failing",
      "rrf_score": 0.0318,
      "score": 0.0318,
      "agent": "coder",
      "tier": "episodic",
      "created_at": "2026-06-19T08:30:00"
    }
  ],
  "query": "auth bug fix",
  "total": 5,
  "project": "myproject",
  "mode": "hybrid",
  "vector_hits": 15,
  "bm25_hits": 15
}
```

**How it works**:
1. Creates its own SQLite connection (thread-safe, `check_same_thread=False`)
2. Detects schema automatically (old: `modality_text`, new: `content`)
3. Vector search via sqlite-vec (cosine distance)
4. BM25 lexical search via `rank_bm25` (token frequency)
5. RRF fusion combines both ranked lists (k=60)

### `memory_push`

Proactive memory delivery — daemon analyzes context and returns relevant memories.

**Params**:
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| context | string | yes | Context text to analyze |
| k | int | no | Max memories to return (default 10) |
| agent | string | no | Filter by agent |

**Returns**:
```json
{
  "memories": [
    {
      "memory_id": "mem_abc123",
      "content": "This project uses React + TypeScript",
      "score": 0.89,
      "agent": "coder",
      "label": "tech-stack"
    }
  ],
  "queries_used": ["react typescript project"],
  "total": 1,
  "cached": false
}
```

### `memory_risk_check`

Check memory content for privacy risks (PersistBench findings: 53% leakage, >90% sycophancy).

**Params**:
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| content | string | yes | Content to check |

**Returns**:
```json
{
  "domain": "medical",
  "leakage_risk": 0.9,
  "sycophancy_risk": 0.1,
  "sensitivity": "high",
  "reasons": ["SSN detected"],
  "safe_to_store": false
}
```

## Threading Model

The daemon uses **one thread per connection** with these safety measures:

1. **VecMemory cache** (`_vec_cache`): Shared instances with `check_same_thread=False` for non-hybrid handlers
2. **Hybrid handler**: Creates its own SQLite connection per request (avoids cross-thread lock contention)
3. **Push cache**: Thread-safe LRU with `threading.Lock`
4. **Connection limit**: Max 50 concurrent connections (prevents thread exhaustion DoS)
5. **Client timeout**: 30s idle timeout per connection

```
Main thread: accept() → spawn handler thread
Handler thread: recv() → process → send() → loop (while True)
```

**Key safety rules**:
- All `VecMemory` instances use `check_same_thread=False`
- `HybridSearch` uses `threading.RLock` on all mutating operations
- Push cache uses `threading.Lock` for shared state
- SQLite WAL mode enables concurrent reads + serialized writes

## Client Usage

### mathir_client.py

```bash
# Check daemon
python -m mathir_lib.mathir_client ping

# Save memory
python -m mathir_lib.mathir_client save "Bug fix: auth token refresh" \
  --agent coder --type episodic --label auth-fix --priority 8

# Recall memories
python -m mathir_lib.mathir_client recall "auth bug" --agent coder --k 5

# Stats
python -m mathir_lib.mathir_client stats

# Delete
python -m mathir_lib.mathir_client delete 42 --reason "outdated"
```

### Python Client

```python
import socket
import json

class MathirClient:
    def __init__(self, host="127.0.0.1", port=7338):
        self.addr = (host, port)
    
    def _call(self, method, params=None):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(self.addr)
            request = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1}
            s.sendall(json.dumps(request).encode() + b"\n")
            response = s.recv(65536).decode()
            return json.loads(response)
    
    def ping(self):
        return self._call("ping")
    
    def save(self, content, agent, block_type, label, priority=5, metadata=None):
        return self._call("memory_save", {
            "content": content,
            "agent": agent,
            "block_type": block_type,
            "label": label,
            "priority": priority,
            "metadata": metadata
        })
    
    def recall(self, query, agent=None, k=5, block_type=None):
        return self._call("memory_recall", {
            "query": query,
            "agent": agent,
            "k": k,
            "block_type": block_type
        })
    
    def stats(self):
        return self._call("memory_stats")

# Usage
client = MathirClient()
client.ping()
client.save("React project with Zustand", "coder", "semantic", "tech-stack", 7)
results = client.recall("what state management", agent="coder", k=3)
```

## Starting the Daemon

```bash
# Basic start
python -m mathir_mcp

# With specific model
python -m mathir_mcp --model nomic-embed-text-v1.5

# With GPU
python -m mathir_mcp --use-gpu

# Background (Linux/Mac)
python -m mathir_mcp &

# Background (Windows)
Start-Process python -ArgumentList "-m mathir_mcp" -WindowStyle Hidden
```

## Testing

```bash
# Quick test
python -m mathir_lib.mathir_client ping

# Full test
python -m mathir_lib.mathir_client save "test memory" --agent test --type semantic --label test-1
python -m mathir_lib.mathir_client recall "test" --agent test --k 1
python -m mathir_lib.mathir_client stats
python -m mathir_lib.mathir_client delete <id> --reason "test cleanup"
```

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Connection refused | Daemon not running | `python -m mathir_mcp` |
| Timeout | Model loading (first start) | Wait 2-5s, check `nvidia-smi` |
| OOM | Model too large for RAM/VRAM | Use smaller model or CPU |
| Slow first request | Cold model load (first start only) | Normal, subsequent requests are fast |
| Port in use | Another daemon running | Kill existing or use `--port 7339` |

# Daemon Architecture

## What the Daemon Does

The MATHIR daemon is a **persistent background process** that:

1. Loads the ONNX embedding model once at startup
2. Keeps the model in RAM/VRAM for instant access
3. Serves requests via TCP socket (JSON-RPC)
4. Manages the SQLite database with vec0 vector index
5. Handles 4-tier cognitive memory routing

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
  "model": "nomic-embed-text-v1.5",
  "dims": 768,
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
| block_type | string | yes | working_memory / episodic / semantic / procedural |
| label | string | yes | Unique label for this memory |
| priority | int | no | 1-10, default 5 |
| metadata | dict | no | Additional key-value pairs |

**Returns**:
```json
{
  "memory_id": "42",
  "dims": 768,
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
    "procedural": 90
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

## Client Usage

### mathir_client.py

```bash
# Check daemon
python bin/mathir_client.py ping

# Save memory
python bin/mathir_client.py save "Bug fix: auth token refresh" \
  --agent coder --type episodic --label auth-fix --priority 8

# Recall memories
python bin/mathir_client.py recall "auth bug" --agent coder --k 5

# Stats
python bin/mathir_client.py stats

# Delete
python bin/mathir_client.py delete 42 --reason "outdated"
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
python bin/mathir_daemon.py

# With specific model
python bin/mathir_daemon.py --model nomic-embed-text-v1.5

# With GPU
python bin/mathir_daemon.py --use-gpu

# Background (Linux/Mac)
python bin/mathir_daemon.py &

# Background (Windows)
Start-Process python -ArgumentList "bin/mathir_daemon.py" -WindowStyle Hidden
```

## Testing

```bash
# Quick test
python bin/mathir_client.py ping

# Full test
python bin/mathir_client.py save "test memory" --agent test --type semantic --label test-1
python bin/mathir_client.py recall "test" --agent test --k 1
python bin/mathir_client.py stats
python bin/mathir_client.py delete <id> --reason "test cleanup"
```

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Connection refused | Daemon not running | `python bin/mathir_daemon.py` |
| Timeout | Model loading (first start) | Wait 2-5s, check `nvidia-smi` |
| OOM | Model too large for RAM/VRAM | Use smaller model or CPU |
| Slow first request | Cold ONNX session | Normal, subsequent requests are fast |
| Port in use | Another daemon running | Kill existing or use `--port 7339` |

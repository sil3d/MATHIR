# MATHIR — How to Ship (v8.9.0)

**Production deployment guide. One daemon, any agent.**

---

## Architecture

```
Your Agent (Claude Code / Cursor / Windsurf / custom)
    │
    │ MCP protocol (HTTP)
    ▼
┌──────────────────────────────┐
│   MATHIR Daemon (port 7338)  │
│   ┌────────────────────────┐ │
│   │ 3-layer auto-cache     │ │
│   │ L1: Embedding LRU      │ │
│   │ L2: Recall TTL 60s     │ │
│   │ L3: Session pre-warm   │ │
│   └────────────────────────┘ │
│   ┌──────────┐  ┌─────────┐ │
│   │ Embedder │  │ SQLite  │ │
│   │ (cached) │  │ + vec   │ │
│   └──────────┘  └─────────┘ │
└──────────────────────────────┘
    │
    ▼
~/.config/MATHIR/data/projects/<project>/mathir.db
```

**One daemon = all agents share the same memory.**

---

## Option A: MCP Integration (Recommended)

### Step 1: Install

```bash
pip install -e /path/to/MATHIR/mathir_mcp
# or from GitHub:
pip install git+https://github.com/sil3d/MATHIR.git#subdirectory=mathir_mcp
```

### Step 2: Start the daemon

```bash
mathir-server &   # port 7338
```

### Step 3: Add to your agent's MCP config

```json
{
  "mcpServers": {
    "mathir": {
      "command": "mathir-mcp"
    }
  }
}
```

### Step 4: Done

The agent now has 23 memory tools. Memory is per-project (auto-routed by CWD). The 3-layer cache gives 18x speedup on repeated queries.

---

## Option B: HTTP API (Custom agents)

If your agent doesn't support MCP, use the HTTP API directly:

```python
import requests

DAEMON = "http://127.0.0.1:7338"

# Save
requests.post(f"{DAEMON}/api/memory/save", json={
    "content": "User prefers dark mode",
    "agent": "my-app",
    "block_type": "episodic",
    "label": "user-pref",
    "priority": 5,
    "project": "my-project"
})

# Recall
results = requests.post(f"{DAEMON}/api/memory/recall", json={
    "query": "user preferences",
    "k": 5,
    "project": "my-project"
}).json()

# Context (auto-loads relevant memories for a task)
context = requests.post(f"{DAEMON}/api/context", json={
    "task": "Fix auth bug in login.py",
    "project": "my-project"
}).json()
```

---

## Option C: Legacy `mathir_dropin/` (Deprecated)

The old `mathir_dropin/` package (embedded library) still works but is **not recommended** for new projects. The daemon architecture is better because:

| | Daemon (v8.9.0) | mathir_dropin (legacy) |
|---|---|---|
| **Shared across agents** | Yes (all agents → same daemon) | No (each process has its own DB) |
| **Auto-cache** | Yes (18x speedup) | No |
| **INT8 quantization** | Yes (4x compression) | No |
| **Cross-encoder reranking** | Yes (+20pp) | No |
| **Per-project routing** | Yes (auto by CWD) | Manual |
| **Auto-start on boot** | Yes (systemd/launchd/Startup) | No |
| **Setup** | `mathir-server &` | Copy folder + code changes |

---

## Production Checklist

- [ ] Daemon running on port 7338
- [ ] Auto-start configured (boot persistence)
- [ ] Agent MCP config points to `mathir-mcp`
- [ ] Per-project DB routing working (check `GET /api/stats`)
- [ ] Cache warming: first query is cold (~37ms), subsequent queries are fast (<1ms)

### Monitoring

```bash
# Health
curl http://127.0.0.1:7338/health

# Cache performance
curl http://127.0.0.1:7338/api/cache/stats

# Memory stats
curl http://127.0.0.1:7338/api/stats
```

### Backup

The database is a single SQLite file:
```bash
cp ~/.config/MATHIR/data/projects/<project>/mathir.db backup-$(date +%Y%m%d).db
```

---

## Docker Deployment

```dockerfile
FROM python:3.11-slim
RUN pip install git+https://github.com/sil3d/MATHIR.git#subdirectory=mathir_mcp
EXPOSE 7338
CMD ["mathir-server", "--host", "0.0.0.0"]
```

```bash
docker run -d -p 7338:7338 -v mathir-data:/root/.config/MATHIR mathir:latest
```

---

## FAQ

### Q: Can I run multiple daemons?
**A:** Yes, on different ports: `mathir-server --port 7339`. Each is independent.

### Q: What if the daemon crashes?
**A:** The MCP bridge returns an error. The agent continues without memory. Restart the daemon — all data is persisted in SQLite.

### Q: How much disk space?
**A:** ~80MB for the embedding model + ~1KB per 1000 memories (with INT8 quantization).

### Q: Thread-safe?
**A:** Yes. Flask + Waitress handles concurrent requests. SQLite uses WAL mode for parallel reads.

### Q: Can I use a custom embedding model?
**A:** Yes. Set `MATHIR_EMBEDDING_MODEL` env var before starting the daemon.

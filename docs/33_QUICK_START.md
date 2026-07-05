# MATHIR — Quick Start Guide (v8.7.0)

**Get MATHIR running in 3 minutes.**

---

## Prerequisites

- Python 3.10+
- pip

---

## 1. Install (one time)

```bash
git clone https://github.com/sil3d/MATHIR.git
cd MATHIR/mathir_mcp
pip install -e .
```

---

## 2. Start the daemon

```bash
mathir-server
```

The daemon starts on `http://127.0.0.1:7338`. The embedder model loads on first request (~2-5s cold, <1ms warm).

---

## 3. Add to your agent

### Claude Code
Add to `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "mathir": {
      "command": "mathir-mcp"
    }
  }
}
```

### OpenCode / MiMoCode
The plugin is auto-configured. Just start the daemon — the auto-inject plugin picks it up.

### Any other agent (Windsurf, Cursor, Cline, etc.)
Add the MCP server to your agent's config. The command is `mathir-mcp`.

---

## 4. Verify it works

```bash
# Health check
curl http://127.0.0.1:7338/health

# Save a memory
curl -X POST http://127.0.0.1:7338/api/memory/save \
  -H "Content-Type: application/json" \
  -d '{"content": "MATHIR is working!", "agent": "test", "block_type": "episodic", "label": "test"}'

# Recall
curl -X POST http://127.0.0.1:7338/api/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "MATHIR", "k": 3}'

# Cache stats
curl http://127.0.0.1:7338/api/cache/stats
```

---

## 5. Use from code (optional)

```python
import requests

DAEMON = "http://127.0.0.1:7338"

# Save
requests.post(f"{DAEMON}/api/memory/save", json={
    "content": "User prefers dark mode",
    "agent": "my-app",
    "block_type": "episodic",
    "label": "user-pref",
    "priority": 5
})

# Recall
results = requests.post(f"{DAEMON}/api/memory/recall", json={
    "query": "user preferences",
    "k": 5
}).json()

for mem in results["memories"]:
    print(f"[{mem['score']:.2f}] {mem['content'][:80]}")
```

---

## 6. Auto-start on boot (optional)

```bash
# Windows
# Put a shortcut to mathir_mcp/bin/auto_start.bat in shell:startup

# Linux
systemctl --user enable mathir-daemon

# macOS
launchctl load -w ~/Library/LaunchAgents/com.mathir.daemon.plist
```

---

## What you get

| Feature | Description |
|---------|-------------|
| 26 MCP tools | Save, recall, search, link, decay, promote, consolidate, god mode |
| 3-layer cache | 18x speedup on repeated queries (zero config) |
| INT8 quantization | 4x embedding compression, zero recall loss |
| Cross-encoder reranking | +20pp quality on retrieval |
| Hybrid search | Vector + BM25 + RRF fusion |
| Per-project memory | Each project gets its own `.mathir/mathir.db` |
| Cross-agent sharing | Claude, MiMo, OpenCode share the same daemon |
| Online learning | Memory evolves as you use it |
| Anomaly detection | Mahalanobis distance (NP-optimal) |

---

## Troubleshooting

### "Port 7338 already in use"
Another MATHIR daemon is running. Kill it or use a different port:
```bash
mathir-server --port 7339
```

### "Model not found"
The first request downloads the embedding model (~80MB). Subsequent requests are instant.

### "Connection refused"
Make sure the daemon is running: `curl http://127.0.0.1:7338/health`

### Legacy training scripts (train.bat, dashboard.bat)
These were part of MATHIR v1-v5 (autonomous driving research). The current system (v8.7.0) is a daemon-based MCP server — no training scripts needed.

---

## Documentation

| Doc | What it covers |
|-----|---------------|
| `README.md` | Full feature list, architecture |
| `CHANGELOG.md` | Version history |
| `mcp_architecture.md` | Architecture diagram |
| `docs/DAEMON.md` | All daemon endpoints |
| `docs/REFERENCES.md` | Academic papers cited |
| `03_MASTER_QA_GUIDE.md` | Q&A for defense |
| `BRAIN_ARCHITECTURE.md` | Brain analogy deep-dive |

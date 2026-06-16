# MATHIR MCP Integration

## What is MCP?

Model Context Protocol (MCP) is an open standard for connecting LLMs to external tools and data sources. It defines a client-server architecture where:

- **MCP Server** — exposes tools/resources to LLMs
- **MCP Client** — queries servers (Claude Desktop, OpenCode, custom apps)

## MATHIR as an MCP Server

MATHIR provides a **4-tier cognitive memory system** as an MCP server:

| Tier | Purpose | Latency |
|------|---------|---------|
| Working | Multi-head attention, context-dependent | ~1ms |
| Episodic | Experience replay for future recall | ~3ms |
| Semantic | Project knowledge (stack, patterns) | ~2ms |
| Immunological | Anomaly detection (AUC=1.0) | ~1ms |

### Key Features

- **Embedding auto-detection**: vec0 table recreated on dimension change
- **Persistent daemon**: model stays loaded in RAM, no cold starts
- **SQLite-vec**: vector search without external databases
- **Daemon Push**: proactive memory delivery without explicit recall
- **4-tier cognitive memory** with KL-constrained router

## Quick Start

### 1. Install

```bash
cd /path/to/MATHIR
pip install -r requirements.txt
```

### 2. Start Daemon

```bash
python bin/mathir_daemon.py
# Daemon listening on TCP 127.0.0.1:7338
```

### 3. Test Connection

```bash
python bin/mathir_client.py ping
# Daemon: OK (uptime: ...)
```

### 4. Use via MCP Protocol

```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async with sse_client("http://127.0.0.1:7338/sse") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # Save a memory
        await session.call_tool("memory_save", {
            "content": "This project uses React + TypeScript",
            "agent": "coder",
            "block_type": "semantic",
            "label": "tech-stack",
            "priority": 7
        })
        
        # Recall relevant memories
        result = await session.call_tool("memory_recall", {
            "query": "what framework does this project use",
            "agent": "coder",
            "k": 5
        })
```

### 5. OpenCode Integration

Add to `~/.config/opencode/opencode.json`:

```json
{
  "mcpServers": {
    "mathir": {
      "url": "http://127.0.0.1:7338/sse"
    }
  }
}
```

## Architecture

```
┌─────────────┐     TCP/JSON-RPC      ┌──────────────┐
│ MCP Client  │ ◄───────────────────► │ MathirDaemon │
│ (LLM/App)  │                        │  Port 7338   │
└─────────────┘                        └──────┬───────┘
                                              │
                                     ┌────────▼────────┐
                                     │  ONNX Runtime   │
                                     │  (embedding)    │
                                     └────────┬────────┘
                                              │
                                     ┌────────▼────────┐
                                     │  SQLite + vec0  │
                                     │  mathir.db      │
                                     └─────────────────┘
```

## Documentation

- [DIMENSIONS.md](DIMENSIONS.md) — Embedding dimension guide
- [MODEL_COMPARISON.md](MODEL_COMPARISON.md) — Model benchmark table
- [GPU_SETUP.md](GPU_SETUP.md) — GPU acceleration setup
- [DAEMON.md](DAEMON.md) — Daemon protocol and architecture
- [INTEGRATION.md](INTEGRATION.md) — Platform integration guides

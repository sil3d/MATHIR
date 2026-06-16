# MATHIR — Agent Deployment Guide

**What to copy, what each file does, how to set up.**

---

## TL;DR

```
You need 3 things:
1. The daemon (mathir_daemon.py) — keeps model loaded in RAM
2. The client (mathir_client.py) — fast commands to save/recall
3. A config file (mathir.json) — model, dimensions, paths

Everything else is optional (dashboard, MCP server, push, etc.)
```

---

## Project Structure

```
MATHIR/
├── mcp/                          ← YOU ARE HERE (docs + dashboard)
│   ├── AGENT.md                  ← This file
│   ├── GLOBAL_INSTRUCTIONS.md    ← Copy into your agent's instructions
│   ├── README.md                 ← MCP integration overview
│   ├── DAEMON.md                 ← Daemon protocol docs
│   ├── DIMENSIONS.md             ← Embedding dimension guide
│   ├── GPU_SETUP.md              ← GPU acceleration setup
│   ├── INTEGRATION.md            ← Platform integration guides
│   ├── MODEL_COMPARISON.md       ← Model benchmark table
│   ├── DASHBOARD_GUIDE.md        ← Dashboard setup guide
│   ├── dashboard_server.py       ← Dashboard backend (standalone)
│   ├── dashboard.html            ← Dashboard frontend (Chart.js)
│   ├── mathir_push.py            ← Push module (proactive delivery)
│   └── test_daemon_push.py       ← Push tests
│
├── mathir_lib/                   ← Core library (optional, for advanced use)
│   ├── __init__.py               ← Package init
│   ├── config.py                 ← Configuration loader
│   ├── router.py                 ← 4-tier KL-constrained router
│   ├── compression.py            ← TurboQuant vector compression
│   ├── plugin.py                 ← Plugin system
│   ├── device_utils.py           ← Device detection (CPU/CUDA)
│   ├── hybrid_device.py          ← Hybrid CPU+GPU routing
│   └── auto_config_mathir_yaml.py← Auto-config generator
│
├── config/                       ← YAML configs
│   ├── default.yaml              ← Default config (bge-large 1024d)
│   ├── edge.yaml                 ← Edge device config (ONNX INT8)
│   ├── research.yaml             ← Research config (Qwen2.5-7B)
│   └── v7.yaml                   ← Legacy v7 config
│
├── mathir_search.py              ← HybridSearch (numpy + USearch)
├── mathir_vec.py                 ← VecMemory (sqlite-vec wrapper)
├── mathir_vec_optimized.py       ← Optimized VecMemory (7 optimizations)
├── mathir_gpu_vec.py             ← GPUVecMemory (torch GPU brute-force)
│
├── tests/                        ← Unit tests
├── benchmarks/                   ← Benchmark scripts
├── examples/                     ← Usage examples
├── docs/                         ← Documentation + paper
├── vision_testing/               ← Vision model tests (5 models)
├── stress_test/                  ← Stress tests
├── results/                      ← Benchmark results
├── _deprecated/                  ← Legacy code (gitignored)
│
├── requirements.txt              ← Python dependencies
├── setup.py                      ← Package setup
├── LICENSE                       ← MIT license
├── CHANGELOG.md                  ← Version history
└── README.md                     ← Project overview
```

---

## Files You MUST Copy

### 1. `mathir_daemon.py` — The Daemon (REQUIRED)

**What it does:**
- Loads the embedding model ONCE at startup (bge-large-en-v1.5, 1024d)
- Keeps model in VRAM/RAM for instant access
- Serves requests via TCP socket (port 7338)
- Manages SQLite database with vector search
- Handles 4-tier cognitive memory routing

**Without it:** Every save/recall loads the model (~3-5s cold start).
**With it:** Model stays loaded, requests complete in ~22ms.

**Where it lives:**
- OpenCode: `~/.config/opencode/bin/mathir_daemon.py`
- Standalone: `/path/to/MATHIR/bin/mathir_daemon.py`

**How to run:**
```bash
# Foreground (see logs)
python mathir_daemon.py

# Background (Linux/Mac)
python mathir_daemon.py &

# Background (Windows)
Start-Process python -ArgumentList "mathir_daemon.py" -WindowStyle Hidden

# Custom port
python mathir_daemon.py --port 8080

# Custom model
python mathir_daemon.py --model BAAI/bge-large-en-v1.5
```

---

### 2. `mathir_client.py` — The Client (REQUIRED)

**What it does:**
- Connects to daemon via TCP socket
- Provides fast commands: save, recall, search, stats, delete, push
- Model stays in daemon RAM — client sends only text

**Where it lives:**
- OpenCode: `~/.config/opencode/bin/mathir_client.py`
- Standalone: `/path/to/MATHIR/bin/mathir_client.py`

**How to use:**
```bash
# Check daemon is running
python mathir_client.py ping

# Save a memory
python mathir_client.py save "content here" \
  --agent coder --type semantic --label my-label --priority 7

# Recall memories
python mathir_client.py recall "query" --k 5

# Fast text search (no embedding)
python mathir_client.py search "query" --k 5

# Memory stats
python mathir_client.py stats

# Push proactive memories
python mathir_client.py push "context" --auto

# Delete a memory
python mathir_client.py delete 42 --reason "outdated"
```

---

### 3. `mathir.json` — Config File (REQUIRED)

**What it does:**
- Defines model, dimensions, device, paths
- Controls 4-tier memory capacities

**Where it lives:**
- OpenCode: `~/.config/opencode/config/mathir.json`
- Standalone: `/path/to/MATHIR/config/mathir.json`

**Default config:**
```json
{
  "model": "BAAI/bge-large-en-v1.5",
  "device": "cuda",
  "embedding_dim": 1024,
  "internal_dim": 512,
  "port": 7338,
  "db_path": ".mathir/mathir.db",
  "memory": {
    "working_capacity": 64,
    "episodic_capacity": 1000,
    "semantic_prototypes": 256,
    "immunological_capacity": 100
  }
}
```

---

## Files You SHOULD Copy (Recommended)

### 4. `mathir_push.py` — Push Module

**What it does:**
- Proactive memory delivery based on context
- Analyzes context → extracts queries → searches memory → returns ranked results
- Cache with TTL (300s) for repeated contexts

**Where it lives:**
- OpenCode: `~/.config/opencode/bin/mathir_push.py`
- MATHIR: `mcp/mathir_push.py`

**How to use:**
```bash
# Auto mode — ready-to-inject text
python mathir_client.py push "current task context" --auto

# JSON mode — structured
python mathir_client.py push "current task context" --json
```

---

### 5. `mathir_mcp_server.py` — MCP Server

**What it does:**
- Exposes MATHIR as an MCP server (6 tools)
- Works with OpenCode, OpenClaude, Kilo, MiMo, Claude Desktop

**Where it lives:**
- OpenCode: `~/.config/opencode/bin/mathir_mcp_server.py`

**Tools exposed:**
| Tool | Description |
|------|-------------|
| `memory_save` | Save a memory block |
| `memory_recall` | Search by similarity |
| `memory_smart_search` | Hybrid search (vector + text) |
| `memory_stats` | Get statistics |
| `memory_delete` | Soft-delete |
| `memory_push` | Proactive delivery |

---

### 6. `dashboard_server.py` — Dashboard Backend

**What it does:**
- Serves JSON API + HTML dashboard on port 7420
- Visualizes 4-tier memory, per-agent stats, timeline

**Where it lives:**
- MATHIR: `mcp/dashboard_server.py`
- OpenCode: `~/.config/opencode/bin/mathir_stats_server.py`

**How to run:**
```bash
python dashboard_server.py
# Open http://127.0.0.1:7420
```

---

### 7. `dashboard.html` — Dashboard Frontend

**What it does:**
- Single-file HTML dashboard (Chart.js)
- 4-tier breakdown, agent grid, timeline, memory table

**Where it lives:**
- MATHIR: `mcp/dashboard.html`
- OpenCode: `~/.config/opencode/bin/mathir_dashboard.html`

---

## Files You MIGHT Need (Advanced)

### 8. `mathir_search.py` — HybridSearch

**What it does:**
- Auto-switches from numpy brute-force (N < 5K) to USearch HNSW mmap (N >= 5K)
- Unified backend for vector search

**When to use:** If you want hybrid search instead of sqlite-vec.

---

### 9. `mathir_vec.py` — VecMemory

**What it does:**
- SQLite-vec wrapper with WAL optimization
- Vector search via `MATCH` syntax

**When to use:** Default vector backend (used by daemon).

---

### 10. `mathir_lib/` — Core Library

**What it does:**
- Config loader, router, compression, plugin system
- Advanced features for custom deployments

**When to use:** Only if you're building a custom MATHIR integration.

---

## Minimum Viable Setup (3 files)

```
your_project/
├── mathir_daemon.py    ← Copy from MATHIR/bin/
├── mathir_client.py    ← Copy from MATHIR/bin/
├── mathir.json         ← Create with your config
└── .mathir/            ← Created automatically
    └── mathir.db       ← Created automatically
```

**Steps:**
```bash
# 1. Copy files
cp /path/to/MATHIR/bin/mathir_daemon.py .
cp /path/to/MATHIR/bin/mathir_client.py .

# 2. Create config
cat > mathir.json << 'EOF'
{
  "model": "BAAI/bge-large-en-v1.5",
  "device": "cuda",
  "embedding_dim": 1024,
  "port": 7338,
  "db_path": ".mathir/mathir.db"
}
EOF

# 3. Start daemon
python mathir_daemon.py

# 4. Test
python mathir_client.py ping
python mathir_client.py save "Hello MATHIR" --agent test --type semantic --label test
python mathir_client.py recall "Hello" -k 1
```

---

## Full Setup with Dashboard (5 files)

```
your_project/
├── mathir_daemon.py
├── mathir_client.py
├── mathir.json
├── dashboard_server.py
├── dashboard.html
└── .mathir/
    └── mathir.db
```

---

## Full Setup with MCP (6 files)

```
your_project/
├── mathir_daemon.py
├── mathir_client.py
├── mathir.json
├── mathir_mcp_server.py
├── mathir_push.py
└── .mathir/
    └── mathir.db
```

Then add to your MCP tool's config:
```json
{ "mcpServers": { "mathir": { "url": "http://127.0.0.1:7338/sse" } } }
```

---

## Dependencies

```bash
# Required
pip install sentence-transformers  # Embedding model
pip install torch                   # PyTorch (CUDA)
pip install sqlite-vec              # Vector search

# Optional
pip install usearch                 # HNSW index (N >= 5K)
pip install onnxruntime-gpu         # ONNX edge deployment
```

---

## File Summary Table

| File | Required | What It Does |
|------|----------|-------------|
| `bin/mathir_daemon.py` | ✅ YES | Persistent daemon, keeps model loaded |
| `bin/mathir_client.py` | ✅ YES | Fast client commands |
| `mathir.json` | ✅ YES | Config (model, dims, paths) |
| `bin/mathir_push.py` | Recommended | Proactive memory delivery |
| `mathir_mcp_server.py` | Recommended | MCP server (6 tools) |
| `dashboard_server.py` | Optional | Dashboard backend |
| `dashboard.html` | Optional | Dashboard frontend |
| `mathir_search.py` | Optional | HybridSearch (numpy + USearch) |
| `mathir_vec.py` | Optional | VecMemory (sqlite-vec) |
| `mathir_vec_optimized.py` | Optional | Optimized VecMemory |
| `mathir_gpu_vec.py` | Optional | GPU brute-force backend |
| `mathir_lib/*` | Optional | Core library (advanced) |
| `config/*.yaml` | Optional | Pre-built configs |

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| "Connection refused" | Daemon not running | `python mathir_daemon.py` |
| "Model not found" | First run, downloading | Wait for model download (~1GB) |
| "CUDA out of memory" | GPU VRAM full | Use `--device cpu` or smaller model |
| "Port in use" | Another daemon running | `--port 8080` or kill existing |
| "Slow first request" | Cold model load | Normal, subsequent requests are fast |
| "No database found" | `.mathir/` doesn't exist | Created automatically on first save |

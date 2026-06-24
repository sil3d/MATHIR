# MATHIR MCP — Universal Installation

**5-tier cognitive memory for 50 AI coding agents. Install once, use everywhere.**

---

## Quick Start (3 Steps)

### 1. Install globally (one time)

```bash
# Clone
git clone https://github.com/sil3d/MATHIR.git /tmp/MATHIR
cp -r /tmp/MATHIR/mcp ~/.config/MATHIR

# Install deps
pip install -r ~/.config/MATHIR/mathir_lib/requirements.txt
```

### 2. Run smart installer

```bash
# Windows (double-click)
~/.config/MATHIR/install.bat

# Mac/Linux
chmod +x ~/.config/MATHIR/install.sh
~/.config/MATHIR/install.sh

# Or directly
python ~/.config/MATHIR/install_smart.py
```

The installer:
- Auto-detects 50 coding agents
- Injects MCP config into each
- Copies instructions where supported

### 3. Done

Restart your agent. MATHIR is ready.

---

## ⚠️ Moving the Folder After Install

The installer writes an **absolute path** to your agent's config. If you move `mathir_mcp/` to another location, **re-run `install.bat` or `install.sh`** from the new location to update the configs.

```
# Example: moved from Downloads to Documents
cd ~/Documents/mathir_mcp
install.bat          # Windows
./install.sh         # Linux/Mac
```

You'll see a warning in the MCP server logs if the path is stale.

---

## Supported Agents (50)

| Category | Agents |
|----------|--------|
| **Dev Platforms** | OpenCode, MiMo, Claude Code, Claude Desktop, Cursor, Cline, Roo Code, Continue.dev, Hermes Agent, pi, OpenHands, Agent Zero, Slate Agent, Kilo Code, Windsurf, Gemini CLI, Zcode, OpenClaw, Kiro, Qwen Code, Crush, Warp, Trae, Factory, Goose, Amp, Zed, Augment Code, Antigravity |
| **Cloud/AI** | GitHub Copilot, Aider, Codex (OpenAI), Codebuff, Letaido, CodeAF, JONI, Minara, V12X, Anakot Agent, PostQode, starchild, Flo Agent, camelAI, Kern Agent, AGNT, Verdent, Ante, GitLawb, Favur, OpenSquilla |

**No agent? Install OpenCode** (free models): https://opencode.ai/

---

## If Installer Fails

Give the entire `~/.config/MATHIR/` folder to your coding agent.
It will read `docs/AGENT.md` and configure MATHIR automatically.

---

## What's Inside

```
~/.config/MATHIR/
├── mathir_lib/                      ← Core library
│   ├── mathir_mcp_server.py         ← MCP server (9 tools)
│   ├── mathir_daemon.py             ← Persistent daemon
│   ├── mathir_client.py             ← CLI client
│   ├── mathir_vec.py                ← VecMemory (sqlite-vec)
│   ├── mathir_search.py             ← HybridSearch
│   └── requirements.txt
├── brain/                           ← Brain architecture (5 phases)
├── config/                          ← Config files
├── dashboard/                       ← Neural dashboard
├── docs/                            ← Documentation
├── install_smart.py                 ← Smart installer
├── install.bat                      ← Windows launcher
├── install.sh                       ← Mac/Linux launcher
└── GLOBAL_INSTRUCTIONS.md           ← Universal AI instructions
```

---

## MCP Tools (9)

| Tool | Description |
|------|-------------|
| `memory_save` | Save a memory block |
| `memory_recall` | Search by similarity |
| `memory_smart_search` | Hybrid search (vector + BM25 + RRF) |
| `memory_audit` | View memory audit trail |
| `memory_export` | Export all memory data as JSON |
| `memory_delete` | Soft-delete a memory |
| `memory_sessions` | List recent memory sessions |
| `memory_stats` | Get statistics |
| `memory_dashboard` | Launch / check Neural Memory Dashboard |

Canonical list — matches `mathir_lib/mathir_mcp_server.py` TOOLS array (lines 249–358).

---

## Embedding Model

**paraphrase-multilingual-MiniLM-L12-v2** — 384 dimensions, 239MB VRAM, 50+ languages

---

## Documentation

| File | Description |
|------|-------------|
| `docs/AGENT.md` | Agent deployment guide |
| `docs/INTEGRATION.md` | Platform integration |
| `docs/BRAIN_ARCHITECTURE.md` | Brain stack details |
| `docs/DAEMON.md` | Daemon protocol |
| `docs/DIMENSIONS.md` | Embedding dimensions |
| `docs/GPU_SETUP.md` | GPU acceleration |
| `docs/DASHBOARD_GUIDE.md` | Dashboard setup |
| `docs/MODEL_COMPARISON.md` | Model benchmarks |

---

## Multilingual Help

**EN:** If installer fails, give `~/.config/MATHIR/` to your coding agent. It reads `docs/AGENT.md`.

**FR:** Si l'installeur echoue, donnez `~/.config/MATHIR/` a votre agent. Il lit `docs/AGENT.md`.

---

## Security & Input Limits

The MCP server enforces strict per-field length caps to prevent DoS via unbounded payloads:

| Field | Default cap | Env var |
|-------|-------------|---------|
| `content` | 100 KB | `MCP_INPUT_MAX` (multiplier, e.g. `2.0` doubles all caps) |
| `query` | 5 KB | `MCP_INPUT_MAX` |
| `label` | 200 B | `MCP_INPUT_MAX` |
| `agent` | 100 B | `MCP_INPUT_MAX` |

Set `MCP_INPUT_MAX=2.0` to relax for batch jobs. Out-of-range values fall back to default. Rejected payloads return `{"error": "<field> exceeds <cap> chars"}`.

Additionally:
- `api_import_db` rejects any `project_name` outside `^[a-zA-Z0-9_-]{1,64}$`.
- `handle_memory_export` whitelists SQL tables; arbitrary table names are blocked.
- New DB files are created with `0o600` permissions where the OS supports it.

**ES:** Si el instalador falla, dea `~/.config/MATHIR/` a su agente. Leera `docs/AGENT.md`.

**ZH:** Ruguio anzhuan shibai, ba `~/.config/MATHIR/` gei nide agent. Hui du `docs/AGENT.md`.

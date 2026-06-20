# MATHIR — Agent Deployment Guide (v8.3+)

**Universal install: one folder, 50 agents, zero config.**

---

## TL;DR

```
You need 3 things:
1. Install MATHIR: ~/.config/MATHIR/ (global, once)
2. Run installer:  python ~/.config/MATHIR/install_smart.py
3. That's it.

The installer auto-detects your coding agents and configures them.
Each project gets its own database at .mathir/mathir.db.
```

---

## Quick Start (3 Steps)

### Step 1: Install MATHIR globally

```bash
# Clone or copy
git clone https://github.com/sil3d/MATHIR.git /tmp/MATHIR
cp -r /tmp/MATHIR/mcp ~/.config/MATHIR

# Or download release
curl -L https://github.com/sil3d/MATHIR/archive/main.zip -o mathir.zip
unzip mathir.zip -d ~/.config/MATHIR
```

### Step 2: Run smart installer

```bash
# Windows (double-click)
~/.config/MATHIR/install.bat

# Mac/Linux (terminal)
chmod +x ~/.config/MATHIR/install.sh
~/.config/MATHIR/install.sh

# Or directly
python ~/.config/MATHIR/install_smart.py
```

### Step 3: Done

- Installer detects your agents (50 supported)
- Injects MCP config + instructions automatically
- Restart your agent to use MATHIR

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

## Global Install Structure

```
~/.config/MATHIR/                    ← Global install (once)
├── mathir_lib/                      ← Core library
│   ├── mathir_mcp_server.py         ← MCP server entry point
│   ├── mathir_daemon.py             ← Persistent daemon
│   ├── mathir_client.py             ← CLI client
│   ├── mathir_vec.py                ← VecMemory (sqlite-vec)
│   ├── mathir_search.py             ← HybridSearch (vector + BM25 + RRF)
│   ├── memory_risks.py              ← Risk mitigation
│   └── requirements.txt             ← Dependencies
├── brain/                           ← Brain architecture
│   ├── mathir_brain.py              ← Master controller
│   ├── mathir_inject_proxy.py       ← Auto-injection proxy
│   ├── mathir_watchdog.py           ← Daemon watchdog
│   ├── mathir_spread.py             ← Spreading activation
│   ├── mathir_consolidate.py        ← Nightly consolidation
│   └── mathir_prime.py              ← Pre-cognitive priming
├── config/
│   └── mathir.json                  ← MATHIR config
├── dashboard/                       ← Neural dashboard
├── docs/                            ← Documentation
│   ├── AGENT.md                     ← This file
│   ├── GLOBAL_INSTRUCTIONS.md       ← Universal AI instructions
│   ├── BRAIN_ARCHITECTURE.md        ← Brain stack details
│   └── ... (8 more docs)
├── install_smart.py                 ← Smart installer (50 agents)
├── install.bat                      ← Windows launcher
├── install.sh                       ← Mac/Linux launcher
└── config_template.json             ← Config template
```

---

## Embedding Model

**paraphrase-multilingual-MiniLM-L12-v2** — 384 dimensions

| Property | Value |
|----------|-------|
| Dimensions | 384 |
| Parameters | 117.7M |
| Max tokens | 128 |
| Languages | 50+ |
| VRAM (fp16) | 239MB |
| Speed | ~22ms/embedding |

All databases must use 384d vectors. If you have old 1024d databases, run migration:
```bash
python ~/.config/MATHIR/dev/migrate_db.py --db /path/to/mathir.db --new-dim 384
```

---

## CRITICAL: Instructions Must Be Loaded Too

**The installer configures MCP (the tool), but the AI also needs INSTRUCTIONS (how to use it).**

Without instructions, the AI sees `memory_save`, `memory_recall` etc. but doesn't know:
- WHEN to save (after every task)
- WHEN to recall (before starting work)
- That memory is PROACTIVE (auto-injected, not queried)
- That MATHIR is already running

### What the installer does automatically

For agents with `supports_instructions: True`:
- OpenCode, MiMo, Claude Code, Claude Desktop, Cursor, Cline, Zcode, etc.
- Installer injects `GLOBAL_INSTRUCTIONS.md` into their instructions file
- **Nothing to do — it just works**

### What YOU must do manually (agents without instruction support)

For agents with `supports_instructions: False` (Kilo Code, Windsurf, Gemini CLI, etc.):

**Step 1:** Copy `GLOBAL_INSTRUCTIONS.md` from `~/.config/MATHIR/`

**Step 2:** Add it to your agent's instructions. Examples:

| Agent | Where to put instructions |
|-------|--------------------------|
| **Kilo Code** | Create `~/.config/kilo/AGENTS.md` or project `AGENTS.md` |
| **Windsurf** | Create `~/.codeium/windsurf/rules/mathir.md` |
| **Gemini CLI** | Add to `~/.gemini/settings.json` under `"instructions"` |
| **Aider** | Create `.aider.conf.yml` with `read: ~/.config/MATHIR/GLOBAL_INSTRUCTIONS.md` |
| **Any agent** | Paste the content into your agent's system prompt or instructions file |

**Step 3:** Verify the agent can see MATHIR:

```
Ask your agent: "Do you have MATHIR memory tools?"
If it says YES → done
If it says NO → instructions not loaded, check Step 2
```

### Why both are needed

```
MCP config     → tells the agent TOOLS EXIST (memory_save, memory_recall)
Instructions   → tells the agent HOW and WHEN to use them

Without MCP:     Agent can't call MATHIR (no tools)
Without instructions: Agent has tools but doesn't use them (forgets to save/recall)

You need BOTH.
```

### The GLOBAL_INSTRUCTIONS.md content

This file tells the agent:
1. MATHIR is your PROACTIVE memory — auto-injected, not queried
2. Save after EVERY task completion
3. Recall before EVERY significant action
4. Memory is BEHAVIORAL, not a command
5. Never say "pre-existing error" and skip fixing
6. Never delete or comment code to hide errors

**It's 75 lines. Copy it into your agent's instructions.**

---

## Changing Models

The default model (384d) is optimized for **speed and low VRAM**. To change:

| Priority | Model | Dims | VRAM | Speed | Why |
|----------|-------|------|------|-------|-----|
| Default | paraphrase-multilingual | 384 | 239MB | ~104ms | 50+ langs, low VRAM |
| Balance | nomic-embed-text-v1.5 | 768 | ~500MB | ~21ms | Better quality |
| Quality | bge-large-en-v1.5 | 1024 | ~1.5GB | ~3ms | High quality, English |
| Max | Qwen2.5-7B-emb | 3584 | ~4.7GB | ~30ms | Best quality |

### How to change:

```bash
# 1. Edit config
nano ~/.config/MATHIR/config/mathir.json
# Change "model" and "embedding_dim"

# 2. Migrate existing DB
python ~/.config/MATHIR/dev/migrate_db.py --db .mathir/mathir.db --new-dim 768

# 3. Restart daemon
python ~/.config/MATHIR/mathir_lib/mathir_client.py stop
python ~/.config/MATHIR/mathir_lib/mathir_daemon.py
```

**Full guide:** See `docs/DIMENSIONS.md`

---

## HybridSearch

Vector + BM25 + RRF fusion (k=60). ~60ms per search.

| Method | Best For |
|--------|----------|
| `memory_recall` | Semantic similarity (vector) |
| `memory_smart_search` | Hybrid (vector + text, best quality) |
| `memory_hybrid_search` | Hybrid with custom weights |
| `memory_search` | Fast text-only (no embedding) |

---

## MCP Tools (9)

| Tool | Description |
|------|-------------|
| `memory_save` | Save a memory block |
| `memory_recall` | Search by similarity |
| `memory_smart_search` | Hybrid search (vector + BM25 + RRF) |
| `memory_hybrid_search` | Hybrid with custom weights |
| `memory_search` | Fast text search (no embedding) |
| `memory_stats` | Get statistics |
| `memory_delete` | Soft-delete |
| `memory_push` | Proactive delivery |
| `memory_audit` | Audit trail |

---

## Brain Architecture (5 Phases)

| Phase | Script | Purpose |
|---|---|---|
| 1 | `mathir_inject_proxy.py` | Auto-inject top-3 memories into every LLM call (port 8182) |
| 2 | `mathir_watchdog.py` | Auto-restart daemon on crash (7s recovery) |
| 3 | `mathir_spread.py` | Spreading activation (related memories via link graph) |
| 4 | `mathir_consolidate.py` | Nightly: merge duplicates, decay unused, archive dead |
| 5 | `mathir_prime.py` | Pre-cognitive: senses cwd/git before user query |

### One-command brain stack

```bash
python ~/.config/MATHIR/brain/mathir_brain.py start
python ~/.config/MATHIR/brain/mathir_brain.py status
```

### How auto-injection works

1. Proxy listens on port 8182
2. Takes last user message from LLM request
3. Calls `daemon.recall(k=3)` in <300ms
4. Injects memories as `{{MATHIR_CONTEXT}}` in system prompt
5. Forwards to your real LLM (8181)

**The agent never needs to call recall. Memories are pre-injected.**

---

## Per-Project Databases

Each project gets its own database automatically:

```
your_project/
├── .mathir/
│   └── mathir.db    ← Auto-created on first save
├── src/
└── ...
```

**DB detection priority:**
1. `.mathir/mathir.db` in CWD → agent working IN a project gets that DB
2. Registry lookup → daemon started from home uses registered project DBs
3. Home directory is NEVER treated as a project → prevents stale home DBs
4. Fallback → first available registry DB when CWD is home

---

## Agent Injection — Make Memory Automatic

### GLOBAL_INSTRUCTIONS.md (at mcp/ ROOT)

Copy `~/.config/MATHIR/GLOBAL_INSTRUCTIONS.md` into your agent's instructions.
This makes every agent aware of MATHIR automatically.

### Per-Agent Injection

For OpenCode, create `.md` files in `~/.config/opencode/agents/`:

```yaml
---
description: My custom agent
mode: subagent
---

<!-- MATHIR_INJECTED -->
# MATHIR MEMORY — AUTO-INJECTION BLOCK
# The {{MATHIR_CONTEXT}} placeholder is filled at RUNTIME by mathir_inject_proxy.py
# with the 3 most relevant memories based on the user's last message.

## YOUR MEMORY IS ALWAYS ACTIVE
... (see GLOBAL_INSTRUCTIONS.md for full block)
```

---

## Platform-Specific Configs

### OpenCode

Config: `~/.config/opencode/opencode.json`
```json
{
  "mcp": {
    "mathir": {
      "type": "local",
      "command": ["python", "~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "environment": { "MATHIR_EMBEDDING_DIM": "384" },
      "enabled": true
    }
  }
}
```

### MiMo

Config: `~/.config/mimocode/mimocode.json`
```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

### Kilo Code

CLI config: `~/.config/kilo/kilo.json` (uses `"mcp"` key)
VS Code extension: `.kilocode/mcp.json` (uses `"mcpServers"` key)

### Claude Code

Config: `~/.claude.json`
```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

### Cursor

Config: `~/.cursor/mcp.json`
```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["~/.config/MATHIR/mathir_lib/mathir_mcp_server.py"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

---

## Dependencies

```bash
# Required
pip install sentence-transformers torch sqlite-vec aiohttp rank_bm25

# Optional (GPU acceleration)
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| "Connection refused" | Daemon not running | `python ~/.config/MATHIR/mathir_lib/mathir_daemon.py` |
| "Model not found" | First run, downloading | Wait for model download (~1GB) |
| "CUDA out of memory" | GPU VRAM full | Use `--device cpu` or smaller model |
| "Port in use" | Another daemon running | `--port 8080` or kill existing |
| "Slow first request" | Cold model load | Normal, subsequent requests are fast |
| "No database found" | `.mathir/` doesn't exist | Created automatically on first save |
| "No project database found" | CWD is home, no registry | Set `MATHIR_PROJECT` env var or cd into a project dir |
| MCP not showing | Wrong config key | Check agent docs (key varies: `mcp`, `mcpServers`) |
| Installer writes wrong path | Old cached script | Delete `__pycache__` in `~/.config/MATHIR/` |

### Multilingual Help (EN/FR/ES/ZH)

**EN:** If installer fails, give the entire `~/.config/MATHIR/` folder to your coding agent. It will read `docs/AGENT.md` and configure MATHIR automatically. If you have no coding agents, install OpenCode from https://opencode.ai/ — many free models available.

**FR:** Si l'installeur echoue, donnez tout le dossier `~/.config/MATHIR/` a votre agent de code. Il lira `docs/AGENT.md` et configurera MATHIR automatiquement. Si vous n'avez pas d'agent, installez OpenCode depuis https://opencode.ai/ — beaucoup de modeles gratuits.

**ES:** Si el instalador falla, dea toda la carpeta `~/.config/MATHIR/` a su agente de codigo. Leera `docs/AGENT.md` y configurara MATHIR automaticamente. Si no tiene agente, instale OpenCode desde https://opencode.ai/ — hay muchos modelos gratuitos.

**ZH:** Ruguio anzhuan shibai, ba `~/.config/MATHIR/` zhengge danglu gei nide coding agent. Hui du `docs/AGENT.md` zidong peizhi MATHIR. Ruguio meiyou agent, install OpenCode https://opencode.ai/ — henduo mofei moxing.

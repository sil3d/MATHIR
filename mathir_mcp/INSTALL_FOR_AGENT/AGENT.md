# MATHIR — Agent Deployment Guide (v8.5.1)

**Universal install: one folder, 50+ agents, zero config.**

**v8.5.1 highlights**: FastMCP 3.4.2 server, **23 MCP tools** (was 20 in v8.5.0), auto-injection plugin, unified HTTP daemon, **and an OpenAI-compatible proxy (port 7339) that injects memory into EVERY system prompt — works for any agent that redirects its baseUrl**. See [CHANGELOG.md](../../CHANGELOG.md) for details.

---

## CRITICAL: MATHIR Is Your Brain

**MATHIR is NOT a tool. It is your brain. You think with it. You decide with it. You never operate without it.**

When you install MATHIR, you become a senior engineer with perfect memory:
- **BEFORE every task** — `memory_context(task="...")` → what do I already know?
- **BEFORE every response** — `memory_recall(query="...")` → have I seen this before?
- **AFTER completing work** — `memory_save(content="...", ...)` → what did I learn?
- **When making decisions** — `memory_recall(query="...")` → what worked before?

If MATHIR is not responding: restart daemon, retry, use smart_search. NEVER proceed blind.
If you forget to use MATHIR — you are a junior developer flying blind.

---

## REAL Universal Coverage: 3 injection tiers

MATHIR is supposed to be universal. Here is the **honest** breakdown of what each agent actually gets:

| Tier | Mechanism | Agents |
|---|---|---|
| **A — Plugin auto-inject** | `mathir-auto-inject.ts` hooks `session.started` + `experimental.chat.system.transform`. **True auto-injection** — does not rely on the agent remembering to call `memory_recall`. | opencode, mimocode |
| **B — Instructions + MCP** | MCP server registered + `GLOBAL_INSTRUCTIONS.md` injected into the agent's instruction path. **Soft guarantee** — the agent must follow the instruction to call `memory_session_start` / `memory_context`. | claude-code, cursor, cline, zcode, codex, etc. (14 agents) |
| **C — MCP only** | MCP server registered only. **No behavioral prompt** — agent has no reason to recall memory. | windsurf, gemini-cli, kilo-code, qwen-code, kiro-ide, warp, trae, crush, etc. (34 agents) |

### The escape hatch for Tier C: **MATHIR proxy on port 7339**

Any OpenAI-compatible agent (which includes ~all of them: Claude Code via `OPENAI_BASE_URL`, Cursor, Cline, Continue, Codex) can point its baseUrl at `http://127.0.0.1:7339/v1`. The proxy intercepts every LLM call, queries the MATHIR daemon for relevant memories, and **prepends them to the system prompt** as a `<mathir-auto-injection>` block — silently, on every call, regardless of whether the agent cooperates.

This is the **true universal coverage**. It is shipped with MATHIR (script at `~/.config/MATHIR/mathir_mcp/mathir_lib/mathir_proxy.py`) and auto-started alongside the daemon on Windows login (see `mathir_daemon_startup.bat` in the Startup folder).

```bash
# Start the proxy (daemon must already be running on 7338)
python -m mathir_mcp.mathir_lib.mathir_proxy     # port 7339
# OR:
python ~/.config/MATHIR/mathir_mcp/mathir_lib/mathir_proxy.py --port 7339

# Then in your agent:
export OPENAI_BASE_URL=http://127.0.0.1:7339/v1
# Done. Every LLM call now has memory auto-injected.
```

### The other escape hatch: **`AGENTS.md` at repo root**

26+ agents (Aider, Amp, Claude Code, Codex, Cursor, Devin, Factory, Goose, JetBrains Junie, Jules, OpenCode, VS Code Copilot, Warp, Zed, etc. — see https://agents.md) auto-read `AGENTS.md` at the project root. MATHIR ships a template at `mathir_mcp/opencode_templates/AGENTS.md` that you can copy to any project:

```bash
cp ~/.config/MATHIR/mathir_mcp/opencode_templates/AGENTS.md /path/to/your/project/AGENTS.md
```

The template instructs the agent to call `memory_session_start` on first turn + `memory_context` before each task. **Combined with the proxy, this gives 100% coverage for the agents that follow agents.md + 100% for OpenAI-compatible ones.**

---

## TL;DR

```
You need 3 things:
1. Install MATHIR: ~/.config/MATHIR/ (global, once) — see Step 1 for the EXACT layout
2. Run installer:  python ~/.config/MATHIR/INSTALL_FOR_DEV/install_smart.py
3. That's it.

The installer auto-detects your coding agents and configures them.
Each project gets its own database at .mathir/mathir.db.
```

**Universal install: one folder, 50 agents, zero config.**

---

## TL;DR

```
You need 3 things:
1. Install MATHIR: ~/.config/MATHIR/ (global, once) — must end with the layout in §"Global Install Structure"
2. Run installer:  python ~/.config/MATHIR/INSTALL_FOR_DEV/install_smart.py
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
cp -r /tmp/MATHIR/mathir_mcp ~/.config/MATHIR
# IMPORTANT (Linux/BSD `cp -r`): copies CONTENTS of mathir_mcp/ into ~/.config/MATHIR/
# After this, ~/.config/MATHIR/ must contain "mathir_mcp/" as a SUBDIRECTORY.

# Or download release
curl -L https://github.com/sil3d/MATHIR/archive/main.zip -o mathir.zip
unzip mathir.zip -d ~/.config/MATHIR
```

**⚠️ Windows (`Copy-Item`) behaves differently from Linux `cp -r`:**
`Copy-Item .\mathir_mcp\* ~/.config/MATHIR\` copies CONTENTS into the destination as well, which produces the correct layout. But if you use `Copy-Item .\mathir_mcp ~/.config\MATHIR\` (without the `\*`), you get `~/.config/MATHIR/mathir_mcp/` as a single subdir, which is what `install_smart.py` ALSO accepts (and is actually clearer). **Either layout works, AS LONG AS `install_smart.py` can find `mathir_mcp/` from where it sits**:

- Layout A: `~/.config/MATHIR/mathir_lib/`, `~/.config/MATHIR/__init__.py`, `~/.config/MATHIR/pyproject.toml` — **install_smart.py will look for `~/.config/MATHIR/mathir_mcp/` here and FAIL**.
- Layout B: `~/.config/MATHIR/mathir_mcp/mathir_lib/`, `~/.config/MATHIR/mathir_mcp/__init__.py`, `~/.config/MATHIR/mathir_mcp/pyproject.toml` — ✅ **this is the one that works**.

If you have layout A, restructure first (see §"Restructuring After Clone" below).

### Step 2: Run smart installer

```bash
# Windows (double-click)
~/.config/MATHIR/INSTALL_FOR_DEV/install.bat

# Mac/Linux (terminal)
chmod +x ~/.config/MATHIR/INSTALL_FOR_DEV/install.sh
~/.config/MATHIR/INSTALL_FOR_DEV/install.sh

# Or directly (note: installer lives under INSTALL_FOR_AGENT/, not at the MATHIR root)
python ~/.config/MATHIR/INSTALL_FOR_DEV/install_smart.py
```

The installer is interactive: type `A` to configure all detected agents, or pick specific numbers (comma-separated).

### Step 3: Done

- Installer detects your agents (50 supported)
- Injects MCP config + instructions automatically
- Restart your agent to use MATHIR

### Restructuring After Clone

If your source repo was cloned and the package files ended up at `~/.config/MATHIR/` (Layout A — wrong), move them into a `mathir_mcp/` subdir:

```bash
# Linux/Mac
mkdir -p ~/.config/MATHIR/mathir_mcp
mv ~/.config/MATHIR/mathir_lib ~/.config/MATHIR/mathir_mcp/
mv ~/.config/MATHIR/__init__.py ~/.config/MATHIR/mathir_mcp/
mv ~/.config/MATHIR/__main__.py ~/.config/MATHIR/mathir_mcp/
mv ~/.config/MATHIR/pyproject.toml ~/.config/MATHIR/mathir_mcp/

# Windows (PowerShell)
New-Item -ItemType Directory -Path "$env:USERPROFILE\.config\MATHIR\mathir_mcp" -Force
Move-Item "$env:USERPROFILE\.config\MATHIR\mathir_lib" "$env:USERPROFILE\.config\MATHIR\mathir_mcp\" -Force
Move-Item "$env:USERPROFILE\.config\MATHIR\__init__.py" "$env:USERPROFILE\.config\MATHIR\mathir_mcp\" -Force
Move-Item "$env:USERPROFILE\.config\MATHIR\__main__.py" "$env:USERPROFILE\.config\MATHIR\mathir_mcp\" -Force
Move-Item "$env:USERPROFILE\.config\MATHIR\pyproject.toml" "$env:USERPROFILE\.config\MATHIR\mathir_mcp\" -Force

# Then re-register the package
pip install -e ~/.config/MATHIR/mathir_mcp
```

Files that should STAY at `~/.config/MATHIR/` root (not moved):
`INSTALL_FOR_AGENT/`, `INSTALL_FOR_DEV/`, `docs/`, `config_template.json`, `GLOBAL_INSTRUCTIONS.md`,
`mathir_dashboard.bat/sh`, `mcp_architecture.md`, `opencode_templates/`,
`mimocode_templates/`, `README.md`, `CHANGELOG.md`.

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
It will read `INSTALL_FOR_AGENT/AGENT.md` and configure MATHIR automatically.

---

## Global Install Structure

```
~/.config/MATHIR/                    ← Global install (once)
├── mathir_mcp/                      ← Python package (for `pip install -e`)
│   ├── __init__.py                  ← Package marker (so `import mathir_mcp` works)
│   ├── __main__.py                  ← `python -m mathir_mcp` entry (launches daemon by default, or `--mcp` for MCP stdio server)
│   ├── pyproject.toml               ← Build metadata (`pip install -e .`)
│   ├── mathir_lib/                  ← Core library (imported as `mathir_mcp.mathir_lib`)
│   │   ├── mathir_mcp_server.py     ← MCP stdio server entry point (line `from mathir_mcp.mathir_lib import mathir_mcp_server`)
│   │   ├── mathir_server.py         ← Persistent HTTP daemon (Flask + Waitress)
│   │   ├── mathir_daemon.py         ← Legacy raw-socket daemon (superseded by mathir_server.py)
│   │   ├── mathir_client.py         ← CLI client
│   │   ├── mathir_vec.py            ← VecMemory (sqlite-vec)
│   │   ├── mathir_search.py         ← HybridSearch (vector + BM25 + RRF)
│   │   ├── memory_risks.py          ← Risk mitigation
│   │   └── requirements.txt         ← Dependencies
│   ├── brain/                       ← Brain architecture
│   │   ├── mathir_brain.py          ← Master controller
│   │   ├── mathir_inject_proxy.py   ← Auto-injection proxy (port 8182)
│   │   ├── mathir_watchdog.py       ← Daemon watchdog
│   │   ├── mathir_spread.py         ← Spreading activation
│   │   ├── mathir_consolidate.py    ← Nightly consolidation
│   │   └── mathir_prime.py          ← Pre-cognitive priming
│   ├── config/
│   │   └── mathir.json              ← MATHIR config
│   ├── dashboard/                   ← Neural dashboard (legacy)
│   ├── dev/                         ← Migration/dev scripts
│   ├── tests/                       ← pytest suite
│   └── ... (other package internals)
├── INSTALL_FOR_AGENT/               ← Auto-installer for AI coding agents (smart installer scripts)
│   ├── install_smart.py             ← The installer — 40+ agents auto-detected
│   ├── install.bat                  ← Windows launcher
│   └── install.sh                   ← Mac/Linux launcher
├── INSTALL_FOR_DEV/                 ← Step-by-step guides for HUMAN developers
│   ├── INSTALL_WINDOWS.md           ← Windows 10/11 walkthrough
│   ├── INSTALL_LINUX.md             ← Linux walkthrough
│   ├── INSTALL_MACOS.md             ← macOS walkthrough
│   └── README.md
│   ├── install_smart.py             ← Smart installer (50 agents)
│   ├── install.bat                  ← Windows launcher
│   ├── install.sh                   ← Mac/Linux launcher
│   ├── INSTALL_WINDOWS.md           ← Windows install walkthrough
│   ├── INSTALL_LINUX.md             ← Linux walkthrough
│   └── INSTALL_MACOS.md             ← macOS walkthrough
├── docs/                            ← Documentation (top-level — what users read first)
│   ├── AGENT.md                     ← This file
│   ├── GLOBAL_INSTRUCTIONS.md       ← Universal AI instructions
│   ├── BRAIN_ARCHITECTURE.md        ← Brain stack details
│   └── ... (8 more docs)
├── opencode_templates/              ← Per-agent template files (NOT installed by default)
├── mimocode_templates/              ← MiMo-code-specific templates
├── mathir_dropin/                   ← Standalone module for ad-hoc `from mathir_dropin import MATHIRMemory`
├── config_template.json             ← Config template
├── GLOBAL_INSTRUCTIONS.md           ← Distributed to each agent's instructions
├── README.md
└── CHANGELOG.md

NOTE: The `~/.config/MATHIR/INSTALL_FOR_DEV/install_smart.py` script reads its source
location via `Path(__file__).resolve().parent.parent` then appends `mathir_mcp`
to find the package to copy into each agent's tools dir. So the layout above
(with `mathir_mcp/` as a subdirectory of `~/.config/MATHIR/`) is the **only
layout that works out of the box**.
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
# Kill the running daemon then start fresh
pkill -f "mathir_mcp" || true
python -m mathir_mcp
```

**Full guide:** See `docs/DIMENSIONS.md`

---

## HybridSearch

Vector + BM25 + RRF fusion (k=60). ~60ms per search.

| Method | Best For |
|--------|----------|
| `memory_recall` | Semantic similarity (vector) |
| `memory_smart_search` | Hybrid (vector + text, best quality) |
| `memory_hybrid_search` | Explicit vector+BM25 fusion with tunable weights |

Note: `memory_search` was removed in v8.3 — functionality folded into `memory_smart_search` (auto-tuned weights, default k=10). v8.5.1 has **23 tools total** (2 auto-injection + 10 basic + 7 lifecycle + 3 advanced + 1 health check).

---

## MCP Tools (23 in v8.5.1)

### Basic (every day)
| Tool | Description |
|------|-------------|
| `memory_save` | Save a memory (5 tiers: working_memory, episodic, semantic, procedural, immunological) |
| `memory_recall` | Semantic search — auto-touches: increments recall_count, boosts stability |
| `memory_smart_search` | Daemon-native search (faster for high-throughput) |
| `memory_hybrid_search` | Vector + BM25 + RRF fusion (best for exact match queries) |
| `memory_audit` | View memory audit trail |
| `memory_export` | Export all memory data as JSON |
| `memory_delete` | Soft-delete a memory (sets tier='archived') |
| `memory_sessions` | List recent memory sessions |
| `memory_stats` | Get statistics by tier/agent/project |
| `memory_dashboard` | Launch / check Neural Memory Dashboard |

### Lifecycle (living memory)
| Tool | Description |
|------|-------------|
| `memory_promote` | Move a memory to the next tier (Ebbinghaus rules) |
| `memory_auto_promote` | Scan and promote all eligible memories |
| `memory_decay` | Apply Ebbinghaus decay (5%/30d), archive when stability < floor |
| `memory_consolidate` | Merge near-duplicates (cosine > threshold) |
| `memory_link` | Add an edge in the link graph (spreading activation) |
| `memory_get_links` | BFS traversal of the link graph |
| `memory_build_links` | Build the graph from cosine similarities |

### 5 memory tiers
| Tier | Use for |
|------|---------|
| `working_memory` | Current session scratchpad |
| `episodic` | Events: bugs fixed, decisions, sessions |
| `semantic` | Stable facts that apply broadly |
| `procedural` | How-to recipes (label must start with `how-to:` or `recipe:`) |
| `immunological` | Immune response — auto-quarantines toxic, biased, or unsafe memories |

Canonical list — matches `mathir_lib/mathir_mcp_server.py` TOOLS array.

---

## Brain Architecture (5 Phases)

| Phase | Script | Purpose |
|---|---|---|
| 1 | `mathir_inject_proxy.py` | Optional auto-inject proxy (port 8182) — only for agents that proxy LLM traffic |
| 2 | `mathir_watchdog.py` | Auto-restart daemon on crash (7s recovery) |
| 3 | `mathir_spread.py` | Spreading activation (related memories via link graph) |
| 4 | `mathir_consolidate.py` | Nightly: merge duplicates, decay unused, archive dead |
| 5 | `mathir_prime.py` | Pre-cognitive: senses cwd/git before user query |

### How memory works

**Two paths, both supported:**

**Path A — MCP tools (recommended for OpenCode, Claude Code, Cursor):**
The agent calls `memory_recall(query, k=5)` directly. Each call:
- Auto-touches the memory (increments recall_count, boosts stability)
- Returns top-k relevant memories
- <100ms latency

**Path B — Inject proxy (port 8182, optional):**
For agents that can't call MCP tools directly:
1. Proxy listens on port 8182
2. Takes last user message from LLM request
3. Calls `daemon.recall(k=3)` in <300ms
4. Injects memories as `{{MATHIR_CONTEXT}}` in system prompt
5. Forwards to your real LLM (8181)

> **Note:** OpenCode doesn't support `baseUrl` configuration, so Path A (direct MCP tools) is the only working path. Other agents (Claude Code, Cursor, MiMo) can use either path.

**In both paths, the agent has 20 tools (2 auto-injection + 10 basic + 7 lifecycle + 1 health check) at its disposal — no manual `memory_recall` is strictly required if instructions are properly loaded.**

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

### GLOBAL_INSTRUCTIONS.md (at mathir_mcp/ ROOT)

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
      "command": ["python", "-m", "mathir_mcp.mathir_lib.mathir_mcp_server"],
      "environment": {
        "MATHIR_EMBEDDING_DIM": "384",
        "MATHIR_PORT": "7338"
      },
      "enabled": true
    }
  }
}
```

> v8.4.1: entry point was `python -m mathir_mcp` (no longer valid in v8.5.0+ — this command now launches the HTTP daemon, not the MCP stdio server). Use `python -m mathir_mcp.mathir_lib.mathir_mcp_server` for the MCP server, or just use the global `mathir-mcp` script entry installed by `pip install -e`.

### MiMo

Config: `~/.config/mimocode/mimocode.json`

**MiMo Code v0.1.3+ is a fork of OpenCode** → uses the same `"mcp"` key and the same
JSON schema as OpenCode (https://opencode.ai/config.json). The older `"mcpServers"`
key is rejected by the current MiMo CLI with `Unrecognized key: "mcpServers"`.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "mathir": {
      "type": "local",
      "command": ["python", "C:\\Users\\<YOU>\\.config\\mimocode\\tools\\mathir_mcp\\mathir_lib\\mathir_mcp_server.py"],
      "environment": {
        "MATHIR_EMBEDDING_DIM": "384",
        "MATHIR_PORT": "7339",
        "PYTHONPATH": "C:\\Users\\<YOU>\\.config\\mimocode\\tools\\mathir_mcp\\mathir_lib"
      },
      "enabled": true
    }
  }
}
```

> Gotchas: `command` must be an **array** (not a single string), env field is
> **`environment`** (not `env`), and `type: "local"` is required.

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
      "args": ["-m", "mathir_mcp.mathir_lib.mathir_mcp_server"],
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
      "args": ["-m", "mathir_mcp.mathir_lib.mathir_mcp_server"],
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
| "Connection refused" | Daemon not running | `python -m mathir_mcp` |
| "Model not found" | First run, downloading | Wait for model download (~1GB) |
| "CUDA out of memory" | GPU VRAM full | Use `--device cpu` or smaller model |
| "Port in use" | Another daemon running | `--port 8080` or kill existing |
| "Slow first request" | Cold model load | Normal, subsequent requests are fast |
| "No database found" | `.mathir/` doesn't exist | Created automatically on first save |
| "No project database found" | CWD is home, no registry | Set `MATHIR_PROJECT` env var or cd into a project dir |
| MCP not showing | Wrong config key | **OpenCode & MiMo Code v0.1.3+** use `"mcp"`; **Claude Code, Cursor, Cline, Windsurf, Gemini, Zcode** use `"mcpServers"` |
| Installer writes wrong path | Old cached script | Delete `__pycache__` in `~/.config/MATHIR/` |

---

## Updating MATHIR (v8.5.1+)

MATHIR self-updates via GitHub Releases. Three things happen automatically:

1. **GitHub Action** (`.github/workflows/release.yml`) builds a `mathir-bundle-<version>.zip` on every push to `main` (dev prerelease) and every `v*.*.*` tag (stable).
2. **Daemon `/health`** polls GitHub every hour (cached) and exposes `update_available`, `latest_version`, `update_command`. The opencode/mimocode auto-inject plugin surfaces this to the agent at session start.
3. **`python -m mathir_mcp update`** is the user-facing command — atomic state machine with backup + migration + rollback.

### Check for updates

```bash
python -m mathir_mcp check              # quick status
python -m mathir_mcp check --force     # bypass cache, hit GitHub now
python -m mathir_mcp check --include-prerelease  # also see -rc/-dev releases
```

Output:
```
MATHIR Update Check
  current:      v8.5.1
  latest:       v8.5.2
  install mode: git
  install path: /home/user/.config/MATHIR/mathir_mcp
  [!] UPDATE AVAILABLE
  release:      https://github.com/sil3d/MATHIR/releases/tag/v8.5.2
  run:          python -m mathir_mcp update
```

### Apply an update

```bash
# To the latest stable release:
python -m mathir_mcp update

# To a specific version:
python -m mathir_mcp update --to v8.5.2

# From a local bundle zip (offline / air-gapped):
python -m mathir_mcp update --from-bundle ./mathir-bundle-8.5.2.zip

# Preview the plan without changing anything:
python -m mathir_mcp update --dry-run

# Auto-apply schema migrations without prompting:
python -m mathir_mcp update --force-apply
```

The state machine does: **DETECT → DISCOVER → BACKUP → DOWNLOAD → APPLY → MIGRATE → RESTART → REPORT**.

If any step fails, it auto-restores from the backup. Backups live in `~/.config/MATHIR/.backups/<old-version>-<timestamp>/` (kept last 10).

### Rollback

If an update breaks something subtle hours/days later:

```bash
python -m mathir_mcp rollback
```

This restores from the most recent backup and re-verifies daemon health.

### How auto-detection works

| Install layout | Mode | Update mechanism |
|---|---|---|
| `~/.config/MATHIR/mathir_mcp/.git` exists | **git** | `git fetch + git checkout v<target>` (refuses if local mods) |
| `~/.config/MATHIR/mathir_mcp/` only (pip install -e) | **bundle** | Downloads `mathir-bundle-<version>.zip` from GitHub Releases |

Set `MATHIR_DAEMON_RESTART` env var to override the daemon-restart mechanism (otherwise Windows uses `mathir_daemon_startup.bat` from Startup folder).

### Agent-copy sync

After every successful update, the CLI also syncs:
- `mathir_mcp_server.py` → `~/.config/opencode/tools/mathir_mcp/mathir_lib/` and `~/.config/mimocode/tools/mathir_mcp/mathir_lib/`
- `mathir-auto-inject.ts` → `~/.config/opencode/plugins/` and `~/.config/mimocode/plugins/`
- `GLOBAL_INSTRUCTIONS.md` → agent config roots

So all agents (opencode + mimocode + any tier-C via proxy 7339) pick up the new code automatically without manual file copying.

### Multilingual Help (EN/FR/ES/ZH)

**EN:** If installer fails, give the entire `~/.config/MATHIR/` folder to your coding agent. It will read `INSTALL_FOR_AGENT/AGENT.md` and configure MATHIR automatically. If you have no coding agents, install OpenCode from https://opencode.ai/ — many free models available.

**FR:** Si l'installeur echoue, donnez tout le dossier `~/.config/MATHIR/` a votre agent de code. Il lira `INSTALL_FOR_AGENT/AGENT.md` et configurera MATHIR automatiquement. Si vous n'avez pas d'agent, installez OpenCode depuis https://opencode.ai/ — beaucoup de modeles gratuits.

**ES:** Si el instalador falla, dea toda la carpeta `~/.config/MATHIR/` a su agente de codigo. Leera `INSTALL_FOR_AGENT/AGENT.md` y configurara MATHIR automaticamente. Si no tiene agente, instale OpenCode desde https://opencode.ai/ — hay muchos modelos gratuitos.

**ZH:** Ruguio anzhuan shibai, ba `~/.config/MATHIR/` zhengge danglu gei nide coding agent. Hui du `INSTALL_FOR_AGENT/AGENT.md` zidong peizhi MATHIR. Ruguio meiyou agent, install OpenCode https://opencode.ai/ — henduo mofei moxing.

---

## Per-Agent Integration

*(merged from INTEGRATION.md — v8.3+)*

### Quick Start (All Platforms)

`ash
# 1. Clone
git clone https://github.com/sil3d/MATHIR.git /tmp/MATHIR
cp -r /tmp/MATHIR/mathir_mcp ~/.config/MATHIR

# 2. Install deps
cd ~/.config/MATHIR
pip install -r mathir_lib/requirements.txt

# 3. Run smart installer
python install_smart.py
# Select your agent(s) from the menu
`

---

### Per-Agent Configs

#### OpenCode

**Config:** ~/.config/opencode/opencode.json

``json
{
  "mcp": {
    "mathir": {
      "type": "local",
      "command": ["python", "-m", "mathir_mcp.mathir_lib.mathir_mcp_server"],
      "environment": { "MATHIR_EMBEDDING_DIM": "384" },
      "enabled": true
    }
  }
}
``

**Instructions:** Copy GLOBAL_INSTRUCTIONS.md into ~/.config/opencode/GLOBAL_INSTRUCTIONS.md

---

#### MiMo

**Config:** ~/.config/mimocode/mimocode.json

> MiMo Code v0.1.3+ is a fork of OpenCode → use `"mcp"` key (NOT `"mcpServers"`).

``json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "mathir": {
      "type": "local",
      "command": ["python", "C:\\Users\\<YOU>\\.config\\mimocode\\tools\\mathir_mcp\\mathir_lib\\mathir_mcp_server.py"],
      "environment": {
        "MATHIR_EMBEDDING_DIM": "384",
        "MATHIR_PORT": "7339",
        "PYTHONPATH": "C:\\Users\\<YOU>\\.config\\mimocode\\tools\\mathir_mcp\\mathir_lib"
      },
      "enabled": true
    }
  }
}
``

**Instructions:** Copy GLOBAL_INSTRUCTIONS.md into ~/.config/mimocode/MEMORY.md

---

#### Claude Code

**Config:** ~/.claude.json

``json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["-m", "mathir_mcp.mathir_lib.mathir_mcp_server"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
``

**Instructions:** Copy GLOBAL_INSTRUCTIONS.md into ~/.claude/CLAUDE.md

---

#### Claude Desktop

**Config:** ~/Library/Application Support/Claude/claude_desktop_config.json (Mac)
or %APPDATA%\Claude\claude_desktop_config.json (Windows)

``json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["-m", "mathir_mcp.mathir_lib.mathir_mcp_server"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
``

---

#### Cursor

**Config:** ~/.cursor/mcp.json

``json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["-m", "mathir_mcp.mathir_lib.mathir_mcp_server"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
``

**Instructions:** Copy GLOBAL_INSTRUCTIONS.md into ~/.cursor/rules/mathir.md

---

#### Kilo Code

**CLI config:** ~/.config/kilo/kilo.json (uses "mcp" key)
**VS Code extension:** .kilocode/mcp.json (uses "mcpServers" key)

``json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["-m", "mathir_mcp.mathir_lib.mathir_mcp_server"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
``

---

#### Cline

**Config:** AppData\Roaming\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json

``json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["-m", "mathir_mcp.mathir_lib.mathir_mcp_server"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
``

---

#### Windsurf

**Config:** ~/.codeium/windsurf/mcp_config.json

``json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["-m", "mathir_mcp.mathir_lib.mathir_mcp_server"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
``

---

#### Gemini CLI

**Config:** ~/.gemini/settings.json

``json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["-m", "mathir_mcp.mathir_lib.mathir_mcp_server"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
``

---

#### Zcode

**Config:** ~/.zcode/v2/config.json

``json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["-m", "mathir_mcp.mathir_lib.mathir_mcp_server"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
``

---

### Manual Integration (Any Agent)

If your agent isn't listed:

1. Find its MCP config file
2. Add the appropriate JSON block:
   - `"mcp"` key (OpenCode schema): **OpenCode, MiMo Code v0.1.3+, Kilo Code CLI**
     → use `"type": "local"`, `"command": [array]`, `"environment": {}`, `"enabled": true`
   - `"mcpServers"` key (Claude Desktop schema): Claude Code, Cursor, Cline, Windsurf, Gemini, Zcode, etc.
3. Use one of these commands (in priority order):
   - `["python", "<absolute_path>/.config/opencode/tools/mathir_mcp/mathir_lib/mathir_mcp_server.py"]` — what `install_smart.py` injects; always works (no `pip install` required)
   - `["python", "-m", "mathir_mcp.mathir_lib.mathir_mcp_server"]` — requires `pip install -e ~/.config/MATHIR/mathir_mcp` to have been run
   - **`["python", "-m", "mathir_mcp"]` does NOT work as an MCP command** — it launches the HTTP daemon, not the MCP stdio server.
4. Add `"environment": {"MATHIR_EMBEDDING_DIM": "384"}` (or `"env"` for the mcpServers schema)
5. Copy `~/.config/MATHIR/GLOBAL_INSTRUCTIONS.md` into your agent's instructions

**If you have no agent, install OpenCode:** https://opencode.ai/

---

### Troubleshooting Integration

| Problem | Fix |
|---------|-----|
| MCP server not showing | Check config key: **OpenCode & MiMo Code use `"mcp"`**, others use `"mcpServers"` |
| `Unrecognized key: "mcpServers"` in mimocode.json | MiMo Code is a fork of OpenCode — rename the key to `"mcp"` and restructure the entry (see MiMo example above) |
| "python not found" | Use full path: `["C:\\Python312\\python.exe", "..."]` |
| Wrong script path | Use one of the 3 commands in §Manual Integration, step 3. The first one (absolute path to `mathir_mcp_server.py`) is what the smart installer injects — that one always works. |
| MCP starts then crashes immediately | Port 7338 already in use by another daemon — kill it (`Stop-Process -Id <pid>`) or set `MATHIR_PORT=7339` in `environment` |
| `ModuleNotFoundError: mathir_mcp` | You ran `pip install -e` from `~/.config/MATHIR/` instead of `~/.config/MATHIR/mathir_mcp/`; reinstall from the inner dir |
| Mathir installs but agent shows no tools | Restart the agent entirely — many agents cache the MCP server list at startup |
| Smart installer reports `Failed to copy MATHIR: [WinError 3]` | Your `~/.config/MATHIR/` is missing a `mathir_mcp/` subdir — see §Restructuring After Clone |
| Agent ignores memory | Ensure GLOBAL_INSTRUCTIONS.md is in instructions |

---

### Multilingual Help (EN/FR/ES/ZH)

**EN:** Give ~/.config/MATHIR/ to your coding agent. It reads INSTALL_FOR_AGENT/AGENT.md and configures automatically.

**FR:** Donnez ~/.config/MATHIR/ a votre agent de code. Il lit INSTALL_FOR_AGENT/AGENT.md et se configure automatiquement.

**ES:** Dea ~/.config/MATHIR/ a su agente de codigo. Leera INSTALL_FOR_AGENT/AGENT.md y se configurara solo.

**ZH:** GEI nide agent ~/.config/MATHIR/, hui du INSTALL_FOR_AGENT/AGENT.md zidong peizhi.

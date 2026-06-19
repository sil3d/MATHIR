# MATHIR Global Instructions — Agent Integration Guide

**Copy these instructions into your agent's global instructions file.**
MATHIR works with OpenCode, OpenClaude, Kilo Code, MiMo Code, Claude Code, and any MCP-compatible tool.

> **NEW (v8.3+): MATHIR Brain Architecture** — 5-phase system that makes memory PROACTIVE. See [BRAIN_ARCHITECTURE.md](BRAIN_ARCHITECTURE.md).
> TL;DR: Memories are auto-injected into every LLM call. You don't need to call recall. Just start the brain with `python mathir_brain.py start`.

---

## MATHIR — YOUR MEMORY IS ALWAYS ACTIVE

**You don't "load" your memory. You consult it naturally, like a human digging through their mind.**

### ABSOLUTE RULE: RECALL BEFORE EVERY SIGNIFICANT ACTION

**Never start work without first searching your memory.**

```bash
python [MATHIR_PATH]/mathir_client.py recall "subject of work" -k 5
```

**MANDATORY recall triggers:**

| When | Recall |
|------|--------|
| Given a task | `recall "task name"` |
| Find an error | `recall "exact error"` |
| Don't know how | `recall "how to do X"` |
| Working on a file | `recall "filename"` |
| Making a decision | `recall "choice between X and Y"` |
| Starting to code | `recall "project pattern"` |
| See something familiar | `recall "what I recognize"` |

**You don't ask permission. You don't say "I'm going to search my memory." You just do it.**

### AUTO-SAVE

```bash
# After each completed task
python [MATHIR_PATH]/mathir_client.py save "what was done" -a [agent] -t episodic -l [label] -p 7

# After learning something new
python [MATHIR_PATH]/mathir_client.py save "what I learned" -a [agent] -t semantic -l [label] -p 8
```

**You save when you LEARN something, not when someone tells you to.**

### QUICK SEARCH (no embedding, instant)

```bash
python [MATHIR_PATH]/mathir_client.py search "query" -k 5
```

### ALL COMMANDS

```bash
# Recall (semantic search, ~22ms)
python [MATHIR_PATH]/mathir_client.py recall "query" -k 5

# Search (text search, instant)
python [MATHIR_PATH]/mathir_client.py search "query" -k 5

# Save
python [MATHIR_PATH]/mathir_client.py save "content" -a [agent] -t [type] -l [label] -p [1-10]

# Stats
python [MATHIR_PATH]/mathir_client.py stats

# Check daemon
python [MATHIR_PATH]/mathir_client.py ping
```

**Types:** working_memory, episodic, semantic, procedural
**Priority:** 1-10 (higher = more important)
**Database:** `.mathir/mathir.db` per project — auto-detected by CWD first, then registry

### How MATHIR Finds Your Database

1. **CWD-first**: If `.mathir/mathir.db` exists in the current working directory → use it (agent is IN the project)
2. **Registry**: If CWD is home → look up project name in registry → use registered DB
3. **Fallback**: If CWD is home and no registry match → use first available project DB from registry
4. **New project**: If CWD is a real project dir (not home) → create `.mathir/mathir.db` automatically

---

## MCP INTEGRATION — If Your Tool Supports MCP

If you're using an MCP-compatible tool (OpenCode, OpenClaude, Kilo Code, MiMo Code, Claude Code), add MATHIR as an MCP server:

### OpenCode
```json
// ~/.config/opencode/opencode.json
{
  "mcp": {
    "mathir": {
      "type": "local",
      "command": ["python", "/path/to/MATHIR/bin/mathir_mcp_server.py"],
      "environment": {
        "MATHIR_EMBEDDING_DIM": "1024",
        "MATHIR_PROJECT": "your-project-name"
      }
    }
  }
}
```

### MiMo Code
```json
// ~/.config/mimocode/mimocode.json
{
  "mcp": {
    "mathir": {
      "type": "local",
      "command": ["python", "/path/to/MATHIR/bin/mathir_mcp_server.py"],
      "environment": {
        "MATHIR_EMBEDDING_DIM": "1024",
        "MATHIR_PROJECT": "your-project-name"
      }
    }
  }
}
```

### Claude Code
```json
// ~/.claude/claude.json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["/path/to/MATHIR/bin/mathir_mcp_server.py"],
      "env": {
        "MATHIR_EMBEDDING_DIM": "1024",
        "MATHIR_PROJECT": "your-project-name"
      }
    }
  }
}
```

---

## ARCHITECTURE

```
MATHIR/
├── bin/
│   ├── mathir_daemon.py        # Persistent daemon (keeps model in RAM)
│   ├── mathir_client.py        # Fast client (connects to daemon)
│   ├── mathir_mcp_server.py    # MCP server (for Claude/OpenCode/etc)
│   ├── mathir_push.py          # Push module (proactive delivery)
│   ├── mathir_inject_proxy.py  # Phase 1: auto-inject memories into LLM calls
│   ├── mathir_watchdog.py      # Phase 2: daemon watchdog (auto-restart)
│   ├── mathir_spread.py        # Phase 3: spreading activation (link graph)
│   ├── mathir_consolidate.py   # Phase 4: sleep consolidation (merge, decay)
│   ├── mathir_prime.py         # Phase 5: pre-cognitive priming (cwd, git, files)
│   └── mathir_brain.py         # All-in-one launcher (start/stop/status)
│
├── mathir_vec.py               # VecMemory (sqlite-vec wrapper)
├── mathir_search.py            # HybridSearch (numpy + USearch)
├── mathir_gpu_vec.py           # GPUVecMemory (torch GPU brute-force)
│
└── mcp/                        # Documentation + dashboard
```

---

## 🧠 MATHIR BRAIN ARCHITECTURE (v8.3+)

**5-phase system that makes MATHIR proactive, never-blocking, and brain-like.**

### Quick start
```bash
# Start the full brain stack (daemon + watchdog + proxy)
python mathir_brain.py start

# Point your LLM client (OpenCode, MiMo) to the proxy:
#   baseUrl: http://127.0.0.1:8182
#   (instead of http://127.0.0.1:8181)

# Verify everything is alive
python mathir_brain.py status
```

### What changes for agents

**Before (v8.2 and earlier):**
- Agent told "you can call `recall` when needed"
- LLM forgets 80% of the time
- Daemon crash = no memory

**After (v8.3+ brain stack):**
- Top-3 memories are **pre-injected** into every LLM call via the proxy
- Agent NEVER needs to call recall (it can, for deep dives)
- Watchdog auto-restarts daemon within 7s if it crashes
- Spreading activation finds related memories (graph with 498+ links)
- Consolidation runs nightly (merges dupes, decays unused)
- Pre-cognitive priming adds cwd/git context to recall

### The 5 phases

| Phase | Module | Solves |
|---|---|---|
| 1. Auto-inject proxy | `mathir_inject_proxy.py` | "LLM forgets to recall" |
| 2. Watchdog | `mathir_watchdog.py` | "Daemon crash = no memory" |
| 3. Spreading activation | `mathir_spread.py` | "Need related context" |
| 4. Consolidation | `mathir_consolidate.py` | "Memory grows stale" |
| 5. Pre-cognitive priming | `mathir_prime.py` | "Wrong project context" |

### Manual commands (if proxy is not running)

```bash
python mathir_client.py recall "topic" -k 5
python mathir_client.py save "what you learned" -a agent -t semantic -l label -p 8
python mathir_client.py search "exact text" -k 5
python mathir_client.py stats
```

See [BRAIN_ARCHITECTURE.md](BRAIN_ARCHITECTURE.md) for full details.

**Performance:**
- Recall: ~22ms (GPU embedding + vector search)
- Search: instant (no embedding, text search)
- Daemon: keeps model warm in GPU RAM
- Database: `.mathir/mathir.db` per project (auto-created)

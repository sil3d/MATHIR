# MATHIR — Global Instructions (v8.5.0)

## ✅ MATHIR IS INJECTED — You Have Persistent Memory

**MATHIR (Memory Architecture for Tiered Heuristic Intelligent Retrieval)** is a 5-tier cognitive memory system that gives you persistent memory across sessions. It is **automatically injected** into every session via MCP tools.

**What MATHIR does:**
- Remembers decisions, bugs, fixes, and knowledge across sessions
- Auto-decays unused memories, promotes frequently-used ones
- Links related memories in a graph for spreading activation
- Works across all 32 agents (swarm, coder, debugger, etc.)

---

## 🚨 MANDATORY — Session Start Protocol

**You MUST call `memory_session_start` at the START of every session.** This returns the most relevant memories for your current context.

```
memory_session_start(session_title="what this session is about")
```

**Then** call `memory_context` before each major task:
```
memory_context(task="description of what you're about to do")
```

**After completing work**, save what you learned:
```
memory_save(content="what you learned", agent="your_name", block_type="episodic", label="short-label")
```

---

## How to Use (3 Steps)

1. **Session start** — `memory_session_start(session_title="...")` → get context
2. **Before each task** — `memory_context(task="...")` → get relevant memories
3. **After each task** — `memory_save(content="...", agent="...", block_type="episodic", label="...")`

---

## Tool Signatures

### Auto-injection (v8.5.0 — call these FIRST)

```
memory_session_start(session_title: str = "", project: str = None) -> dict
  # Returns: relevant_memories, stats, instruction
  # Call at session start with a brief title of what you're working on

memory_context(task: str, project: str = None) -> dict
  # Returns: memories grouped by tier (semantic, episodic, procedural, working_memory)
  # Call before each major task with a description of what you're about to do
```

### Basic (every day)

```
memory_save(content: str, agent: str, block_type: str, label: str, priority: int = 5)
memory_recall(query: str, k: int = 5, agent: str = None)
memory_smart_search(query: str, k: int = 10)
memory_hybrid_search(query: str, k: int = 5, alpha: float = 0.5)
memory_audit(agent: str = None, limit: int = 50)
memory_export()
memory_delete(memory_id: str, reason: str = "user requested")
memory_sessions(limit: int = 10)
memory_stats()
memory_dashboard(action: str = "status")
```

### Lifecycle (v8.4.0 — living memory)

```
memory_promote(memory_id: str = None, force: bool = False)
memory_auto_promote()
memory_decay(threshold_days: int = 30, archive_floor: float = 0.05)
memory_consolidate(threshold: float = 0.95, dry_run: bool = False, limit: int = 1000)
memory_link(source_id: str, target_id: str, weight: float = 1.0)
memory_get_links(memory_id: str, depth: int = 2, decay: float = 0.5)
memory_build_links(threshold: float = 0.7, limit: int = 1000)
```

**19 tools total** (2 auto-injection + 10 basic + 7 lifecycle).

**block_type:** `working_memory` | `episodic` | `semantic` | `procedural`
**priority:** 1–10 (see scale below)

---

## When to Save

- After completing a task
- After learning something new
- After fixing a bug
- After making an architectural decision
- After discovering a reusable pattern

Save when you **learn** something, not when told to.

---

## When to Recall

- Before starting any task
- When you see something familiar
- When debugging an error
- When making a decision

You don't ask permission. You just do it.

---

## Memory Types

| Type | When to Use | Example |
|------|-------------|---------|
| `working_memory` | Current task context | "Fixing auth bug in login.py" |
| `episodic` | Past actions | "Refactored user service on 2026-06-19" |
| `semantic` | Knowledge / facts | "Project uses React + TypeScript" |
| `procedural` | How-to instructions | "Run tests with `pytest -x`" |

---

## Priority Scale

1 = low, 5 = normal, 8 = important, 10 = critical.

---

## Rules

- **No hardcoded paths.** Do NOT mention `D:\`, `C:\`, or any machine-specific locations.
- The MCP server auto-detects the project database — nothing to configure.

---

## 🚀 Cross-Platform Auto-Start (v8.4.2+)

The MATHIR daemon runs on port 7338 and must be started after every PC reboot. Three cross-platform helpers are shipped:

| Platform | Helper file | Install (optional, boot-time) |
|---|---|---|
| **Windows** | `bin/auto_start.bat` / `bin/auto_start_helpers.ps1` | Put a shortcut to `auto_start.bat` in `shell:startup` |
| **Linux** | `bin/auto_start.sh` / systemd `bin/mathir-daemon.service` | `systemctl --user enable mathir-daemon` |
| **macOS** | `bin/auto_start.sh` / launchd `bin/com.mathir.daemon.plist` | `launchctl load -w ~/Library/LaunchAgents/com.mathir.daemon.plist` |

**Source repo:** `D:\SECRET_PROJECT\MATHIR\mathir_mcp\bin\`
**Deployed:** `~/.config/opencode/bin/`
**Full install guides:** `mathir_mcp/INSTALL/INSTALL_{WINDOWS,LINUX,MACOS}.md`

If the user asks to install, set up, or troubleshoot auto-start — point them to the matching `INSTALL_*.md`.

---

## Input Limits & Security

The MCP server enforces per-field length caps to prevent DoS via unbounded payloads:

| Field | Default cap |
|-------|-------------|
| `content` (memory_save) | 100 KB |
| `query` (memory_recall) | 5 KB |
| `label` | 200 B |
| `agent` | 100 B |

Tune with the `MCP_INPUT_MAX` env var (multiplier — `MCP_INPUT_MAX=2.0` doubles all caps). Out-of-range values fall back to default. Rejected payloads return `{"error": "<field> exceeds <cap> chars"}`.
- Database lives at `.mathir/mathir.db` per project, created automatically on first write.

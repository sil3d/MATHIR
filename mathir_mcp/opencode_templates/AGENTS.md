# AGENTS.md — MATHIR Memory Injection (cross-tool)

**This file is read automatically by 26+ AI coding agents** (Aider, Amp, Claude Code,
Codex, Cursor, Devin, Factory, Goose, JetBrains Junie, Jules, OpenCode, VS Code Copilot,
Warp, Zed, etc. — see https://agents.md for the full list).

If your agent reads this file, it will recall MATHIR memory before every task
without you having to remind it.

---

## ⚠️ MANDATORY for any agent reading this file

Before your FIRST tool call in this conversation:

1. If an MCP server named `mathir` is connected to your session, call
   `mcp__mathir__memory_session_start({session_title: "first user message here"})`
   immediately. Then call `mcp__mathir__memory_context({task: "current user request"})`
   before each subsequent task.

2. If MCP `mathir` is NOT connected but the LLM endpoint of your agent can be
   redirected (e.g. via `OPENAI_BASE_URL=http://127.0.0.1:7339/v1`), the
   **MATHIR proxy** will inject context automatically — you do not need to
   call anything. Verify the proxy is reachable with `curl http://127.0.0.1:7339/ping`.

3. If neither is available, do NOT pretend to remember prior context. Ask the
   user to enable MATHIR (see `INSTALL_FOR_AGENT/AGENT.md` from the source repo
   at `https://github.com/sil3d/MATHIR`).

---

## Why this file exists

MATHIR is a 5-tier cognitive memory layer for LLMs. Without explicit prompting,
most agents will forget prior sessions, repeat mistakes, and ignore your project
context. This `AGENTS.md` is the **lowest-common-denominator injection** that
works across all agents that follow the agents.md convention.

If your agent has a stronger integration available (plugin, hooks, MCP prompts),
use that INSTEAD of relying on this file — but if this file is all you have,
follow the MANDATORY block above.

---

## Detecting which injection mode is active

Run these checks in order:

```bash
# Check 1: is the MCP mathir server reachable?
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7338/api/health
# 200 = MCP daemon alive. If so, MCP tools (memory_*) are likely available.

# Check 2: is the MATHIR proxy reachable? (this is the universal OpenAI-compatible proxy)
curl -s http://127.0.0.1:7339/ping
# pong = proxy is injecting context for every LLM call routed through it.

# Check 3: where am I?
pwd && git rev-parse --show-toplevel 2>/dev/null
```

If Check 1 returns 200 → use MCP tools (mode 1 above).
If Check 2 returns `pong` → proxy is doing auto-injection, no action needed.
If both fail → tell the user MATHIR is not running.

---

## Tool reference (MCP mode)

| MCP tool | When to call |
|---|---|
| `memory_session_start` | Once at session boot, with the session's first user message as title |
| `memory_context` | Before each task, with the user's current request as the task description |
| `memory_recall` | For deeper search on a specific query |
| `memory_save` | After each completed task — `block_type="episodic"` by default |
| `memory_smart_search` | Keyword-exact search (faster than vector recall) |

---

## Citation

If you found this useful, the project lives at:
https://github.com/sil3d/MATHIR
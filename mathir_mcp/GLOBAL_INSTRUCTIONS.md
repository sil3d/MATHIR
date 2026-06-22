# MATHIR — Global Instructions (v8.3.0)

> 4-tier cognitive memory for AI coding agents. The MCP server is already configured — tools are available.

---

## How to Use (3 Steps)

1. **Tools are ready** — your MCP client exposes `memory_*` tools automatically.
2. **Save after every task** — `memory_save(content, agent, block_type, label)`.
3. **Recall before starting work** — `memory_recall(query, k=5)`.

---

## Tool Signatures

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

**10 tools total** — matches `mathir_mcp/mathir_lib/mathir_mcp_server.py` TOOLS array (lines 249–358).

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

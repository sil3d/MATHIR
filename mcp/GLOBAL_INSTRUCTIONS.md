# MATHIR — Global Instructions

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
memory_hybrid_search(query: str, k: int = 5)
memory_stats()
memory_delete(memory_id: str)
```

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
- Database lives at `.mathir/mathir.db` per project, created automatically on first write.

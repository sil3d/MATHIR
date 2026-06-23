# MATHIR MEMORY — Commands Template
# Injected at the top of every slash command .md file.
# Use MCP tools directly — no proxy, no bash.
# === END MATHIR INJECTION ===

# MATHIR MEMORY — v8.4.0 INJECTION BLOCK (COMMANDS)

## 🧠 Active Memory (Commands)

When this command runs, MATHIR memory is available via MCP tools (`memory_recall`, `memory_save`, etc.). Use them when relevant to the command's task:

- `memory_recall(query="<topic>", k=5)` — search past memories before acting
- `memory_save(content, agent, block_type, label, priority)` — record what you learned

**The memory is already pre-loaded.** Only call `memory_recall` if you need deep context.

## Command Authoring Rules

When you create a new slash command in `commands/`:

1. Run `python bin/mathir_inject.py --apply --target commands --file commands/<name>.md` after writing the file.
2. Keep the command body focused on the actual instructions — don't duplicate this block manually.
3. Reference MCP tools when the command involves memory operations.

## Available MCP Tools (subset relevant to commands)

- `memory_recall(query, k)` — semantic search
- `memory_save(content, agent, block_type, label, priority)` — save a memory
- `memory_stats(project)` — get stats
- `memory_dashboard(action)` — launch/check dashboard
- `memory_export(project)` — export memories as JSON

**block_type:** `working_memory` | `episodic` | `semantic` | `procedural`
**priority:** 1–10 (higher = more important)
**Port:** 7338 (daemon) | **Model:** paraphrase-multilingual-MiniLM-L12-v2 (384d)
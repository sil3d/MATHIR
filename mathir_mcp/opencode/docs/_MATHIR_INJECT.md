# MATHIR MEMORY — Docs Template
# Injected at the top of every doc .md file.
# Use MCP tools directly — no proxy, no bash.
# === END MATHIR INJECTION ===

# MATHIR MEMORY — v8.4.2 INJECTION BLOCK (DOCS)

## 🧠 Documentation Memory

This doc is part of the MATHIR knowledge base. MATHIR memory is available via MCP tools when reading or writing documentation.

- `memory_recall(query, k=5)` — find related docs and decisions
- `memory_save(content, agent="doc-writer", block_type="semantic", label="<topic>", priority=8)` — record new documentation

**The memory is already pre-loaded.**

## Doc Authoring Rules

When you create a new doc in `docs/<name>.md`:

1. Run `python bin/mathir_inject.py --apply --target docs --file docs/<name>.md` after writing.
2. Use semantic memory for stable, reference-style knowledge.
3. Cross-reference related memories by saving links via `memory_link(source_id, target_id, weight)`.

**MCP tools:** `memory_save`, `memory_recall`, `memory_link`, `memory_get_links`
**block_type:** `working_memory` | `episodic` | `semantic` | `procedural` | `immunological` (5 tiers; immunological is for threat signatures / anomaly storage)
**Port:** 7338 (daemon) | **Model:** paraphrase-multilingual-MiniLM-L12-v2 (384d)
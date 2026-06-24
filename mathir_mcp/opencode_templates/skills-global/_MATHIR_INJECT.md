# MATHIR MEMORY — Skills-Global Template
# Same as skills/ template, but for skills-global/.
# Use MCP tools directly — no proxy, no bash.
# === END MATHIR INJECTION ===

# MATHIR MEMORY — v8.4.2 INJECTION BLOCK (SKILLS-GLOBAL)

## 🧠 Active Memory (Skills-Global)

When a global skill is loaded, MATHIR memory is available via MCP tools. These skills are shared across all OpenCode agents and projects.

- `memory_recall(query, k=3)` — search for prior knowledge
- `memory_save(content, agent, block_type, label, priority=8)` — record discoveries

**The memory is already pre-loaded.**

## Skill Authoring Rules

When you create a new global skill in `skills-global/<name>/SKILL.md`:

1. Run `python bin/mathir_inject.py --apply --target skills-global --file skills-global/<name>/SKILL.md` after writing.
2. Use semantic memory for stable, cross-project knowledge.
3. Global skills apply broadly — keep the body portable (no hardcoded paths).

**MCP tools:** `memory_save`, `memory_recall`, `memory_stats`
**block_type:** `working_memory` | `episodic` | `semantic` | `procedural` | `immunological` (5 tiers; immunological is for threat signatures / anomaly storage)
**Port:** 7338 (daemon) | **Model:** paraphrase-multilingual-MiniLM-L12-v2 (384d)
# `opencode_templates/` — MATHIR injection templates for OpenCode

> **This folder contains the SOURCE TEMPLATES for MATHIR's memory injection system, specifically designed for OpenCode and OpenCode-compatible tools (MiMo Code, Zcode, etc.).**

It is NOT the actual runtime config — that lives in the user's `~/.config/opencode/` and `~/.config/mimocode/` after `mathir_inject.py` runs.

## What's here

```
opencode_templates/
├── README.md                       ← you are here
├── agents/
│   └── _MATHIR_INJECT.md           ← full block, ~9.4 KB / 237 lines
├── commands/
│   └── _MATHIR_INJECT.md           ← short block, ~2.4 KB
├── skills/
│   └── _MATHIR_INJECT.md           ← minimal block, ~1.6 KB
├── skills-global/
│   └── _MATHIR_INJECT.md           ← minimal block, ~1.2 KB
└── docs/
    └── _MATHIR_INJECT.md           ← reference footer, ~1.2 KB
```

## What this is vs what it isn't

| ✅ This folder (`opencode_templates/`) | ❌ NOT this |
|---|---|
| Canonical source-of-truth templates | The user's runtime config (`~/.config/opencode/`) |
| Versioned with the code on GitHub | Their local injected agents/commands/skills |
| Edited to change the injection block | Edited to change user-specific settings |
| Shipped to all MATHIR users | Specific to one user/machine |

The 5 `_MATHIR_INJECT.md` files here are the **source of truth** for what gets injected. When you edit them and run `mathir_inject.py --apply`, the changes propagate to:
- `~/.config/opencode/agents/*.md` (32 agents)
- `~/.config/opencode/commands/*.md` (10 commands)
- `~/.config/opencode/skills/*/SKILL.md` (68 skills)
- `~/.config/opencode/skills-global/*/SKILL.md` (88 skills-global)
- `~/.config/mimocode/...` (same structure for MiMo Code)

If you fork MATHIR and want to customize the injection block, edit the templates here, then re-run `mathir_inject.py --apply --target all` to push the changes.

---

## What's in this folder

```
opencode/
├── README.md                       ← you are here
├── agents/
│   └── _MATHIR_INJECT.md           ← full block, ~9.4 KB / 237 lines
├── commands/
│   └── _MATHIR_INJECT.md           ← short block, ~2.4 KB
├── skills/
│   └── _MATHIR_INJECT.md           ← minimal block, ~1.6 KB
├── skills-global/
│   └── _MATHIR_INJECT.md           ← minimal block, ~1.2 KB
└── docs/
    └── _MATHIR_INJECT.md           ← reference block, ~1.2 KB
```

**Five templates, one per *kind* of file** that OpenCode loads at runtime:

| Template | Size | When it fires | Purpose |
|---|---:|---|---|
| `agents/_MATHIR_INJECT.md` | 9.4 KB | Every sub-agent system prompt | Full block: daemon health check, recall workflow, 17 MCP tools, save tiers, lifecycle housekeeping, error rules |
| `commands/_MATHIR_INJECT.md` | 2.4 KB | Slash-command system prompts | Shortened block: just the recall + save cheatsheet, no housekeeping |
| `skills/_MATHIR_INJECT.md` | 1.6 KB | `SKILL.md` frontmatter in `~/.config/opencode/skills/` | Minimal: "memory is pre-loaded; save non-obvious things" |
| `skills-global/_MATHIR_INJECT.md` | 1.2 KB | `SKILL.md` frontmatter in `~/.config/opencode/skills-global/` | Same as `skills/`, kept separate so skills-global can drift later |
| `docs/_MATHIR_INJECT.md` | 1.2 KB | Markdown docs loaded as `instructions` in `opencode.json` | Reference card, not for action |

The block sizes are deliberate: agents get the whole cognitive architecture (it's their job to use it); slash commands and skills get a tight reference; docs get a footer-style mention.

---

## The 5 `_MATHIR_INJECT.md` files, in detail

### 1. `agents/_MATHIR_INJECT.md` — the canonical block

This is the block that gets injected into every one of the 32 agents (`@coder`, `@debugger`, `@swarm`, …). It contains:

- **Daemon health check** — the agent's first instruction is `Test-NetConnection -ComputerName localhost -Port 7338`. If the daemon is down, the agent is told to start it, not assume MCP is broken.
- **Active memory auto-injection slot** — `{{MATHIR_CONTEXT}}`, filled in at session start with the top-K most relevant prior memories.
- **5-tier model recap** — `working_memory` / `episodic` / `semantic` / `procedural` / `immunological`, with the rule "start with `episodic`".
- **17 MCP tools** — basic CRUD, lifecycle (`promote`, `decay`, `consolidate`, `link`, `build_links`), and end-of-session housekeeping commands.
- **End-of-session recipe** — the exact `memory_auto_promote()` / `memory_decay()` / `memory_consolidate()` / `memory_build_links()` sequence every agent runs before exiting.
- **Error rules** — no "pre-existing error" excuses, no commented-out code, no hardcoded paths.
- **Rule 0.5 / 0.6** — search for existing systems first, test with the real stack (no mocks).

### 2. `commands/_MATHIR_INJECT.md` — slash-command variant

Slash commands (`/mathir_inject`, `/commit`, etc.) are short-lived and don't run multi-turn workflows. The block is trimmed to:

- Confirm the daemon is up.
- Recall the 2-3 most relevant memories for the command's task.
- Save the result at the end.

No lifecycle, no housekeeping, no 17-tool catalog.

### 3. `skills/_MATHIR_INJECT.md` — minimal variant

Skills (`SKILL.md` files in `~/.config/opencode/skills/<name>/SKILL.md`) are loaded into the agent's context as needed. The block tells the agent:

- Memory is already pre-loaded — you typically don't need to call recall.
- Save a `semantic` or `procedural` memory when you discover something non-obvious while running the skill.

### 4. `skills-global/_MATHIR_INJECT.md` — same as skills

Kept as a separate file so `skills-global/` can diverge from `skills/` later (e.g. global skills often need a *different* recall discipline than per-project skills). Today the two are nearly identical.

### 5. `docs/_MATHIR_INJECT.md` — reference footer

Loaded as `instructions` in `opencode.json` for documentation-heavy projects. Acts as a footer reminding the agent that MATHIR exists; doesn't try to teach the agent how to use it (the agent template already does that).

---

## How `mathir_inject.py` uses these templates

The script lives at `mathir_lib/mathir_inject.py`. The high-level algorithm:

```python
TARGETS = {
    "agents":        ("agents",        "*.md"),
    "commands":      ("commands",      "*.md"),
    "skills":        ("skills",        "*/SKILL.md"),
    "skills-global": ("skills-global", "*/SKILL.md"),
    "docs":          ("docs",          "*.md"),
}

# 1. find_config_root(): look in OPENCODE_CONFIG env, then ~/.config/opencode,
#    then ..  (for in-tree dev). The user config, not the repo, is the target.

# 2. load_template(target, config_root):
#    - read <config_root>/<target_subdir>/_MATHIR_INJECT.md
#    - fall back to <config_root>/_MATHIR_INJECT.md
#    - fall back to <config_root>/agents/_MATHIR_INJECT.md (back-compat)
#    - extract just the actual block (skip meta-comments above "# MATHIR MEMORY — v")

# 3. discover_targets(config_root, target):
#    - glob <config_root>/<target_subdir>/<pattern>
#    - skip README.md and the _MATHIR_INJECT.md files themselves

# 4. process_file(path, template):
#    - read file
#    - extract YAML frontmatter (keep at the top)
#    - detect existing block via anchor patterns: "# MATHIR MEMORY — v", etc.
#    - find block end via "# === END MATHIR INJECTION ===" → standard
#      "Model: paraphrase-multilingual-MiniLM-L12-v2" → heuristic heading
#    - if existing block is content-equivalent to template, skip (idempotent)
#    - else: strip old block, insert new block after frontmatter
#    - write file

# 5. Exit codes: 0 = success, 1 = error, 2 = --check found stale/missing files
```

**Idempotent:** running it twice is safe. `_content_equivalent()` normalizes whitespace and blank lines before comparing, so a no-op template change still gets the file rewritten, but an *unchanged* template is detected as a no-op.

**Anchor patterns** (the script finds the existing block by looking for these):
- `# MATHIR MEMORY — v` (em-dash, the canonical form)
- `# MATHIR MEMORY - v` (hyphen, for editors that mangle unicode)
- `# MATHIR MEMORY v`
- `# MATHIR MEMORY`

The block ends at the first occurrence of `# === END MATHIR INJECTION ===`, or the standard end-marker `**Model:** paraphrase-multilingual-MiniLM-L12-v2`, or — as a last resort — the first `## AGENT` / `## ROLE` / `## YOU ARE` / etc. heading that comes after line 5 of the block.

---

## Target directories and injection behavior

The "user config root" is resolved in this order:

1. `$OPENCODE_CONFIG` env var, if set and points to a directory.
2. `~/.config/opencode/` (Windows: `C:\Users\<YOU>\.config\opencode\`).
3. The parent of the script's own directory, *only* if it contains an `opencode.json` (this is the in-tree dev fallback).

Inside that root, the script touches exactly these paths:

| Target | Pattern | Skipped |
|---|---|---|
| `agents/` | `*.md` (32 files) | `README.md`, `_MATHIR_INJECT*.md` |
| `commands/` | `*.md` (9 files) | same |
| `skills/` | `*/SKILL.md` (68 files) | same |
| `skills-global/` | `*/SKILL.md` (88 files) | same |
| `docs/` | `*.md` | same |

The repo's own `opencode/` folder only contains the **templates** — not the 32 agents, 9 commands, 68 skills, 88 global skills, or docs. Those are user-owned files that get created in *their* `~/.config/opencode/` (typically by `mathir_sync.py` copying them from this repo, or by hand).

**Why the split?** The template is a *shared, versioned* artifact — every user should get the same MCP cheatsheet, daemon health check, and error rules. The agents, skills, and commands are *per-user* artifacts — they have your colors, your model choices, your custom agents. Keeping the template in the repo and the per-user files in `~/.config/opencode/` lets us update the template in one place and re-inject into all users.

---

## Common workflows

### You just created a new agent in the repo

```bash
# 1. Copy the new agent to the user config
python mathir_lib/mathir_sync.py --file opencode/<not here — see mathir_sync.py>

# 2. Inject the MATHIR block into the one file
python mathir_lib/mathir_inject.py --apply --file ~/.config/opencode/agents/<name>.md
```

### You edited `agents/_MATHIR_INJECT.md` and want to push the new block to all agents

```bash
# Dry run first — see what would change
python mathir_lib/mathir_inject.py --check --target agents

# Apply
python mathir_lib/mathir_inject.py --apply --target agents
```

### You edited all 5 templates and want a full repo-wide refresh

```bash
python mathir_lib/mathir_inject.py --apply --target all
```

This re-injects 32 agents + 9 commands + 68 skills + 88 global skills + docs in one pass. Idempotent — files that already match the template are left untouched.

### You want to know what's available

```bash
python mathir_lib/mathir_inject.py --list
# Output:
#   agents         agents              files=32   template=yes
#   commands       commands            files=9    template=yes
#   skills         skills              files=68   template=yes
#   skills-global  skills-global       files=88   template=yes
#   docs           docs                files=N    template=yes
```

---

## Editing rules

1. **Edit the template, not the agent files.** If you change the agent's body directly, your change is invisible to the next `mathir_sync.py` import. The template is the source of truth.
2. **Keep YAML frontmatter at the top.** The script preserves it. If you add a new key, put it between `---` markers, above the injected block.
3. **End the template with `# === END MATHIR INJECTION ===`.** This is the most reliable block-end marker; the script's other heuristics exist only as fallbacks.
4. **Don't edit a file's existing block by hand.** Run `mathir_inject.py --apply --file <path>` instead. Manual edits to the injected block are silently overwritten on the next sync.
5. **Keep blocks cross-platform.** The block is identical on Windows, macOS, and Linux. Don't put `C:\`-style paths in the template — those rot when the same user moves to WSL or another machine. Use `~` (which OpenCode expands) or `os.path.expanduser()`-style examples in code snippets.

---

## Cross-platform install

This repo is platform-agnostic. The `bin/` folder ships one autostart unit per OS family:

| File | Targets | Used by |
|---|---|---|
| `bin/mathir-daemon.service` | systemd user unit (Linux) | `INSTALL/INSTALL_LINUX.md` |
| `bin/com.mathir.daemon.plist` | launchd LaunchAgent (macOS) | `INSTALL/INSTALL_MACOS.md` |
| (no file) | Windows uses Task Scheduler — see guide | `INSTALL/INSTALL_WINDOWS.md` |

The MCP server registration in `~/.config/opencode/opencode.json` is **identical** on all three platforms — it uses `~`-relative paths that OpenCode expands at runtime.

See `../INSTALL/INSTALL_{WINDOWS,LINUX,MACOS}.md` for full step-by-step instructions.

---

## Related scripts

| Script | Role |
|---|---|
| `mathir_lib/mathir_inject.py` | This system — inject templates into user config |
| `mathir_lib/mathir_sync.py` | Copy new files from the source repo into the user config (pair this with inject) |
| `mathir_lib/mathir_daemon.py` | The long-running background process that holds the embedding model |
| `mathir_lib/mathir_mcp_server.py` | The stdio MCP server that OpenCode launches per session (connects to the daemon) |
| `install_smart.py` | One-shot installer for the *first* setup: detects agents and injects config |

Typical first-time flow:

```bash
# (one-time) install the daemon + MCP server + opencode.json block
./install.sh        # or install.bat on Windows

# (per-repo change) bring in a new skill from the source repo
python mathir_lib/mathir_sync.py --target skills --file skills/<name>/SKILL.md
python mathir_lib/mathir_inject.py --apply --target skills --file ~/.config/opencode/skills/<name>/SKILL.md
```

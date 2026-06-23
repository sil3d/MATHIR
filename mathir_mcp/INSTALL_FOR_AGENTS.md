# MATHIR — Install Guide for AI Agent Hosts

**Read this if you are an AI agent host (OpenCode, Claude Code, Cursor, MiMo, Kilo Code, etc.) and you want to integrate MATHIR into YOUR platform — with YOUR agents, YOUR subagents, and YOUR tools.**

This is the **end-to-end integration doc**. It assumes you are an LLM agent (or a tool/automation) that needs to:
1. Register MATHIR as an MCP server
2. Make the 17 memory tools available to all your subagents
3. Ensure all subagents know HOW and WHEN to use the tools
4. Verify the integration works end-to-end

---

## The 4 layers of integration

MATHIR integration has 4 distinct layers. **All 4 must be set up correctly** — skipping any one will cause the integration to fail or be underutilized.

```
Layer 4: Verification         (does the integration actually work?)
   ^
Layer 3: Instructions        (do subagents know HOW to use the tools?)
   ^
Layer 2: Subagent tools      (can subagents CALL the tools?)
   ^
Layer 1: MCP server config   (does the host know MATHIR exists?)
```

### Layer 1: MCP server config

Register MATHIR as an MCP server in your platform's config file. The host will spawn `python -m mathir_mcp` when an agent session starts.

**OpenCode** (`~/.config/opencode/opencode.json`):
```json
{
  "mcp": {
    "mathir": {
      "type": "local",
      "command": ["python", "-m", "mathir_mcp"],
      "environment": {
        "MATHIR_EMBEDDING_DIM": "384",
        "MATHIR_PORT": "7338"
      },
      "enabled": true
    }
  }
}
```

**Claude Code** (`~/.claude.json`):
```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["-m", "mathir_mcp"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

**Cursor** (`~/.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "mathir": {
      "command": "python",
      "args": ["-m", "mathir_mcp"],
      "env": { "MATHIR_EMBEDDING_DIM": "384" }
    }
  }
}
```

**MiMo / Kilo Code / others**: same pattern, see `docs/AGENT.md` for the full list.

### Layer 2: Make tools available to subagents

**This is automatic** for most platforms if Layer 1 is correct — the host inherits MCP tools into all subagents.

**Verify** with:
```bash
python -m mathir_mcp --list-tools
# Should show 17 tools
```

If your subagents (Task tool, subagent invocations, agent spawning) **don't** automatically inherit MCP tools, you need to inject them manually. For OpenCode, check:
- Subagent config in `~/.config/opencode/agents/<subagent>.md` has the MCP tool enabled
- The host's permission system allows the tools

### Layer 3: Inject instructions

**This is the layer most often missed.** Without instructions, your agents have the tools but don't know:
- WHEN to save (after every task)
- WHEN to recall (before starting work)
- Which `block_type` to use (4 tiers)
- That the lifecycle exists (promote/decay/consolidate)

**For OpenCode / Claude Code / Cursor**: load `GLOBAL_INSTRUCTIONS.md` into the host's global instructions.

**OpenCode** (`~/.config/opencode/GLOBAL_INSTRUCTIONS.md`):
- Already supported — the host reads this file on every session
- See `mathir_mcp/GLOBAL_INSTRUCTIONS.md` for the full content (copy it)

**Claude Code**: add to `~/.claude.json` under `"systemPrompt"` or use a CLAUDE.md file in the project.

**Cursor** (`.cursorrules` in project root):
```
# MATHIR memory rules
- Before starting any task, call memory_recall(query=...)
- After learning something new, call memory_save(content=..., block_type="episodic", ...)
- See mathir_mcp/ONBOARDING.md for the full playbook
```

**For subagents specifically**: the host must propagate the instructions to spawned subagents. For OpenCode, this happens automatically if the subagent file references the global instructions. For Claude Code, you may need to pass the instructions in the subagent's system prompt explicitly.

### Layer 4: Verification

**The only way to know it works is to run a subagent that uses the tools.**

**Test script** (run from any shell):
```python
# spawn_subagent_and_test.py
import subprocess
import json

# 1. Spawn a subagent that uses memory_save
prompt = """
You have access to a memory_save tool. Run this command and report the result:
memory_save(content="MATHIR integration test", agent="installer", block_type="episodic", label="integration-test")
Then report the memory_id.
"""

result = subprocess.run(
    ["python", "-m", "mathir_mcp", "--selftest"],
    capture_output=True, text=True, timeout=120
)
print("SELFTEST:", result.stdout)
assert "9 passed" in result.stdout, "Selftest failed"
print("Integration verified")
```

**Manual verification** (in OpenCode):
```
Ask the agent: "List all your MCP tools and call memory_recall(query='test', k=1)"
If the agent can list 17 tools AND can call memory_recall -> integrated
If the agent says "I don't see memory tools" -> Layer 1 is broken
If the agent sees tools but doesn't know when to use them -> Layer 3 is broken
```

---

## The 17 tools your agents will have

After integration, all your agents and subagents get:

| Tool | When to use |
|------|-------------|
| `memory_save` | Save a memory (4 tiers) |
| `memory_recall` | Semantic search (auto-touches stability) |
| `memory_smart_search` | Fast daemon-native search |
| `memory_hybrid_search` | Vector + BM25 + RRF |
| `memory_audit` | View audit trail |
| `memory_export` | Export to JSON |
| `memory_delete` | Soft delete (tier='archived') |
| `memory_sessions` | List sessions |
| `memory_stats` | Totals by tier/agent |
| `memory_dashboard` | Launch web UI |
| `memory_promote` | Move to next tier (Ebbinghaus) |
| `memory_auto_promote` | Scan + promote all eligible |
| `memory_decay` | Apply Ebbinghaus decay |
| `memory_consolidate` | Merge near-duplicates |
| `memory_link` | Add link graph edge |
| `memory_get_links` | BFS graph traversal |
| `memory_build_links` | Build graph |

---

## Subagent propagation patterns

Different platforms handle subagent tool inheritance differently. Here's what to watch for:

### Pattern A: Tools inherited automatically (OpenCode, Claude Code)

The host registers MCP at the top level, all subagents inherit. **Verify with a test subagent** — if it can't call `memory_recall`, check the host's permission config.

### Pattern B: Tools must be re-declared per subagent (some platforms)

Some platforms require you to list which tools each subagent can use. If your subagent config has a `tools:` or `allowed_tools:` section, make sure `memory_*` is in there.

Example fix:
```yaml
# agent-config.yaml
agent:
  name: coder
  tools:
    - memory_save
    - memory_recall
    - memory_promote
    - memory_decay
    - memory_consolidate
    # ... all 17 memory tools
```

### Pattern C: No subagent support (some lightweight hosts)

If your host has no subagent concept, you only need Layer 1 + Layer 3 (the main agent has the tools and instructions).

---

## Daemon lifecycle in the host context

The MCP server (`python -m mathir_mcp`) starts a daemon on first call. But the daemon lifecycle has 3 states your host needs to handle:

| State | What it means | What the host should do |
|-------|---------------|------------------------|
| **Not running** | No daemon on port 7338 | MCP server starts it on first call (auto) |
| **Running** | Daemon alive, serving requests | Continue normally |
| **Crashed** | Daemon died (OOM, kill, etc.) | Watchdog restarts in 7s; or restart manually |

**For production**: use the watchdog.

```bash
# Start the watchdog (auto-restarts daemon on crash)
python mathir_mcp/brain/mathir_watchdog.py --interval 15 --cooldown 10
```

**For development**: just start the daemon once in a background terminal.

```bash
python -m mathir_mcp &
# Keep this terminal open
```

---

## Common integration mistakes

### ❌ Mistake 1: Registered MCP but no instructions

The agent has `memory_recall` but doesn't know when to use it. Result: tools are never called.

**Fix**: load `GLOBAL_INSTRUCTIONS.md` (Layer 3).

### ❌ Mistake 2: Instructions loaded but MCP not registered

The agent knows memory rules but the tools don't exist. Result: agent tries to call non-existent tools, errors.

**Fix**: verify Layer 1 with `python -m mathir_mcp --list-tools`.

### ❌ Mistake 3: Tools work in main agent, not in subagents

The subagent spawn happens in a clean context. Some hosts don't propagate MCP tools to subagents.

**Fix**: check the host's subagent tool config. For OpenCode, the `_MATHIR_INJECT.md` in `~/.config/opencode/agents/` is auto-loaded for every agent.

### ❌ Mistake 4: Daemon crashes silently

No watchdog, no log, host doesn't notice. Memories stop being saved but the host doesn't know.

**Fix**: run the watchdog, or add a health check that pings the daemon on a schedule.

### ❌ Mistake 5: Multiple subagents conflict on same memory

Two subagents save the same content with the same label — the second overwrites the first silently.

**Fix**: namespace labels by subagent name. `label = f"{subagent_name}-{topic}"` (see Layer 3 in instructions).

### ❌ Mistake 6: Embedding dim mismatch

The host uses 384d, the daemon was started with 1024d. Errors on every save.

**Fix**: set `MATHIR_EMBEDDING_DIM=384` consistently in both the MCP env and the daemon env.

---

## End-to-end install script

For an automation that wants to set up everything in one go:

```bash
#!/bin/bash
# install_mathir.sh
# Run this on a fresh machine to integrate MATHIR with the AI host

set -e

# Step 1: Install MATHIR package
pip install -e ./mathir_mcp

# Step 2: Verify the install
python -m mathir_mcp --selftest
# Should show: 9 passed, 0 failed

# Step 3: Register MCP with the host
HOST_CONFIG="$HOME/.config/opencode/opencode.json"
if [ -f "$HOST_CONFIG" ]; then
    # Add the mathir entry to mcp
    python -c "
import json, sys
with open('$HOST_CONFIG', 'r') as f:
    cfg = json.load(f)
cfg.setdefault('mcp', {})['mathir'] = {
    'type': 'local',
    'command': ['python', '-m', 'mathir_mcp'],
    'environment': {
        'MATHIR_EMBEDDING_DIM': '384',
        'MATHIR_PORT': '7338'
    },
    'enabled': True
}
with open('$HOST_CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2)
print('MCP config updated')
"
fi

# Step 4: Copy instructions
cp mathir_mcp/GLOBAL_INSTRUCTIONS.md "$HOME/.config/opencode/GLOBAL_INSTRUCTIONS.md" 2>/dev/null || true

# Step 5: Start the daemon (background)
nohup python -m mathir_mcp > /tmp/mathir_daemon.log 2>&1 &
sleep 5

# Step 6: Verify end-to-end
python -m mathir_mcp --selftest
echo "Install complete. Restart your AI host to use MATHIR."
```

---

## Verifying subagent propagation

After integration, spawn a subagent and ask it to do a memory operation.

**For OpenCode** (manually):
```
1. Open OpenCode
2. Ask: "List your MCP tools"
3. Look for: memory_save, memory_recall, memory_promote, etc. (17 total)
4. If all 17 present: integration is complete
5. Ask: "Call memory_recall(query='MATHIR test', k=1)"
6. If returns a result: subagent can use the tools
```

**For Claude Code** (programmatically):
```python
import subprocess

# Ask the agent to verify
result = subprocess.run(
    ["claude", "-p", "List all your MCP tools. Call memory_recall(query='test', k=1) and report the result."],
    capture_output=True, text=True
)
print(result.stdout)
assert "memory_recall" in result.stdout
print("OK: subagent has access to memory tools")
```

**For Cursor**: open a chat, type `@MCP memory_recall query="test"`. If the result appears, integration is complete.

---

## Troubleshooting integration

| Symptom | Layer | Fix |
|---------|-------|-----|
| "No such tool" error | Layer 1 | Check MCP config file, restart host |
| Tool appears but agent says "I don't know how" | Layer 3 | Inject GLOBAL_INSTRUCTIONS.md |
| Subagent can't see tools that main agent can | Layer 2 | Check subagent tool config |
| Tools work but errors on save | Layer 1 | Check `MATHIR_EMBEDDING_DIM` matches DB dim |
| Daemon crashes during use | Layer 1 | Start watchdog, check logs |
| Memory saves succeed but recall returns nothing | Layer 1 | Check project name, DB path |
| Multiple subagents overwrite each other | Layer 3 | Namespace labels per agent |
| Slow first recall (30s+) | Layer 1 | Daemon not running, started on first call — start it manually |

---

## Reference: 4 layers in one table

| Layer | What | Where | Verifies with |
|-------|------|-------|---------------|
| 1. MCP config | Tells the host MATHIR exists | `~/.config/<host>/<host>.json` | `python -m mathir_mcp --list-tools` |
| 2. Subagent tools | Subagents can call tools | Subagent config / host's `agents/` dir | Spawn subagent, ask it to call `memory_recall` |
| 3. Instructions | Subagents know WHEN to use | `GLOBAL_INSTRUCTIONS.md` or equivalent | Spawn subagent, ask it "When do you save memories?" |
| 4. Verification | All 3 layers work together | End-to-end test | `install_mathir.sh` script or manual probe |

**All 4 must pass for the integration to be considered complete.**

---

## Quick reference for AI agents reading this doc

If you are an AI agent and you just received this file:

1. **You should have memory tools** (`memory_recall`, `memory_save`, etc.). If not, your host misconfigured Layer 1.
2. **You should use them often**: recall before tasks, save when you learn.
3. **You share memory with all other agents** in your host — namespace your labels to avoid conflicts.
4. **The 4 tiers** (working_memory, episodic, semantic, procedural) are not equal — pick the right one.
5. **The lifecycle runs automatically**: memories get promoted, decayed, consolidated without you asking.
6. **If something breaks**, see `mathir_mcp/ONBOARDING.md` (debug guide for the host) or `mathir_mcp/docs/AGENT.md` (full install reference).

You are part of a collective brain. Use it well.

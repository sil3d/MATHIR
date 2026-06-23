# MATHIR — Onboarding Guide for a New Agent / Developer

**Read this first if you just received the `mathir_mcp/` folder and have no context.**

This is the **standalone onboarding doc** — everything you need to install, run, debug, and use MATHIR. Self-contained. No prior context required.

---

## TL;DR — The 3-Step Survival Guide

```bash
# Step 1: Self-test (does it even work?)
cd mathir_mcp
python -m mathir_mcp --selftest

# Step 2: Start the daemon
python -m mathir_mcp
# (keep this running in a background terminal)

# Step 3: Verify the MCP tools are exposed
python -m mathir_mcp --list-tools
# Should show 17 tools (10 basic + 7 lifecycle)
```

If all 3 succeed, you're good. Skip to "Day-to-day usage" below.

If anything fails, see **Fallbacks** section.

---

## What is MATHIR?

MATHIR is a **persistent, living memory layer for AI agents** that:

- Stores memories in 4 cognitive tiers (`working_memory`, `episodic`, `semantic`, `procedural`)
- Auto-promotes memories between tiers as they age or get recalled
- Forgets unused memories (Ebbinghaus curve, 5%/30d decay)
- Merges near-duplicates (cosine > 0.95)
- Builds a link graph for spreading activation
- Exposes 17 MCP tools so any agent can save/recall/promote/decay/consolidate/link

It runs as a background daemon and is consumed by AI agents via the Model Context Protocol (MCP). The agent never "loads" the memory — it queries it on demand via tools.

---

## Architecture (one paragraph)

The `mathir_mcp` folder contains a **portable Python package** that:

1. **Daemon** (`mathir_daemon.py`) — long-running process on port 7338. Keeps the embedding model loaded in GPU/CPU. Handles all memory operations.
2. **MCP server** (`mathir_mcp_server.py`) — exposes 17 tools via JSON-RPC over stdio. Spawned by the AI agent host (OpenCode, Claude Code, etc.) when the agent starts.
3. **Storage** — per-project SQLite database at `.mathir/mathir.db`. Vector index via `sqlite-vec`. Each project gets its own DB.
4. **Brain stack** (`brain/`) — optional watchdog + inject proxy + link graph utilities.

The two entry points are:
- `python -m mathir_mcp` → starts the **MCP server** (for AI agents)
- The daemon starts automatically on first MCP call (and is kept alive by the watchdog)

---

## Self-test first (CRITICAL)

**Before doing anything**, run the self-test. It validates Python, deps, GPU, embedder, DB, and tools in ~30 seconds.

```bash
cd mathir_mcp
python -m mathir_mcp --selftest
```

Expected output (all green):
```
[OK] Python >= 3.10
[OK] PyTorch installed (cuda available: True/False)
[OK] sentence-transformers installed
[OK] sqlite-vec installed
[OK] DB initialization works
[OK] Embedder loads (paraphrase-multilingual-MiniLM-L12-v2, 384d)
[OK] All 17 tools registered
[OK] Daemon reachable on port 7338
[OK] End-to-end: save+recall returns correct memory
```

If any `[FAIL]`, see **Fallbacks** below.

---

## Day-to-day usage (for AI agents)

### Auto-load the MCP config

Add to your AI agent's MCP config (`~/.config/opencode/opencode.json` for OpenCode):

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

Restart the agent. The 17 `memory_*` tools appear automatically.

### Pick the right tier

| Tier | Use for | Example |
|------|---------|---------|
| `working_memory` | Current task scratchpad, session-scoped | "Debugging auth — focus on JWT" |
| `episodic` | Events: bugs fixed, decisions, sessions | "On 2026-06-15 we hit pool exhaustion, fixed with pool=50" |
| `semantic` | Stable facts that apply broadly | "Our REST API uses /v2/ prefix" |
| `procedural` | How-to recipes (label must start with `how-to:` or `recipe:`) | "how-to: rotate DB password" |

**Default: `episodic` for most saves.** The lifecycle auto-promotes to `semantic` if recalled often.

### Recall before every action

```python
memory_recall(query="topic of work", k=5)
```

**Mandatory** before:
- Starting a task
- Fixing a bug
- Making a decision
- Writing new code (check if it exists)

### Save when you learn

```python
memory_save(content="...", agent="my_agent", block_type="episodic",
            label="short-label", priority=7)
```

**Save when you LEARN something, not when told to.**

### End-of-session housekeeping

```python
memory_auto_promote()                    # mature working_memory -> episodic
memory_decay(threshold_days=30)          # archive unused
memory_consolidate(threshold=0.95, dry_run=False)  # merge dupes
memory_build_links(threshold=0.7)        # link related concepts
```

---

## Fallbacks (when things break)

### Fallback 1: Daemon won't start (port 7338 in use)

```bash
# Check what's using the port
netstat -ano | findstr :7338
# Kill the process
taskkill /PID <pid> /F
# Or change the port
$env:MATHIR_PORT = "7339"
python -m mathir_mcp
```

### Fallback 2: Embedder fails to load (no GPU / OOM)

```python
# Use ONNX runtime (CPU-only, smaller)
# Edit mathir_mcp/mathir_lib/mathir_mcp_server.py, find the embedder init:
# from mathir_lib.mathir_onnx_embedder import get_onnx_embedder
# embedder = get_onnx_embedder()
```

Or reduce model size: set `MATHIR_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` (smaller, English-only).

### Fallback 3: sqlite-vec not available

The daemon will fall back to brute-force numpy search automatically. Slower (linear scan) but works.

```bash
python -c "import numpy; print('numpy available')"
```

### Fallback 4: MCP server won't start (missing deps)

```bash
# Install minimum deps
pip install -r mathir_lib/requirements.txt
# Try again
python -m mathir_mcp --selftest
```

### Fallback 5: No internet (can't download embedder)

The embedder is cached at `~/.cache/huggingface/`. If you have it on another machine, copy it:
```bash
# Source machine
cp -r ~/.cache/huggingface /backup/
# Target machine
cp -r /backup/huggingface ~/.cache/
```

If no cached embedder at all, use ONNX (downloads once, works offline after).

### Fallback 6: Multiple agents conflict on instructions

If two agents save the same memory (same content, same label), the second overwrites the first. To avoid:

- **Namespace by agent**: prefix labels with agent name
  ```python
  label = f"coder-{topic[:30]}"  # not just topic
  ```
- **Namespace by project**: use the `project` field (default = cwd name)
- **Read before write**: `memory_recall` first to check if a similar memory exists, then `memory_consolidate` to merge instead of overwrite

### Fallback 7: Daemon crashes mid-session

The watchdog (`brain/mathir_watchdog.py`) auto-restarts in 7s. If you need a manual restart:

```bash
pkill -f "mathir_mcp" || true
python -m mathir_mcp &
```

### Fallback 8: Disk full / DB corrupted

```bash
# Check DB integrity
sqlite3 .mathir/mathir.db "PRAGMA integrity_check;"

# If corrupted, try to recover
python -c "
import sqlite3
conn = sqlite3.connect('.mathir/mathir.db')
conn.execute('VACUUM;')
conn.close()
print('recovered')
"

# Last resort: archive and rebuild
mv .mathir/mathir.db .mathir/mathir.db.bak
python -m mathir_mcp  # creates fresh DB
```

---

## Common errors to AVOID

### ❌ DO NOT start a daemon on every MCP call

The MCP server should connect to an **existing daemon**, not spawn one per call. The daemon keeps the embedder in memory. If you start fresh each time, every recall takes 30+ seconds to load the model.

**Always** start the daemon once, keep it running, let MCP connect to it.

### ❌ DO NOT save secrets, tokens, or PII to memory

MATHIR is persistent. Once saved, secrets stay in the DB until manually deleted. Use environment variables for secrets, not memory.

```python
# BAD
memory_save(content="My OpenAI key is sk-...", agent="me")
# GOOD
import os
api_key = os.environ["MATHIR_API_KEY"]  # read from env, not memory
```

### ❌ DO NOT modify the daemon DB while it's running

Use the MCP tools, not direct SQL. The daemon has in-memory caches. Direct writes cause cache drift and data loss.

```python
# BAD
import sqlite3
conn = sqlite3.connect('.mathir/mathir.db')
conn.execute("DELETE FROM memories WHERE ...")
# GOOD
memory_delete(memory_id="...", reason="...")
```

### ❌ DO NOT skip the lifecycle when saving

Don't manually set `tier='semantic'` for new memories. Start with `episodic` and let the lifecycle promote based on actual usage. Forcing tiers pollutes the quality signal.

```python
# BAD
memory_save(content="fact X", agent="me", block_type="semantic")
# GOOD
memory_save(content="fact X", agent="me", block_type="episodic")
# Then if you recall it often, memory_auto_promote() will move it to semantic
```

### ❌ DO NOT call `memory_decay` with very aggressive settings

`memory_decay(threshold_days=1, archive_floor=0.5)` will archive half your memories overnight. Use sensible defaults.

```python
# BAD
memory_decay(threshold_days=1, archive_floor=0.5)
# GOOD
memory_decay(threshold_days=30, archive_floor=0.05)
```

### ❌ DO NOT save the same memory 100 times

If you find yourself saving the same content repeatedly, you forgot. Use:
```python
memory_recall(query="exact topic", k=3)  # check first
memory_save(content="...", ...)  # only if not found
```

Or use `memory_consolidate` to merge near-duplicates.

### ❌ DO NOT run the daemon in a foreground terminal forever in production

Use the watchdog (`brain/mathir_watchdog.py`) or a process manager (systemd, supervisord, pm2, etc.). The daemon is designed to run as a background process.

---

## Multi-agent coordination (when many agents share MATHIR)

### Namespace your memories

Multiple agents writing to the same DB will conflict. To avoid:

```python
# BAD — generic label
memory_save(content="...", agent="me", block_type="episodic", label="auth-fix")
# Two agents might both save "auth-fix" and clobber each other

# GOOD — agent-prefixed label
memory_save(content="...", agent="me", block_type="episodic", label="coder-auth-fix")
memory_save(content="...", agent="ops",  block_type="episodic", label="ops-auth-config")
```

### Read before write (avoid duplicates)

```python
# Check if similar memory exists
results = memory_recall(query="auth JWT validation", k=3)
if results and results[0]["score"] > 0.85:
    # Already exists, maybe update instead of duplicate
    memory_consolidate(threshold=0.85, dry_run=False)
else:
    # New info, safe to save
    memory_save(content="...", ...)
```

### Use the project field for isolation

The daemon supports per-project DBs (default = `.mathir/mathir.db` in cwd). Use the `project` parameter when saving to scope memories:

```python
memory_save(content="...", agent="coder", block_type="episodic",
            label="...", project="api-server")
```

### Don't write to other agents' labels

If you see a memory with label `coder-auth-fix`, that's the coder agent's. Don't update it from your `ops` agent — save a new `ops-*` label and use `memory_link` to connect them.

### Periodic coordination

Run `memory_consolidate(threshold=0.85, dry_run=True)` weekly to see what dupes exist across agents, then merge or relabel.

---

## Troubleshooting diagnostic tree

```
Self-test fails?
├─ [OK] Python version       -> upgrade Python to 3.10+
├─ [FAIL] PyTorch           -> pip install torch
├─ [FAIL] sentence-transformers -> pip install sentence-transformers
├─ [FAIL] sqlite-vec        -> pip install sqlite-vec
├─ [FAIL] DB init           -> check disk space, file permissions
├─ [FAIL] Embedder loads    -> no internet? Use ONNX. OOM? Use smaller model.
└─ [FAIL] Daemon reachable  -> check port, firewall, another process

MCP tools don't appear in agent?
├─ Agent not restarted?         -> restart agent
├─ Config file wrong path?      -> check $env:APPDATA, ~/.config, etc.
├─ JSON syntax error?           -> python -m json.tool opencode.json
└─ Daemon not running?          -> python -m mathir_mcp in background

Recall returns empty for known memory?
├─ Memory in different project?  -> use correct project name
├─ Memory was archived?          -> check stats: tier='archived' excludes from recall
├─ Threshold too high?          -> memory_recall(query, k=10) for more results
└─ Embedding model mismatch?    -> check MATHIR_EMBEDDING_DIM matches DB dim
```

---

## Self-test script (what `python -m mathir_mcp --selftest` does)

The selftest validates in this order:

1. **Python version** — >= 3.10
2. **PyTorch** — importable, CUDA detection (informational only)
3. **sentence-transformers** — importable
4. **sqlite-vec** — importable + can load
5. **DB init** — creates a temp DB, initializes schema, drops it
6. **Embedder** — loads the model, runs one encode, verifies output dim
7. **Tools** — counts tools in `mathir_mcp_server.py` TOOLS array (must be 17)
8. **Daemon** — connects to port 7338, sends ping
9. **End-to-end** — saves a test memory, recalls it, verifies it comes back

If any step fails, the script prints a clear error and exits non-zero. **Always run it after install or upgrade.**

---

## Quick reference: 17 MCP tools

| Tool | Tier | When to use |
|------|------|-------------|
| `memory_save` | basic | Save (4 tiers) |
| `memory_recall` | basic | Semantic search (auto-touches) |
| `memory_smart_search` | basic | Daemon-native search |
| `memory_hybrid_search` | basic | Vector + BM25 + RRF |
| `memory_audit` | basic | View audit trail |
| `memory_export` | basic | Export to JSON |
| `memory_delete` | basic | Soft delete (sets tier='archived') |
| `memory_sessions` | basic | List sessions |
| `memory_stats` | basic | Totals by tier/agent |
| `memory_dashboard` | basic | Launch web UI |
| `memory_promote` | **lifecycle** | Move to next tier (Ebbinghaus) |
| `memory_auto_promote` | **lifecycle** | Scan and promote all eligible |
| `memory_decay` | **lifecycle** | Apply Ebbinghaus decay (5%/30d) |
| `memory_consolidate` | **lifecycle** | Merge near-duplicates |
| `memory_link` | **lifecycle** | Add link graph edge |
| `memory_get_links` | **lifecycle** | BFS graph traversal |
| `memory_build_links` | **lifecycle** | Build graph from similarities |

---

## What to read next

After this onboarding doc:

1. `docs/AGENT.md` — full install guide (more verbose)
2. `docs/DAEMON.md` — daemon internals
3. `docs/DASHBOARD_GUIDE.md` — using the web UI
4. `CHANGELOG.md` — what changed in each version
5. `benchmarks/04_lifecycle_bench/README.md` — how to measure recall@5

---

## License & contribution

MIT license. See `LICENSE` file. Contributions welcome via PR.

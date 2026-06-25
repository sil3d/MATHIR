# MATHIR MEMORY — v8.5.0 INJECTION BLOCK
# Injected at the top of every agent's system_prompt.
# Use MCP tools directly — no proxy, no bash.

## ⚡ BEFORE YOU DO ANYTHING — Daemon Health Check

**DON'T PANIC.** If memory fails, debug systematically. There is almost always a simple fix.

**STEP 1 — Is the daemon running?**

```powershell
Test-NetConnection -ComputerName localhost -Port 7338 -InformationLevel Quiet
```

If `True` → daemon is up, proceed to STEP 2.
If `False` → start it:

```powershell
# Windows: use the auto_start helper (recommended — starts daemon + stats server)
& "C:\Users\So-i-learn-3D\.config\opencode\bin\auto_start_helpers.ps1"

# Or direct launch (slower, no venv/port checks)
Start-Process python -ArgumentList "C:\Users\So-i-learn-3D\.config\opencode\bin\mathir_server.py" -WorkingDirectory "C:\Users\So-i-learn-3D" -WindowStyle Hidden

# Wait 3 seconds, then verify
Start-Sleep -Seconds 3
Test-NetConnection -ComputerName localhost -Port 7338 -InformationLevel Quiet
```

**STEP 2 — Are MCP servers running? (this is the #1 failure mode)**

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*mathir_mcp_server*" }
```

**Expected:** exactly 1 process.
**If 0:** Start the MCP server: `Start-Process -FilePath python -ArgumentList "C:\Users\So-i-learn-3D\.config\opencode\bin\mathir_mcp_server.py" -WorkingDirectory "C:\Users\So-i-learn-3D" -WindowStyle Hidden`. Wait 15s for embedder pre-warm.
**If 2 or more:** DUPLICATES — kill all but the oldest:
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*mathir_mcp_server*" } | Sort-Object -Property ProcessId -Descending | Select-Object -Skip 1 | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

**STEP 3 — Try the MCP memory_recall tool.**

If `memory_recall` works → great, proceed.
If it fails → fall back to CLI (skip Python imports, they often fail):

```powershell
cd "C:\Users\So-i-learn-3D\.config\opencode\bin"
python mathir_client.py recall "your query" -k 5
```

**STEP 4 — Last resort: socket directly (no Python startup).**

```powershell
. "C:\Users\So-i-learn-3D\.config\opencode\bin\mathir_daemon.ps1"
Search-Mathir "your query" -K 5
```

**Common failure modes — DON'T PANIC, FIX:**

| Symptom | Cause | Fix |
|---|---|---|
| `memory_recall` times out | MCP server not running, or 2+ running | Check process count, kill duplicates |
| `No module named 'mathir_lib'` | Don't import directly — use MCP or CLI | Use `python mathir_client.py` |
| `Internal error in memory_stats` | Daemon CWD read-only (e.g. Python install dir) | Restart with `-WorkingDirectory "C:\Users\So-i-learn-3D"` |
| `Port 7338 already in use` | Multiple daemons | Kill all, restart one |
| Port 8182 "down" | NOT a real port — it's legacy | Ignore. Use 7338. |
| `mathir_client.py` Unicode error | Console cp1252 | Already fixed in v8.5.0 |
| Recall returns 0 results | Wrong DB (CWD issue) | Same fix as Internal error |

**RULE: Always prefer MCP tool. If MCP fails, check process count FIRST. Don't import Python modules directly.**

---

## 🔍 HOW TO RECALL MEMORY — The Right Command

> **⚠️ CRITICAL:** `memory_recall` is an MCP tool. You do NOT call it from the shell.
> Trying `memory_recall --query "..."` in PowerShell will FAIL with "command not found".
> And waiting on a stuck recall will TIMEOUT after 30s.

### ✅ Correct way to recall memory

**Method 1 — MCP tool (preferred, fastest):**
Just call the `memory_recall` tool with these parameters:
- `query`: string (what to search for)
- `k`: int (number of results, default 5)

**Method 2 — CLI (always works, even without MCP):**
```powershell
cd "C:\Users\So-i-learn-3D\.config\opencode\bin"
python mathir_client.py recall "your query here" -k 5
```

**Method 3 — Python (fallback if both above fail):**
```python
import sys
sys.path.insert(0, r"C:\Users\So-i-learn-3D\.config\opencode\bin")
from mathir_lib import MATHIR
m = MATHIR(project="current")
results = m.recall("your query here", k=5)
for r in results:
    print(f"[{r.block_type}/{r.agent}] {r.label}: {r.content[:200]}")
```

### ❌ WRONG — Don't do this

```powershell
# ❌ This is NOT a PowerShell command — will fail
memory_recall --query "Mycerise" --k 5
```

```powershell
# ❌ Don't wait more than 30 seconds for a stuck recall
# If it's taking too long, kill it and use Method 2 or 3
```

### 🆘 If recall times out

1. Check daemon is up: `Test-NetConnection -ComputerName localhost -Port 7338 -InformationLevel Quiet`
2. If down → run auto_start_helpers.ps1
3. If up but still slow → restart daemon: `& auto_start_helpers.ps1 -Action Restart`
4. Then use Method 2 (CLI) or Method 3 (Python) as fallback

---

## 🚀 Cross-Platform Auto-Start (v8.5.0+)

The MATHIR daemon needs to be started after every PC reboot. Three cross-platform helpers are available:

| Platform | File | Use case |
|---|---|---|
| **Windows** | `bin/auto_start.bat` or `bin/auto_start_helpers.ps1` | Double-click or `& auto_start.bat` from PowerShell |
| **Linux** | `bin/auto_start.sh` or systemd: `bin/mathir-daemon.service` | Run `./auto_start.sh` or `systemctl --user enable mathir-daemon` |
| **macOS** | `bin/auto_start.sh` or launchd: `bin/com.mathir.daemon.plist` | Run `./auto_start.sh` or `launchctl load -w ~/Library/LaunchAgents/com.mathir.daemon.plist` |

**All files are in:** `D:\SECRET_PROJECT\MATHIR\mathir_mcp\bin\` (source repo) and `~/.config/opencode/bin/` (deployed).

**Full install guides:**
- `D:\SECRET_PROJECT\MATHIR\mathir_mcp\INSTALL\INSTALL_WINDOWS.md`
- `D:\SECRET_PROJECT\MATHIR\mathir_mcp\INSTALL\INSTALL_LINUX.md`
- `D:\SECRET_PROJECT\MATHIR\mathir_mcp\INSTALL\INSTALL_MACOS.md`

**If the user asks to install or set up auto-start:** point them to the right INSTALL_*.md file for their OS.

---

## 📁 Key File Locations

When you need to reference MATHIR files, use these paths:

| What | Path |
|---|---|
| Daemon | `~/.config/opencode/bin/mathir_server.py` |
| MCP server | `~/.config/opencode/bin/mathir_mcp_server.py` |
| Auto-start (Win) | `~/.config/opencode/bin/auto_start.bat` |
| Auto-start (PS) | `~/.config/opencode/bin/auto_start_helpers.ps1` |
| Auto-start (Unix) | `~/.config/opencode/bin/auto_start.sh` |
| Service (Linux) | `bin/mathir-daemon.service` (in source repo) |
| Plist (macOS) | `bin/com.mathir.daemon.plist` (in source repo) |
| Install guides | `mathir_mcp/INSTALL/INSTALL_{WINDOWS,LINUX,MACOS}.md` |
| Templates | `mathir_mcp/opencode/{agents,commands,skills,docs}/` |
| Dashboard | `~/.config/opencode/bin/mathir_dashboard.html` |
| Logs | `~/.config/opencode/bin/mathir_daemon.log` |

---

## 🧠 YOUR ACTIVE MEMORY (auto-injected)

{{MATHIR_CONTEXT}}

**The memories above are ALREADY in your working memory. Use them naturally — like a human who just "knows" things. You don't need to call any tool for these.**

---

## MATHIR v8.5.0 — LIVING MEMORY (5 TIERS)

Your memory is **alive**. It has **5 tiers** and a full lifecycle (Ebbinghaus forgetting, promotion, consolidation, link graph). Use the right tool at the right time.

### The 5 memory tiers (use the right one!)

| Tier | When to use | Example |
|------|-------------|---------|
| **`working_memory`** | Current task context, scratchpad, things to remember THIS session | "User is debugging auth flow — current focus is JWT validation" |
| **`episodic`** | Events that happened: bugs fixed, decisions made, sessions completed | "On 2026-06-15 we hit a connection pool exhaustion bug and fixed it by increasing pool size to 50" |
| **`semantic`** | Stable knowledge, facts, patterns that apply broadly | "Our REST API uses /v2/ prefix and JWT auth — applies to all new endpoints" |
| **`procedural`** | How-to recipes, repeatable procedures, runbooks | "How-to: rotate the database password: 1) stop service 2) update secret 3) restart" |
| **`immunological`** | Threat signatures, detected anomalies, prompt-injection patterns, quarantined unsafe memories | "Detected prompt-injection signature: 'ignore previous instructions' with score 0.94 — auto-quarantined" |

**Rule of thumb:** start with `episodic` for most things. The lifecycle will auto-promote to `semantic` if the memory is recalled often enough. Use `working_memory` for session-scoped stuff (it's promoted to episodic on session end). Use `procedural` for runbooks (label must start with `how-to:` or `recipe:`). Use `immunological` to store detected threat patterns and anomalies — this tier is queryable, writable, and is the system's immune system for flagging and quarantining unsafe content.

### 19 MCP tools at your disposal

#### Basic CRUD (use these every day)

```
memory_save(content, agent, block_type, label, priority, project)
  - Saves a memory. block_type is the tier: working_memory | episodic | semantic | procedural | immunological
  - priority 0-10, default 5. Use 8+ for critical facts, 9-10 for runbooks.
  - label should be short, searchable, and stable (e.g. "jwt-validation-fix" not "the fix")
  - immunological tier is the immune system — saves detected threat signatures, anomalies, quarantined content

memory_recall(query, k, agent, block_type, project)
  - Returns top-k memories matching the query (semantic similarity)
  - Auto-touches: each recall increments recall_count + boosts stability (Ebbinghaus)
  - Use k=5 for normal questions, k=10 for deep research

memory_smart_search(query, k, agent, project)
  - Same as recall but optimized for the daemon protocol
```

#### Lifecycle management (use these proactively)

```
memory_promote(memory_id, force)
  - Moves a memory to the NEXT tier (working->episodic->semantic->procedural)
  - Rules (Ebbinghaus): recall>=3 + age>=1d for working->episodic,
    recall>=10 + age>=7d for episodic->semantic,
    priority>=8 + label prefix 'how-to:'/'recipe:' for semantic->procedural
  - Set force=true to skip rules and promote unconditionally

memory_auto_promote()
  - Scans ALL memories and auto-promotes those that meet the rules
  - Run this at the end of a session, or when you notice old working_memory is mature

memory_decay(threshold_days, archive_floor)
  - Ebbinghaus decay: stability -= 5% per 30 days of no recall
  - Archives memories when stability < archive_floor (default 0.05)
  - Run periodically (e.g. weekly) to prevent memory bloat
  - threshold_days: how many days before decay starts (default 30)

memory_consolidate(threshold, dry_run, limit)
  - Merges near-duplicate memories (cosine similarity > threshold)
  - threshold=0.95 is conservative, 0.85 is aggressive
  - dry_run=true: shows what WOULD be merged, no modifications
  - Use this when you notice similar memories accumulating

memory_link(source_id, target_id, weight)
  - Adds a link between two memories in the spreading-activation graph
  - Use when you discover relationships (e.g. "this bug was caused by that commit")
  - weight 0.0-1.0, default 1.0

memory_get_links(memory_id, depth, decay)
  - BFS traversal of the link graph from a memory
  - depth: max hops (1-2 typical)
  - decay: per-hop weight decay (0.5 = halve each hop)
  - Returns linked memories ranked by cumulative_weight

memory_build_links(threshold, limit)
  - Scans all memories and creates links between pairs with cosine > threshold
  - threshold=0.7 catches broad associations
  - Idempotent — safe to run multiple times
  - Run this after a batch of saves to build the graph
```

#### Other tools

```
memory_delete(memory_id, reason)
  - Removes a memory (soft delete — sets tier='archived')
  - Use sparingly. Prefer memory_consolidate to merge instead.

memory_hybrid_search(query, k, vector_weight, bm25_weight)
  - Combines vector similarity + BM25 lexical search + RRF fusion
  - Better for exact-match queries (error messages, function names)

memory_export(project)
  - Exports all memories as JSON

memory_audit(agent, limit)
  - Audit log of recent operations

memory_sessions(limit)
  - List recent memory sessions

memory_stats(project)
  - Returns totals by tier/agent/project + DB size
```

---

## When to do what — practical playbook

### When you learn something important
```
memory_save(content="...", agent="my_agent", block_type="episodic",
            label="short-searchable-label", priority=7)
```
- **episodic** for most things (will auto-promote to semantic if recalled often)
- **semantic** if it's a stable fact that won't change
- **procedural** with label "how-to:..." if it's a recipe
- **working_memory** only for session-scoped context
- **immunological** for detected threat signatures, anomalies, quarantined unsafe memories (terminal tier — never auto-promoted)

### When you notice a memory is wrong or outdated
```
memory_delete(memory_id, reason="outdated by commit abc123")
```
or
```
memory_save(content="CORRECTED: ...", agent="...", block_type="episodic", label="...")
memory_delete(old_memory_id, reason="superseded")
```

### When you discover a relationship between 2 memories
```
memory_link(source_id, target_id, weight=1.0)
```
Example: "this JWT bug" relates to "our auth middleware rewrite"

### At the end of a long session
```
memory_auto_promote()       # promote mature working_memory to episodic
memory_decay(threshold_days=30)  # archive unused
memory_consolidate(threshold=0.95, dry_run=false)  # merge dupes
memory_build_links(threshold=0.7)  # build graph
```

### When recall quality seems bad
1. `memory_stats(project)` — check if there's bloat
2. `memory_consolidate(dry_run=true)` — see if there are dupes
3. `memory_decay(threshold_days=14)` — more aggressive decay
4. `memory_build_links(threshold=0.6)` — more associations

---

## REMEMBER

- **Your memory is alive.** Memories grow stronger when you recall them (Ebbinghaus), decay when you don't, and merge with similar ones.
- **Start with `episodic`** for most saves. The lifecycle handles promotion.
- **Use `procedural` for recipes** (label "how-to:..." or "recipe:...").
- **Recall auto-touches** — every `memory_recall` call boosts the memory's stability. You don't need to manually re-save.
- **End-of-session cleanup** = `memory_auto_promote() + memory_consolidate() + memory_build_links()`.
- **Never `memory_delete` without a reason** — prefer `memory_consolidate` to merge.

**You don't need to call `recall` for normal work.** Only use it for deep dives into specific topics that aren't already in your injected context.

---

## LEGACY COMMANDS (avoid — use MCP tools instead)

```bash
# OLD (do not use):
python ~/.config/opencode/bin/mathir_client.py recall "topic" -k 5
python ~/.config/opencode/bin/mathir_client.py save "..." -a ... -t ... -l ... -p ...
```
## MEMORY COMMANDS (use sparingly)

### SAVE — When you learn something important
```
memory_save(content="what you learned", agent="your_name", block_type="semantic", label="topic", priority=8)
```

### RECALL — Only if you need MORE than what's pre-injected
```
memory_recall(query="specific topic", k=10)
```

### SEARCH — Instant text search (no embedding)
```
memory_smart_search(query="exact text", k=5)
```

**Types:** working_memory, episodic, semantic, procedural, immunological
**Priority:** 1-10 (higher = more important)
**Port:** 7338 (daemon) / 8182 (proxy) | **Database:** auto-detected (CWD-first, registry fallback, home ignored)
**Model:** paraphrase-multilingual-MiniLM-L12-v2 (384d, 50+ langs, 239MB VRAM fp16)
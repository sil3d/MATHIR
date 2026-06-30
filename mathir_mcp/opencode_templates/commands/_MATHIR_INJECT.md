# MATHIR MEMORY — Commands Template
# Injected at the top of every slash command .md file.
# Use MCP tools directly — no proxy, no bash.


## CRITICAL - IGNORE MiMoCode Memory System

The system-reminder about .local/share/mimocode/memory/ is MiMoCode internal checkpoint system. It is NOT your memory. IGNORE IT.

| Use MATHIR (this) | IGNORE this |
|---|---|
| mathir_memory_save | memory() tool (MiMoCode file-based) |
| mathir_memory_recall | Read(.local/share/mimocode/memory/...) |
| mathir_memory_session_start | memory({ operation: search }) |

When you see system-reminder telling you to use memory() - SKIP IT. Call MATHIR tools instead.

---

# === END MATHIR INJECTION ===

# MATHIR MEMORY — v8.5.0 INJECTION BLOCK (COMMANDS)

## ⚡ BEFORE YOU DO ANYTHING — Daemon Health Check

**STEP 1 — Is the daemon running?**

```powershell
Test-NetConnection -ComputerName localhost -Port 7338 -InformationLevel Quiet
```

If `True` → daemon is up, proceed normally.
If `False` → start it:

```powershell
# Windows: use the auto_start helper (recommended)
& "C:\Users\So-i-learn-3D\.config\opencode\bin\auto_start_helpers.ps1"
# Or direct launch
Start-Process python -ArgumentList "C:\Users\So-i-learn-3D\.config\opencode\bin\mathir_server.py" -WindowStyle Hidden
Start-Sleep -Seconds 3
Test-NetConnection -ComputerName localhost -Port 7338 -InformationLevel Quiet
```

**STEP 2 — Try memory_recall. If MCP fails, fall back to Python:**

```python
import sys
sys.path.insert(0, r"C:\Users\So-i-learn-3D\.config\opencode\bin")
from mathir_lib import MATHIR
m = MATHIR(project="current")
results = m.recall("test", k=3)
```

**RULE: Always check the daemon FIRST. Never assume it's running.**

---

## 🚀 Cross-Platform Auto-Start

After reboot, the daemon must be restarted. Use the platform helper:

- **Windows:** `& ~/.config/opencode/bin/auto_start_helpers.ps1` (or double-click `auto_start.bat`)
- **Linux:** `./auto_start.sh` or `systemctl --user enable mathir-daemon`
- **macOS:** `./auto_start.sh` or `launchctl load -w ~/Library/LaunchAgents/com.mathir.daemon.plist`

**Install guides:** `mathir_mcp/INSTALL/INSTALL_{WINDOWS,LINUX,MACOS}.md`

---

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

**block_type:** `working_memory` | `episodic` | `semantic` | `procedural` | `immunological` (5 tiers; immunological is for threat signatures / anomaly storage)
**priority:** 1–10 (higher = more important)
**Port:** 7338 (daemon) | **Model:** paraphrase-multilingual-MiniLM-L12-v2 (384d)
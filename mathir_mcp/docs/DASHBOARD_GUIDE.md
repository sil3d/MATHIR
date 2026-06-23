# MATHIR Dashboard — Setup Guide

## What Is This?

A real-time web dashboard that visualizes your MATHIR neural memory system:
- 4-tier memory breakdown (working, episodic, semantic, procedural)
- Per-agent statistics
- Memory creation timeline
- Router weight visualization
- Memory search and delete

## Quick Start

### 1. Start the Server

```bash
# From the mathir_mcp/ directory (where pyproject.toml lives)
cd /path/to/MATHIR/mathir_mcp
python -m mathir_lib.mathir_stats_server

# Or, if you installed mathir_mcp with `pip install -e .` from anywhere:
python -m mathir_lib.mathir_stats_server
```

Output:
```
MATHIR Dashboard Server
  Database: /path/to/your/mathir.db
  URL:      http://127.0.0.1:7420
```

### 2. Open the Dashboard

Open http://127.0.0.1:7420 in your browser.

The dashboard auto-refreshes every 30 seconds.

## Platform-Specific Instructions

### Windows

**PowerShell:**
```powershell
cd /path/to/MATHIR/mathir_mcp
python -m mathir_lib.mathir_stats_server
```

**Double-click:** Create a `start_dashboard.bat` file:
```bat
@echo off
cd /d "%~dp0"
python -m mathir_lib.mathir_stats_server
pause
```

### Linux

```bash
cd /path/to/MATHIR/mathir_mcp
python3 -m mathir_lib.mathir_stats_server
```

**As a systemd service** (auto-start on boot):
```ini
# /etc/systemd/system/mathir-dashboard.service
[Unit]
Description=MATHIR Dashboard
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/MATHIR/mathir_mcp
ExecStart=/usr/bin/python3 -m mathir_lib.mathir_stats_server
Restart=always
Environment=MATHIR_STATS_PORT=7420

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable mathir-dashboard
sudo systemctl start mathir-dashboard
```

### Mac

```bash
cd /path/to/MATHIR/mathir_mcp
python3 -m mathir_lib.mathir_stats_server
```

**With launchd (auto-start):**
```xml
<!-- ~/Library/LaunchAgents/com.mathir.dashboard.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mathir.dashboard</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>-m</string>
        <string>mathir_lib.mathir_stats_server</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/MATHIR/mathir_mcp</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.mathir.dashboard.plist
```

## Agent Integration — How Your Agent Should Use MATHIR

### The 3 Rules

1. **ALWAYS recall before acting** — never start a task without checking memory
2. **ALWAYS save after completing** — every task completion gets saved
3. **NEVER say "pre-existing error"** — fix it or escalate with full context

### Agent Workflow

```
SESSION START
    ↓
1. Recall existing memories (what do I know?)
    ↓
2. Work on task
    ↓
3. Save what I learned (skills, patterns, fixes)
    ↓
SESSION END
```

### Commands for Agents

**At session start — recall what you know:**
```bash
# Via daemon (fast, model already loaded)
python -m mathir_lib.mathir_client recall "project context" -k 10

# Via MCP (if using MCP integration)
memory_recall(query="project context", k=10)
```

**After completing a task — save what you did:**
```bash
# Via daemon
python -m mathir_lib.mathir_client save "Task completed: [what was done]" \
  --agent [your_agent_name] --type episodic --label task-[short-description] --priority 7

# Via MCP
memory_save(content="Task completed: [what was done]", agent="[your_agent_name]",
            block_type="episodic", label="task-[short-description]", priority=7)
```

**When you discover a pattern — save it as a skill:**
```bash
python -m mathir_lib.mathir_client save "SKILL: [Problem Title]

PROBLEM:
- What: [Exact description]
- When: [When does it happen?]

ROOT CAUSE:
- Why: [Technical explanation]

SOLUTION:
- Step 1: [Exact command or code]
- Verification: [How to verify]

PREVENTION:
- How to avoid: [What to do differently]

EXAMPLES:
- Bad: [Code that causes problem]
- Good: [Code that avoids problem]" \
  --agent [your_agent_name] --type semantic --label skill-[problem-slug] --priority 9
```

### Memory Types — When to Use Each

| Type | When | Example |
|------|------|---------|
| `working_memory` | Active task, current bug | "Bug: null pointer in auth.py:42" |
| `episodic` | After completing a task | "Fixed login refresh token bug in PR #42" |
| `semantic` | Discovery about the project | "This project uses JWT tokens in Authorization header" |
| `procedural` | Workflow that works well | "How to debug auth issues: 1. Check token, 2. Check CORS..." |

### What NOT to Save

- ❌ Temporary state ("I'm currently looking at file X")
- ❌ Duplicate of existing memory (search first!)
- ❌ Vague notes ("something is wrong with auth")
- ❌ Implementation details that will change

### Proactive Memory Delivery (Daemon Push)

If the daemon supports push mode, your agent can get relevant memories automatically:

```bash
# Get memories relevant to current context
python -m mathir_lib.mathir_client push "current task context" --auto

# JSON mode for structured consumption
python -m mathir_lib.mathir_client push "current task context" --json
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATHIR_STATS_PORT` | `7420` | Dashboard port |
| `MATHIR_DB` | auto-detect | Path to mathir.db file |
| `MATHIR_CONFIG` | auto-detect | Path to config JSON |

### Database Auto-Detection

The server searches for `mathir.db` in this order:

1. `MATHIR_DB` environment variable
2. `./.mathir/mathir.db` (current directory)
3. `../.mathir/mathir.db` (parent directory)
4. `~/.mathir/mathir.db` (home directory)
5. Scan `~/Documents`, `~/Desktop`, `~/Projects` for `.mathir/mathir.db`

### Using with Different Projects

Point to a specific database:

```bash
# Windows
set MATHIR_DB=C:\my_project\.mathir\mathir.db
python -m mathir_lib.mathir_stats_server

# Linux/Mac
export MATHIR_DB=/home/user/my_project/.mathir/mathir.db
python3 -m mathir_lib.mathir_stats_server
```

## API Reference

All endpoints return JSON. Add `?project=name` to filter by project.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/overview` | GET | Global stats: tier counts, agents, config |
| `/api/memories` | GET | List memories (params: limit, offset, agent, block_type) |
| `/api/tiers` | GET | Detailed tier breakdown with capacity |
| `/api/router` | GET | Router weights and KL coefficient |
| `/api/agents` | GET | Per-agent statistics |
| `/api/timeline` | GET | Memory creation timeline (hourly) |
| `/api/projects` | GET | List all detected MATHIR databases |
| `/api/memory/delete` | POST | Delete memory (body: {memory_id, reason}) |

## Troubleshooting

### "No database found"

- Check that `.mathir/mathir.db` exists in your project
- Set `MATHIR_DB` env var to the full path
- Run `python -c "import sqlite3; print(sqlite3.connect('path/to/mathir.db').execute('SELECT COUNT(*) FROM memories').fetchone())"` to verify DB

### Port already in use

```bash
# Change port
set MATHIR_STATS_PORT=7421  # Windows
export MATHIR_STATS_PORT=7421  # Linux/Mac
python -m mathir_lib.mathir_stats_server
```

### Charts not loading

- The dashboard loads Chart.js from CDN: `https://cdn.jsdelivr.net/npm/chart.js@4.4.0`
- Requires internet connection on first load
- After first load, it's cached by the browser

## Architecture

```
mathir_lib/mathir_stats_server.py   ← Python HTTP server (no dependencies)
mathir_lib/mathir_stats_dashboard.html  ← Single-file HTML dashboard (Chart.js)
                                     ↓
                          http://127.0.0.1:7420
                                     ↓
                          Browser renders charts + tables
                                     ↓
                          API calls to /api/* endpoints
                                     ↓
                          Reads from .mathir/mathir.db (SQLite)
```

No external dependencies beyond Python stdlib + sqlite3.

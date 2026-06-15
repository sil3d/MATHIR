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
cd D:\SECRET_PROJECT\MATHIR\mcp
python dashboard_server.py
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
cd D:\SECRET_PROJECT\MATHIR\mcp
python dashboard_server.py
```

**Command Prompt:**
```cmd
cd D:\SECRET_PROJECT\MATHIR\mcp
python dashboard_server.py
```

**Double-click:** Create a `start_dashboard.bat` file:
```bat
@echo off
cd /d "%~dp0"
python dashboard_server.py
pause
```

### Linux

```bash
cd /path/to/MATHIR/mcp
python3 dashboard_server.py
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
WorkingDirectory=/path/to/MATHIR/mcp
ExecStart=/usr/bin/python3 dashboard_server.py
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
cd /path/to/MATHIR/mcp
python3 dashboard_server.py
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
        <string>/path/to/MATHIR/mcp/dashboard_server.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/MATHIR/mcp</string>
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
set MATHIR_DB=D:\my_project\.mathir\mathir.db
python dashboard_server.py

# Linux/Mac
export MATHIR_DB=/home/user/my_project/.mathir/mathir.db
python3 dashboard_server.py
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
python dashboard_server.py
```

### Charts not loading

- The dashboard loads Chart.js from CDN: `https://cdn.jsdelivr.net/npm/chart.js@4.4.0`
- Requires internet connection on first load
- After first load, it's cached by the browser

## Architecture

```
dashboard_server.py     ← Python HTTP server (no dependencies)
dashboard.html          ← Single-file HTML dashboard (Chart.js)
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

# God-Mode Bridge — Mycerise V2 Tauri

Solves the **async coordination gap** between orchestrator and workers (no native waiting pool).

## What is it?

A lightweight polling daemon (`god_bridge.py`) + thin shell wrappers (`god_poll.ps1` / `god_poll.sh`)
that poll the **MATHIR HTTP daemon** (port `7338`) and notify terminals about new god-mode events.

Uses the **already-implemented** routes:
- `POST /api/god/poll` — workers fetch their next `god:task:*:pending`
- `POST /api/memory/audit` — observers see new entries by label prefix
- `GET /api/god/agents` — orchestrator lists active workers

→ See **[PROTOCOL.md](PROTOCOL.md)** for the full label taxonomy and message flow.

## Quick start

### 1. Worker terminal (e.g. MiMo Code Agent)

```bash
python scripts/god-mode/god_bridge.py --mode worker --name mimo-code --interval 5
```

Or PowerShell equivalent:
```powershell
.\scripts\god-mode\god_poll.ps1 -Mode worker -Name mimo-code -Interval 5
```

### 2. Orchestrator terminal (this opencode instance)

```bash
python scripts/god-mode/god_bridge.py --mode orchestrator --interval 5 --project Mycerise_V2_Taur
```

Press `Ctrl+C` to stop.

### 3. Observer (optional — any third terminal)

```bash
python scripts/god-mode/god_bridge.py --mode observer --interval 10
```

## Environment variables (cross-platform, no hardcoded paths)

| Var | Default | Purpose |
|-----|---------|---------|
| `MATHIR_DAEMON_URL` | `http://localhost:7338` | Daemon URL |
| `MYCERISE_STATE_DIR` | `$XDG_CONFIG_HOME/mycerise` (POSIX) or `%USERPROFILE%\.config\mycerise` (Win) | State + log dir |
| `MYCERISE_LOG_FILE` | `$MYCERISE_STATE_DIR/god_bridge.log` | Log file path |

> ⚠️ **No hardcoded paths.** Override per machine via env vars if defaults don't fit.

## What you get

- 🔔 **Console beep** when new event detected
- 💻 **Log file** at the resolved `MYCERISE_LOG_FILE`
- 📝 **State file** at the resolved state dir (orchestrator mode — last-seen ids, prevents dup notifications)
- 🪟 **Cross-platform**: Windows (`winsound.MessageBeep` + `[Console]::Beep`) · Linux (`paplay` if installed + `\a`) · macOS (`afplay` + `\a`)

## Requirements

- Python 3.8+ (uses only stdlib — no extra deps)
- MATHIR daemon running, accessible via `MATHIR_DAEMON_URL`
- For Linux audio: `paplay` (PulseAudio) recommended, falls back to terminal bell otherwise

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `HTTP ERROR` constantly | MATHIR daemon not running → `curl $MATHIR_DAEMON_URL/health` |
| No notifications | Check label: must start with `god:task:`, `god:result:`, etc. |
| Spam on every poll | Orchestrator mode remembers `seen_ids` in state file |
| State file permission denied | Override `MYCERISE_STATE_DIR` env var |

## Integration with workers

Each worker agent **should** check the polling endpoint on every turn:

```python
import os, requests
r = requests.post(
    f"{os.environ.get('MATHIR_DAEMON_URL', 'http://localhost:7338')}/api/god/poll",
    json={"agent": "mimo-code", "status": "pending"},
)
task = r.json().get("task")
if task:
    # execute task["content"], then save result with label god:result:<id>:mimo-code:completed
```

## See also

- `PROTOCOL.md` — label spec + message flow diagrams
- `../install_mathir.bat` / `install_mathir.sh` — MATHIR daemon installer (Phase 2 deliverable)
- `mathir_server.py:1422` (in your MATHIR install) — `/api/god/poll` source

# God-Mode Tooling — MATHIR

Solves the **async coordination gap** between an orchestrator and N worker agents (no native waiting pool).

Two ways to run god-mode, depending on whether a human is watching each worker terminal:

| Mode | Files | Human needed per worker? |
|---|---|---|
| **Notify-only bridge** | `god_bridge.py`, `god_poll.ps1`, `god_poll.sh` | Yes — beeps/logs a new task, a human still has to trigger the agent to act on it |
| **Headless on-demand workers** | `god_mode_start.py`, `god_mode_stop.py`, `god_worker_daemon.py`, `god_mode_report.py` | No — spawns the actual coding-agent CLI headlessly and executes the task unattended |

Uses the **already-implemented** daemon routes (in `mathir_lib/mathir_server.py`):
- `POST /api/god/poll` — workers fetch their next `god:task:*:pending` (atomic claim: SELECT+UPDATE wrapped in `BEGIN IMMEDIATE`, so two pollers sharing a name can't double-claim the same task)
- `POST /api/god/ack` — flips a claimed task's label in place (`pending`→`running`→`completed`/`failed`) instead of creating a duplicate memory each time
- `POST /api/memory/audit` — observers see new entries by label prefix
- `GET /api/god/agents` — orchestrator lists active workers

→ See **[PROTOCOL.md](PROTOCOL.md)** for the full label taxonomy and message flow.

## Notify-only bridge

### 1. Worker terminal (e.g. MiMo Code Agent)

```bash
python god_bridge.py --mode worker --name mimo-code --interval 5
```

Or PowerShell equivalent:
```powershell
.\god_poll.ps1 -Mode worker -Name mimo-code -Interval 5
```

### 2. Orchestrator terminal

```bash
python god_bridge.py --mode orchestrator --interval 5 --project <your-project>
```

Press `Ctrl+C` to stop.

### 3. Observer (optional — any third terminal)

```bash
python god_bridge.py --mode observer --interval 10
```

## Headless on-demand workers

For unattended execution — the orchestrator launches a real coding-agent CLI (opencode, mimo, claude, openclaude, codex, gemini, aider, cursor-agent, copilot) in the background, and it runs the whole task to completion with no human in the loop.

### 1. Detect installed tools

```bash
python god_mode_start.py --detect
```

### 2. Launch a headless worker (only on explicit human request — never autostarted)

```bash
python god_mode_start.py --launch opencode --name mimo-code --cwd D:\SECRET_PROJECT\MATHIR --project MATHIR
```

Internally this spawns `god_worker_daemon.py --tool opencode --name mimo-code --cwd ...` as a detached background process (Windows: `CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`; POSIX: `start_new_session=True`) and records it in a state file at `$MATHIR_HOME/logs/god_mode_workers.json`.

`god_worker_daemon.py` then loops: poll `/api/god/poll` → on a claimed task, `ack` it to `running` → build a headless invocation for the target tool (each tool's real CLI flags, verified against its own docs — e.g. `opencode run --auto`, `claude -p --dangerously-skip-permissions`, `codex exec --sandbox workspace-write`) → stream its stdout/stderr live to the worker's log file → retry up to `--retries` times if the process exits 0 but never actually saved a `god:result` (observed: model finishes its turn without calling `memory_save`) or hangs past `--timeout` (default 450s) → `ack` the task `completed` or `failed`.

### 3. Stop workers

```bash
python god_mode_stop.py --name mimo-code
python god_mode_stop.py --all
```

### 4. Get a deterministic report

Because relying on an orchestrator LLM to remember and relay every worker's result has failed in practice (a real incident, 2026-07-21: 3 workers' answers never reached the human), pull a report directly from the SQLite DB instead of trusting the orchestrator's own memory:

```bash
python god_mode_report.py --cwd D:\SECRET_PROJECT\MATHIR
```

This reads `<cwd>/.mathir/mathir.db` directly (bypassing `/api/memories`'s project-resolution quirks), groups records by `task_id`, and prints per-task assignments + results — including correctly handling multi-target fan-out (a prior bug collapsed multiple targets down to one "latest" record).

## Environment variables (cross-platform, no hardcoded paths)

| Var | Default | Purpose |
|-----|---------|---------|
| `MATHIR_DAEMON_URL` | `http://localhost:7338` | Daemon URL |
| `MATHIR_HOME` | `~/.config/MATHIR` | Base config/state dir — used by `god_mode_start.py`/`god_mode_stop.py` for the worker state file |
| `MATHIR_STATE_DIR` | `$XDG_CONFIG_HOME/mathir` (POSIX) or `%USERPROFILE%\.config\mathir` (Win) | State + log dir used by `god_bridge.py` specifically |
| `MATHIR_LOG_FILE` | `$MATHIR_STATE_DIR/god_bridge.log` | Log file path for `god_bridge.py` |
| `MATHIR_GOD_TASK_TIMEOUT` | `450` | Seconds before `god_worker_daemon.py` kills a hung headless run |

> ⚠️ **No hardcoded paths.** Override per machine via env vars if defaults don't fit.

## What the notify-only bridge gives you

- 🔔 **Console beep** when new event detected
- 💻 **Log file** at the resolved `MATHIR_LOG_FILE`
- 📝 **State file** at the resolved state dir (orchestrator mode — last-seen ids, prevents dup notifications)
- 🪟 **Cross-platform**: Windows (`winsound.MessageBeep` + `[Console]::Beep`) · Linux (`paplay` if installed + `\a`) · macOS (`afplay` + `\a`)

## Requirements

- Python 3.8+ (uses only stdlib — no extra deps)
- MATHIR daemon running, accessible via `MATHIR_DAEMON_URL`
- For headless workers: the target CLI tool (opencode/claude/codex/etc.) installed and on `PATH`, with its own permission/auto-approve config already set up (headless mode cannot answer interactive prompts — see `god_worker_daemon.py`'s module docstring for the documented hang bugs this works around)
- For Linux audio (bridge mode): `paplay` (PulseAudio) recommended, falls back to terminal bell otherwise

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `HTTP ERROR` constantly | MATHIR daemon not running → `curl $MATHIR_DAEMON_URL/health` |
| No notifications (bridge mode) | Check label: must start with `god:task:`, `god:result:`, etc. |
| Spam on every poll | Orchestrator mode remembers `seen_ids` in state file |
| State file permission denied | Override `MATHIR_STATE_DIR` (bridge) or `MATHIR_HOME` (headless workers) |
| Headless worker exits 0 but no result saved | Expected/handled — `god_worker_daemon.py` retries automatically (`--retries`, default 2 extra attempts); check its log if it still fails after all attempts |
| Headless worker hangs forever | Check the tool-specific hang mitigations in `god_worker_daemon.py`'s docstring (e.g. opencode/mimo need `"permission": {"*": "allow"}` in their own config or they hang on the first bash call) |

## Integration with workers (manual/MCP-tool flow)

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
- `god_worker_daemon.py` module docstring — per-tool headless CLI flags and their doc sources
- `/api/god/poll`, `/api/god/ack`, `/api/god/agents` — see `mathir_lib/mathir_server.py`'s god-mode route handlers

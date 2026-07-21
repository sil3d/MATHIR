# MATHIR `bin/` — File Reference

This folder contains all the runtime scripts and cross-platform helpers.
After a fresh `git clone`, this is the **only folder needed** to bootstrap
MATHIR on a fresh machine.

## Daemon (the core)

| File | What it does |
|---|---|
| **`mathir_server.py`** | The persistent background process. Loads the embedding model into RAM/VRAM once, then answers JSON-RPC requests over TCP port 7338. **Without this, nothing works.** |
| (MCP server) | See `../INSTALL_FOR_AGENT/` for the smart installer which sets up the MCP server |

| (Stats server) | See `../INSTALL_FOR_DEV/` |
## Client tools (for humans and scripts)

| File | What it does | When to use |
|---|---|---|
| **`mathir_client.py`** | Python CLI: `python mathir_client.py recall "query" -k 5` | Fallback universal — works on all platforms with Python |
| **`mathir_client.ps1`** | PowerShell direct-socket module: `. ./mathir_client.ps1; Search-Mathir "q" -K 5` | **Windows, fastest** (50-150ms, no Python startup) |
| **`mathir_client.sh`** | Bash direct-socket module: `source ./mathir_client.sh; mathir_recall "q" 5` | **Mac/Linux, fastest** (200-450ms via /dev/tcp) |

> **Naming convention:** `mathir_server.py` = the daemon. `mathir_client.{ps1,sh}` = clients that talk to the daemon.

## Auto-start (after PC reboot)

| File | Platform | What it does |
|---|---|---|
| **`auto_start.bat`** | Windows | Starts daemon (cmd.exe) |
| **`auto_start.sh`** | Mac/Linux | Starts daemon (bash) |
| **`auto_start_helpers.ps1`** | Windows | PowerShell version with retry logic + health check + **also starts stats server** |
| **`auto_start_vbs.vbs`** | Windows | VBScript wrapper to run `.bat` hidden (no console window) |
| **`com.mathir.daemon.plist`** | macOS | launchd LaunchAgent (auto-starts on login) |
| **`mathir-daemon.service`** | Linux | systemd user unit (auto-starts on login) |

## Injection / sync (developer tools)

| File | What it does |
|---|---|
| **`mathir_inject.py`** | Injects the MATHIR memory block into all agent `.md` files (agents, commands, skills, docs). Idempotent. |
| **`mathir_sync.py`** | Copies new files from source repo to deployed configs. Safe by default (never overwrites). |

## God-mode orchestration (multi-agent coordination)

Solves the **async coordination gap** between an orchestrator and N worker agents (no native waiting pool).
Uses the daemon's `/api/god/poll` and `/api/god/agents` endpoints (defined in `mathir_lib/mathir_god.py`).

| File | What it does |
|---|---|
| **`god/god_bridge.py`** | Cross-platform polling daemon — 3 modes: `worker` / `orchestrator` / `observer`. Stdlib only. Beeps + logs when new god-events detected. A human still has to act on the notification. |
| **`god/god_poll.ps1`** | PowerShell one-shot poller (Windows, faster boot) |
| **`god/god_poll.sh`** | Bash one-shot poller (POSIX) |
| **`god/god_mode_start.py`** | On-demand launcher — detects installed agent CLIs (`--detect`) and spawns a headless `god_worker_daemon.py` as a detached background process (`--launch <tool> --name <n> --cwd <path>`). Never autostarted — human-triggered only. |
| **`god/god_mode_stop.py`** | Kills headless workers started by `god_mode_start.py`, by `--name` or `--all`. |
| **`god/god_worker_daemon.py`** | The actual headless execution loop: polls `/api/god/poll`, claims a task, spawns the target CLI (opencode/claude/codex/gemini/aider/cursor-agent/copilot/...) with its documented headless flags, streams output live, retries on silent no-op or timeout, then acks `completed`/`failed`. No human needed once launched. |
| **`god/god_mode_report.py`** | Deterministic (LLM-independent) text report — reads the SQLite DB directly and groups results by task, for when the orchestrator's own memory of a dispatch round can't be trusted. |
| **`god/PROTOCOL.md`** | Full label taxonomy (`god:task:…`, `god:result:…`, etc.) + message flow |
| **`god/README.md`** | Usage, env vars, troubleshooting for both the notify-only bridge and the headless workers |

**Quick start (notify-only bridge, worker terminal):**
```bash
python god/god_bridge.py --mode worker --name mimo-code --interval 5
```

**Quick start (notify-only bridge, orchestrator terminal):**
```bash
python god/god_bridge.py --mode orchestrator --interval 5 --project <your-project>
```

**Quick start (headless, unattended worker):**
```bash
python god/god_mode_start.py --launch opencode --name mimo-code --cwd <path> --project <your-project>
python god/god_mode_report.py --cwd <path>   # after dispatching tasks
```

**Env vars** (override per machine):
- `MATHIR_DAEMON_URL` (default `http://localhost:7338`)
- `MATHIR_HOME` (base config/state dir for headless workers, default `~/.config/MATHIR`)
- `MATHIR_STATE_DIR` (bridge-mode state + log dir, default `$XDG_CONFIG_HOME/mathir`)
- `MATHIR_LOG_FILE` (default `$MATHIR_STATE_DIR/god_bridge.log`)

> Cross-platform by design: no hardcoded paths, no env pollution, portable XDG state dir.

## Smart installer

> Moved to `../INSTALL_FOR_AGENT/` and `../INSTALL_FOR_DEV/` to keep this folder lean. The installer scripts
> (`install.bat`, `install.sh`, `install_smart.py`) and platform guides
> live there.

## Benchmarks

| Method | ping | stats | recall | save |
|---|---|---|---|---|
| MCP tool (stdio) | ~10ms | ~15ms | ~50ms | ~80ms |
| PowerShell direct socket | 57ms | 66ms | 149ms | 124ms |
| Bash + /dev/tcp | 239ms | 241ms | 263ms | 324ms |
| Python wrapper | ~200ms | ~50ms | ~50ms | ~80ms (after warm-up) |

## Quick reference — "How do I...?"

| Task | Command |
|---|---|
| Start daemon | `& auto_start_helpers.ps1` (Windows) or `./auto_start.sh` (Mac/Linux) |
| Check if daemon is up | `Test-NetConnection -ComputerName 127.0.0.1 -Port 7338` (PS) or `nc -z 127.0.0.1 7338` (bash) |
| Recall memory | Use MCP tool, or PowerShell module, or `python mathir_client.py recall "q" -k 5` |
| Save memory | Use MCP tool, or PowerShell module, or `python mathir_client.py save "content" -a my_agent -t episodic -l "label"` |
| Inject into all agents | `python mathir_inject.py --apply --target all` |
| Sync source to deployed | `python mathir_sync.py` (dry-run) then `--force` |
| Start god-mode bridge (orchestrator) | `python god/god_bridge.py --mode orchestrator --interval 5` |
| Start god-mode bridge (worker)    | `python god/god_bridge.py --mode worker --name <my-name> --interval 5` |
| Launch a headless god-mode worker | `python god/god_mode_start.py --launch <tool> --name <my-name> --cwd <path>` |
| Get a deterministic god-mode report | `python god/god_mode_report.py --cwd <path>` |
| View dashboard | http://localhost:7420 (after starting stats server) |

## Dependencies

- Python 3.10+
- For GPU: CUDA 12.4+ + cuDNN
- For Mac/Linux shell: `nc` (netcat) or bash 4+ (for /dev/tcp)
- For Windows: PowerShell 5.1+ (built-in)

## Port reference

| Port | Service |
|---|---|
| 7338 | Daemon (JSON-RPC) |
| 7420 | Stats server / Dashboard (HTTP) |
| 8182 | Proxy (legacy) |

## See also

- `../INSTALL_FOR_AGENT/INSTALL_WINDOWS.md` — full Windows install
- `../INSTALL_FOR_AGENT/INSTALL_LINUX.md` — full Linux install
- `../INSTALL_FOR_AGENT/INSTALL_MACOS.md` — full macOS install
- `../opencode/README.md` — template injection system docs

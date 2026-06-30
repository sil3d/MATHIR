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

- `../INSTALL_FOR_DEV/INSTALL_WINDOWS.md` — full Windows install
- `../INSTALL_FOR_DEV/INSTALL_LINUX.md` — full Linux install
- `../INSTALL_FOR_DEV/INSTALL_MACOS.md` — full macOS install
- `../opencode/README.md` — template injection system docs

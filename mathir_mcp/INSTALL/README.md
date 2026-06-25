# MATHIR `install/` — File Reference

This folder contains all the smart installer scripts and platform-specific
installation guides. After a fresh `git clone`, you can install MATHIR with
**one command** from the repo root:

```bash
# Linux / macOS
./install.sh

# Windows (cmd)
install.bat

# Windows (PowerShell)
.\install.bat
```

## Scripts

| File | What it does | Size |
|---|---|---|
| **`install.bat`** | Windows wrapper. Just calls `python install_smart.py` and pauses at end. | 1 KB |
| **`install.sh`** | Mac/Linux wrapper. Calls `python3 install_smart.py`. | 1.3 KB |
| **`install_smart.py`** | The actual installer. 41 KB of Python. **This is the brain.** | 41 KB |

## `install_smart.py` — what it does

1. **Auto-detects** the OS (Windows / macOS / Linux) and the shell
2. **Finds** the user's coding agents by scanning for config files:
   - OpenCode (`opencode.json`)
   - Claude Desktop (`claude_desktop_config.json`)
   - Cursor, Windsurf, Kilo Code, Cline
   - MiMo Code, Zcode
   - GitHub Copilot, Continue.dev, Cody
   - And ~30 more (see `mathir_mcp/docs/AGENT.md` for the full list)
3. **Injects** the MATHIR MCP server config into each agent's config
4. **Injects** the `GLOBAL_INSTRUCTIONS.md` into the agent's system prompt
5. **Installs** the daemon auto-start (Task Scheduler on Win, launchd on Mac, systemd on Linux)
6. **Verifies** everything works at the end

## Full install guides (markdown)

| File | Platform | Length |
|---|---|---|
| `INSTALL_WINDOWS.md` | Windows 10/11/Server | ~350 lines |
| `INSTALL_LINUX.md` | Ubuntu/Debian/Fedora/Arch | ~330 lines |
| `INSTALL_MACOS.md` | macOS 12+ (Intel + Apple Silicon) | ~360 lines |

Each guide includes:
- Prerequisites
- Step-by-step install (manual + one-liner)
- Auto-start setup
- Configuration reference
- Troubleshooting matrix
- Unattended/scripted install (CI, dotfiles)

## Two install paths

### Path 1 — Smart installer (recommended, one command)
```bash
./install.sh       # Mac/Linux
install.bat        # Windows
```
Auto-detects your agents and configures everything.

### Path 2 — Manual (read the INSTALL_*.md)
```bash
# Read the right guide
cat INSTALL/INSTALL_LINUX.md
# Follow step-by-step
```

Use manual if:
- You want to understand what's happening
- The smart installer failed
- You need to install on a CI/provisioning system
- You want to customize the config

## What gets installed

```
~/.config/opencode/bin/         ← daemon, MCP server, helpers
~/.config/opencode/config/      ← mathir.json
~/.config/opencode/data/        ← sqlite-vec DBs (one per project)
~/.config/opencode/GLOBAL_INSTRUCTIONS.md
~/.config/opencode/opencode.json (modified — MCP added)
~/.config/opencode/agents/*.md  (modified — MATHIR block injected)
```

Mac/Linux: same paths but `~/Library/Application Support/` or `~/.config/` depending on tool.

## Run from a clean clone

```bash
git clone https://github.com/sil3d/MATHIR.git
cd MATHIR/mathir_mcp
./install.sh        # or install.bat on Windows
# Done — daemon running, MCP configured, agents updated
```

## After install

| Check | Command |
|---|---|
| Daemon is up | `Test-NetConnection -ComputerName 127.0.0.1 -Port 7338` |
| Stats server | `Test-NetConnection -ComputerName 127.0.0.1 -Port 7420` |
| Dashboard | Open http://localhost:7420 |
| Memory has data | `python mathir_client.py stats` |
| All agents updated | `python mathir_inject.py --check --target all` |

## One-click dashboard launcher

At the **repo root** (`mathir_mcp/`), there are two shortcuts to launch the
dashboard without going through the smart installer:

| File | Platform | What it does |
|---|---|---|
| `mathir_dashboard.bat` | Windows | Starts `mathir_stats_server.py` + opens browser to http://127.0.0.1:7420 |
| `mathir_dashboard.sh` | Mac/Linux | Same (uses `xdg-open` / `open` / `wslview` for cross-platform browser) |

Both scripts auto-resolve paths (no hardcoded `C:\...` or `/home/...`), so
they work from any clone.

## Uninstall

```bash
python install_smart.py --uninstall
```

Removes MCP config from all detected agents, deletes auto-start, leaves
the daemon + DB alone (run `mathir_server.py --uninstall` for those).

## See also

- `../bin/README.md` — runtime scripts reference
- `../INSTALL/INSTALL_*.md` — full platform guides
- `../opencode/README.md` — template injection system

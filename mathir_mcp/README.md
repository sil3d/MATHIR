# MATHIR MCP — Universal Installation

**5-tier cognitive memory for 50 AI coding agents. Install once, use everywhere.**

> **v8.5.0** — FastMCP 3.4.2, 20 MCP tools, auto-injection plugin, unified server.

---

## Quick Start (3 Steps)

```bash
# 1. Copy the repo to a stable location (one time)
git clone https://github.com/sil3d/MATHIR.git
cp -r MATHIR/mathir_mcp ~/.config/MATHIR

# 2. Install deps + run smart installer
pip install -r ~/.config/MATHIR/mathir_lib/requirements.txt
python ~/.config/MATHIR/install_smart.py

# 3. Restart your agent. MATHIR is ready.
```

### Windows launcher
```bat
%~dp0\install.bat
```

### macOS / Linux launcher
```bash
~/.config/MATHIR/install.sh
```

The installer auto-detects 50 coding agents and injects MCP config + instructions.

---

## Documentation Index

| Doc | Purpose |
|---|---|
| **[INSTALL/INSTALL_WINDOWS.md](INSTALL/INSTALL_WINDOWS.md)** | Windows install (daemon, auto-start, MCP wiring) |
| **[INSTALL/INSTALL_LINUX.md](INSTALL/INSTALL_LINUX.md)** | Linux install (systemd user service) |
| **[INSTALL/INSTALL_MACOS.md](INSTALL/INSTALL_MACOS.md)** | macOS install (launchd plist) |
| **[docs/AGENT.md](docs/AGENT.md)** | Per-agent config & troubleshooting |
| **[docs/DAEMON.md](docs/DAEMON.md)** | Daemon protocol, JSON-RPC methods |
| **[docs/DASHBOARD_GUIDE.md](docs/DASHBOARD_GUIDE.md)** | Neural dashboard setup |
| **[docs/GPU_SETUP.md](docs/GPU_SETUP.md)** | GPU/ONNX acceleration |
| **[docs/DIMENSIONS.md](docs/DIMENSIONS.md)** | Embedding model selection |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history |

---

## ⚠️ Moving the Folder After Install

The installer writes an **absolute path**. Re-run `install.bat` / `install.sh` after moving the folder.

---

## Supported Agents (50)

See **[docs/AGENT.md § Supported Agents](docs/AGENT.md)** for the full list (OpenCode, Claude Code, Cursor, Cline, MiMo, Windsurf, Gemini CLI, etc.).

---

## MCP Tools (20)

| Category | Tools |
|---|---|
| **Auto-injection** | `memory_session_start`, `memory_context` |
| **Basic** | `memory_save`, `memory_recall`, `memory_smart_search`, `memory_hybrid_search`, `memory_delete`, `memory_stats`, `memory_audit`, `memory_export`, `memory_sessions`, `memory_dashboard` |
| **Lifecycle** | `memory_promote`, `memory_auto_promote`, `memory_decay`, `memory_consolidate`, `memory_link`, `memory_get_links`, `memory_build_links` |

Full signatures: see **[`mathir_lib/mathir_mcp_server.py`](mathir_lib/mathir_mcp_server.py)** — 20 `@mcp.tool()` decorators.

---

## If Installer Fails

Give the entire `~/.config/MATHIR/` folder to your coding agent. It reads `docs/AGENT.md` and configures MATHIR automatically.

---

## Security & Input Limits

The MCP server enforces strict per-field length caps (DoS protection). Full table: **[docs/DAEMON.md § Security](docs/DAEMON.md)**.

---

## Multilingual Help

If installer fails, give `~/.config/MATHIR/` to your agent. It will read `docs/AGENT.md`.

- **FR** : Si l'installeur échoue, donnez `~/.config/MATHIR/` à votre agent.
- **ES** : Si el instalador falla, dea `~/.config/MATHIR/` a su agente.
- **ZH** : 如果安装失败,把 `~/.config/MATHIR/` 给你的 agent。

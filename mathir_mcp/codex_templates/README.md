# Codex Templates

Portable templates for connecting Codex Desktop / CLI to MATHIR.

## Files

- **`hooks.json`** — drop-in replacement for `~/.codex/hooks.json`. Replace `<YOU>` with your username (Windows) or `<USER>` (macOS/Linux), or use the dedicated `claude_code_hook.py` path from your install.

- **`config.toml.snippet`** — TOML blocks to merge into your existing `~/.codex/config.toml`. Replace the placeholders:
  - `<ABS_PYTHON>` — the absolute path of the Python interpreter that should run the MCP server. Verify with `where.exe python` (Windows) or `which python3` (macOS/Linux). If you use a conda/venv, point at that interpreter specifically.
  - `<HOME>` — your home directory as a Windows backslash path on Windows (`C:\\Users\\<YOU>`) or POSIX path on macOS/Linux (`/Users/<USER>` or `/home/<USER>`).

## Why a template and not a copy?

The Codex `config.toml` and `hooks.json` accept **absolute paths only** because Codex (Electron) does not shell-expand `~` in MCP env vars and does not always inherit `PATH`. Hardcoding any user's path in the template would be misleading.

The official setup guide ([docs/CODEX_INTEGRATION.md](../../../docs/CODEX_INTEGRATION.md)) walks through the substitutions and the three layers of integration.

## What lives where after install

| Template field | Resolves to (after substitution) |
|---|---|
| `<ABS_PYTHON>` | `C:\Users\<YOU>\miniconda3\python.exe` (or wherever `where.exe python` finds first) |
| `<HOME>\.config\MATHIR\mathir_mcp\mathir_lib\mathir_mcp_server.py` | the deployed mathir_mcp_server.py |
| `<HOME>\.config\MATHIR\config\mathir.json` | the daemon runtime config |
| `<HOME>\.config\MATHIR\mathir_mcp\mathir_lib` | the deployed mathir_lib directory (PYTHONPATH for the MCP server) |
| `<HOME>\.config\MATHIR\mathir_mcp\bin\claude_code_hook.py` | the auto-inject hook shared with Claude Code |

## Related docs

- [`../../../docs/CODEX_INTEGRATION.md`](../../../docs/CODEX_INTEGRATION.md) — full setup guide (config + autostart + troubleshooting, all 3 layers).
- [`../troubleshooting/codex-mcp-tools-invisible-three-layers.md`](../troubleshooting/codex-mcp-tools-invisible-three-layers.md) — incident narrative for the 3-layer failure that broke Codex MCP until `mcp.run(show_banner=False)`.

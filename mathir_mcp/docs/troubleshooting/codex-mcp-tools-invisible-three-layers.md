# Codex MCP Tools Invisible — Three-Layer Path to Root Cause

Symptom: "MCP server `mathir` is configured but `mcp__mathir__*` tools never appear
in Codex's tool list, even after restart."

This went through **three** distinct failures before resolving. Documenting
each so the next agent doesn't burn the same time.

> **Companion doc**: [`../../../docs/CODEX_INTEGRATION.md`](../../../docs/CODEX_INTEGRATION.md) — full setup guide (install + config + autostart + cheat sheet).
> This file is the incident narrative; the integration guide is the canonical reference.

---

## Layer 1: `~` not expanded in MCP env vars on Windows

**Symptom**: `from mathir_paths import CONFIG_PATH` fails inside the MCP server
stdio loop, server crashes silently, Codex sees "no tools".

**Cause**: Codex's MCP process spawn does **not** shell-expand `~` in env
vars before passing them to the child. Python's `os.path.expanduser()` only
works for paths encountered *inside* Python, not for env vars at startup.

**Evidence**:
```bash
$ PYTHONPATH='~/.config/MATHIR/mathir_mcp/mathir_lib' python -c \
    "from mathir_paths import CONFIG_PATH"
ModuleNotFoundError: No module named 'mathir_paths'
# Python interpreted it as relative to CWD, joined → D:\path\~\.config\...

$ PYTHONPATH='C:\\Users\\princ\\.config\\MATHIR\\mathir_mcp\\mathir_lib' \
    python -c "from mathir_paths import CONFIG_PATH"
OK — CONFIG_PATH = C:\Users\princ\.config\MATHIR\config\mathir.json
```

**Fix**: replace all `~/.config/MATHIR/...` paths in `~/.codex/config.toml`
`[mcp_servers.mathir.env]` block with absolute `C:\Users\princ\...` paths.
Use escaped backslashes in TOML double-quoted strings:
`"C:\\\\Users\\\\princ\\\\..."`.

---

## Layer 2: `python` not on PATH for Codex's child process

**Symptom**: even after fixing `~` paths, MCP server still not exposed.

**Cause**: Codex (Electron) doesn't inherit the user's PATH for MCP
subprocesses the same way cmd does. `command = "python"` fails to resolve.

**Fix**: use absolute Python path:
```toml
command = "C:\\Users\\princ\\miniconda3\\python.exe"
args = ["C:\\Users\\princ\\.config\\MATHIR\\mathir_mcp\\mathir_lib\\mathir_mcp_server.py"]
startup_timeout_sec = 30
```

`startup_timeout_sec` is added in the same pattern as `[mcp_servers.node_repl]`
for safety.

---

## Layer 3 (the actual root cause): FastMCP banner on stdout

**Symptom**: STILL no `mcp__mathir__*` tools after fixing layers 1+2 and
restarting Codex.

**Cause**: FastMCP 3.4.4 prints a large colored ASCII-art banner on **stdout**
at `mcp.run()`. MCP-over-stdio uses stdout for JSON-RPC frames, so this
corrupts the framing and the host's `initialize` or `tools/list` handshake
silently fails before the server even gets to enumerate tools.

**Evidence** (only visible when you actually run the server with stdio piped):
```
+-----------------------------------------------------------------------------+
|                        ▄▀▀ ▄▀▀█ █▀▀█ ▀▄▄ █▀▀█ █▄▄▀ ...                        |
|                                 FastMCP 3.4.4                                |
|                             https://gofastmcp.com                            |
|                  🖥  Server:      mathir-mcp, 3.4.4                           |
|                  🚀 Deploy free: https://horizon.prefect.io                  |
+-----------------------------------------------------------------------------+
[07/31/26 00:27:56] INFO     Starting MCP server 'mathir-mcp'
                             with transport 'stdio'
```

**Fix**: `mathir_mcp_server.py:1490`
```python
# before:
mcp.run()
# after:
mcp.run(show_banner=False)
```

stderr logging is unaffected. All daemon / cache / sanitizer messages still
visible on stderr.

---

## Verification of the layered fix

With all three fixes applied:
1. Run server with the exact command Codex uses:
   ```bash
   C:\Users\princ\miniconda3\python.exe \
       C:\Users\princ\.config\MATHIR\mathir_mcp\mathir_lib\mathir_mcp_server.py \
       < initialize.json + tools/list.json > response.json
   ```
   Returns 27 tools, zero stdout banner, valid JSON-RPC.
2. Restart Codex Desktop fully (not just new task — kill the Electron process).
3. Codex's tools/list now contains `mcp__mathir__memory_recall`, `memory_save`,
   `mathir_health`, etc.

---

## Related lessons

- **Codex caches MCP server config at process startup.** Editing
  `~/.codex/config.toml` does nothing until the Codex process is killed
  and relaunched. "Open new task" is NOT sufficient.
- **stderr vs stdout in MCP stdio**: only stdout is the JSON-RPC channel.
  Anything that writes to stdout at any point in the server's lifetime
  breaks framing. Always run stdio MCP servers with `show_banner=False`,
  `print(..., file=sys.stderr)`, and never `print()` for any reason.
- **guardrail-sync-deployed-daemon**: when Codex patched only the deployed
  copy of `mathir_mcp_server.py`, the source-of-truth repo still had the
  bug. A subsequent `pip install -e .` or `git pull` would silently undo
  the fix. Always propagate MCP-server fixes back to the repo immediately.

## Files touched in this incident

- `~/.codex/config.toml` — added `[mcp_servers.mathir]` block (3 fix passes)
- `~/.codex/config.toml.bak.codex-mcp-fix-20260731` — backup after pass 1
- `~/.codex/config.toml.bak.codex-mcp-fix-20260731-pre-python` — backup after pass 2
- `~/.codex/config.toml.mathir-bad-inject-20260730-2335.bak` — pre-existing
- `~/.codex/hooks.json` — pre-existing, points at `claude_code_hook.py` (works)
- `mathir_mcp/mathir_lib/mathir_mcp_server.py:1490` — `mcp.run(show_banner=False)`
  in BOTH repo and deployed copies
- `mathir_mcp/CHANGELOG.md` — Unreleased entry added
- `mathir_mcp/bin/auto_start.bat` — proxy target changed `anthropic.com` → `openai.com`
  (Codex uses OpenAI-format Chat Completions, not Anthropic-format)

# Codex MCP Server Path Expansion Pitfall (Windows) — 2026-07-31

## Symptom
Codex says "I see MCP servers in settings but the mathir tools aren't exposed" — the
configured MCP server silently fails to start (no error visible in Codex UI).

## Root cause (verified by reproducing)
The previous config used shell-style `~` expansion in MCP env vars:

```toml
[mcp_servers.mathir.env]
MATHIR_CONFIG = "~/.config/MATHIR/config/mathir.json"
PYTHONPATH    = "~/.config/MATHIR/mathir_mcp/mathir_lib"
```

On Windows, **Codex's MCP spawn does NOT expand `~` before passing env vars to the child
process**. Python then interprets `~/.config/MATHIR/...` literally:
- `sys.path` contains an unresolved user-home path joined to the current working directory
- `from mathir_paths import CONFIG_PATH` → `ModuleNotFoundError`
- The MCP server exits before its first stdio handshake → Codex sees "no tools"

Verified empirically:
```bash
$ PYTHONPATH='~/.config/MATHIR/mathir_mcp/mathir_lib' python -c \
    "from mathir_paths import CONFIG_PATH"
# ModuleNotFoundError: No module named 'mathir_paths'
$ PYTHONPATH='<ABSOLUTE_MATHIR_HOME>/mathir_mcp/mathir_lib' python -c \
    "from mathir_paths import CONFIG_PATH"
# OK — CONFIG_PATH resolves to <ABSOLUTE_MATHIR_HOME>/config/mathir.json
```

## Fix
Replace `~/.config/MATHIR/...` with absolute paths to the deployed
`<MATHIR_HOME>/...` tree in `~/.codex/config.toml` `[mcp_servers.mathir.env]`.
Also add `MATHIR_DAEMON_URL = "http://127.0.0.1:7338"` for consistency.

Codex requires absolute paths for these child-process environment values on
Windows. `<MATHIR_HOME>` and `<ABSOLUTE_PYTHON_EXE>` are documentation
placeholders; substitute the real paths on the target machine.

## Action required from user
Open a NEW Codex task to pick up the env change (Codex doesn't hot-reload MCP servers).
Confirm by asking Codex to list MCP tools — should see 27 (mathir_dashboard etc.).

## Related
- Codex binary: installed separately; use the configured Codex executable
- Codex config: `~/.codex/config.toml`
- Codex hooks: `~/.codex/hooks.json`
- Pre-existing bak: `config.toml.mathir-bad-inject-20260730-2335.bak` (17784 bytes —
  earlier failed injection attempt — unrelated, keep as-is)
- New backup made during this fix: `config.toml.bak.codex-mcp-fix-20260731`

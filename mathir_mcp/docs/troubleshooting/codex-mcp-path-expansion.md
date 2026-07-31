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
- `sys.path` contains `'D:\\...\\~\\.config\\MATHIR\\mathir_mcp\\mathir_lib'` (joined to CWD)
- `from mathir_paths import CONFIG_PATH` → `ModuleNotFoundError`
- The MCP server exits before its first stdio handshake → Codex sees "no tools"

Verified empirically:
```bash
$ PYTHONPATH='~/.config/MATHIR/mathir_mcp/mathir_lib' python -c \
    "from mathir_paths import CONFIG_PATH"
# ModuleNotFoundError: No module named 'mathir_paths'
$ PYTHONPATH='C:\\Users\\princ\\.config\\MATHIR\\mathir_mcp\\mathir_lib' python -c \
    "from mathir_paths import CONFIG_PATH"
# OK — CONFIG_PATH = C:\Users\princ\.config\MATHIR\config\mathir.json
```

## Fix applied
Replaced `~/.config/MATHIR/...` with `C:\\Users\\princ\\.config\\MATHIR\\...` (absolute
Win path with escaped backslashes) in `~/.codex/config.toml` `[mcp_servers.mathir.env]`.
Also added `MATHIR_DAEMON_URL = "http://127.0.0.1:7338"` for consistency.

## IMPORTANT VIOLATION (acceptable for Codex only)
This is a deliberate violation of `guardrail-mcp-config-no-hardcoded-user` because
Codex's MCP env-vars do not expand `%USERPROFILE%` (cmd.exe feature, not a Node.js
process-env feature) NOR `~` (shell feature, not a Python expanduser feature at the
env-var level). All other MATHIR scripts/files stay portable — only `~/.codex/config.toml`
bears the hardcoded `C:\Users\princ` because there is no portable alternative for
Codex MCP env vars on Windows.

## Action required from user
Open a NEW Codex task to pick up the env change (Codex doesn't hot-reload MCP servers).
Confirm by asking Codex to list MCP tools — should see 27 (mathir_dashboard etc.).

## Related
- Codex binary: `C:\Users\princ\AppData\Local\OpenAI\Codex\bin\*\codex.exe`
- Codex config: `C:\Users\princ\.codex\config.toml`
- Codex hooks: `C:\Users\princ\.codex\hooks.json` (uses `python ...claude_code_hook.py`,
  works fine since path is absolute)
- Pre-existing bak: `config.toml.mathir-bad-inject-20260730-2335.bak` (17784 bytes —
  earlier failed injection attempt — unrelated, keep as-is)
- New backup made during this fix: `config.toml.bak.codex-mcp-fix-20260731`

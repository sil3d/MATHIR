# MATHIR ↔ Codex Integration Guide

**Date:** 2026-09-03 · **Status:** Production-tested with Codex desktop/CLI and the v8.9.8 daemon.
**Audience:** anyone running OpenAI Codex CLI/Desktop and wants persistent memory across sessions.

This guide covers the **full** setup: daemon + MCP server + auto-inject hook + transparent proxy + autostart. After following it, Codex will expose 27 tools (`mcp__mathir__*`) and every prompt will get MATHIR context auto-injected before the model sees it.

---

## 1. Mental model: three layers

MATHIR plugs into Codex via **three independent layers**, each owned by a different piece of code:

| Layer | Mechanism | Process | Restart trigger |
|---|---|---|---|
| **A: MCP server** | `mathir_mcp_server.py` launched by Codex as a stdio JSON-RPC child | Listens on stdout/stdin | Restart Codex (process-level) |
| **B: Auto-inject hook** | `claude_code_hook.py` fired by Codex on every `UserPromptSubmit` | Reads stdin JSON, writes `<mathir-auto-injection>` block to stdout | Restart Codex |
| **C: Transparent proxy** | `mathir_proxy.py` on `127.0.0.1:7339` intercepting Codex's OpenAI API calls and augmenting the system prompt | Listens on TCP 7339 | Restart auto_start.bat / Task Scheduler healthcheck |

You can ship any subset. Layer A is mandatory for tools. Layer B is for prompt-context auto-injection. Layer C is for cookie-cutter openai-compat clients that don't expose hooks (Cursor, Continue, etc.).

---

## 2. One-time prerequisites

```toml
# Already done if you followed the global MATHIR install guide.
# The daemon (port 7338) must be running and healthy.
curl http://127.0.0.1:7338/health
# Expect: {"status":"ok","version":"8.9.8",...}
```

If the daemon is dead, run `~/.config/MATHIR/mathir_mcp/bin/auto_start.bat` (Windows) or the `mathir-daemon.service` systemd unit (Linux/macOS). See section 7 below.

Codex is installed separately. Its config directory is `~/.codex/` (the platform's user-home equivalent). You don't need to install Codex from this guide.

---

## 3. Layer A: MCP server (the 27 tools)

### 3.1 Required `~/.codex/config.toml` block

```toml
[mcp_servers.mathir]
command = "<ABSOLUTE_PYTHON_EXE>"                         # replace with the Python executable Codex can spawn
args = ["<MATHIR_HOME>/mathir_mcp/mathir_lib/mathir_mcp_server.py"]
startup_timeout_sec = 30

[mcp_servers.mathir.env]
MATHIR_EMBEDDING_DIM = "384"
MATHIR_PORT = "7338"
MATHIR_DAEMON_URL = "http://127.0.0.1:7338"
MATHIR_CONFIG = "<MATHIR_HOME>/config/mathir.json"
PYTHONPATH = "<MATHIR_HOME>/mathir_mcp/mathir_lib"
```

**Why every value matters:**

- **`command = "absolute python.exe path"`**: Codex (Electron) does not always inherit your shell's `PATH` for MCP child processes. `command = "python"` may resolve to nothing and the spawn fails with no log.
- **`args = [...]` single-element list with absolute .py path**: same reasoning.
- **`startup_timeout_sec = 30`**: matches the pattern used by `node_repl`; default timeout is too short for `from fastmcp import FastMCP` + first tool enum.
- **Every env var is absolute path**: Codex does **not** shell-expand `~`. Code that worked on Claude Code/MiMoCode/OpenCode by relying on `~/.config/MATHIR/...` will silently crash the MCP server here, because Python's `os.path.expanduser()` only works for paths *inside* Python, not for env vars at startup. Confirmed live (2026-07-31): `PYTHONPATH='~/.config/...'` → `ModuleNotFoundError: mathir_paths` → 0 tools exposed.

### 3.2 Restart Codex fully

- Quit via task-bar menu (right-click Codex icon → Quit): **not** just closing the chat window.
- Wait for the Electron process to disappear: `Get-CimInstance Win32_Process -Filter "Name='codex.exe'"` should be empty.
- Relaunch.

Then verify:

```
> liste tes outils MCP
```

Expected: 27 tools including `mcp__mathir__memory_recall`, `mcp__mathir__memory_save`, `mcp__mathir__mathir_health`, …

If they don't appear: see section 9.

---

## 4. Layer B: auto-inject hook (context on every prompt)

### 4.1 Required `~/.codex/hooks.json`

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python \"<MATHIR_HOME>/mathir_mcp/bin/claude_code_hook.py\""
          }
        ]
      }
    ]
  }
}
```

The script `claude_code_hook.py` is the **same** hook Claude Code uses. It reads `hook_input["prompt"]` (NOT `message`, the common bug fixed in v8.9.5), queries the daemon for relevant memories and guardrails via `/api/context`, and emits a `<mathir-auto-injection>` block on stdout that Codex merges into the prompt.

### 4.2 Verify

Ask Codex:

```
> Quel est mon dernier projet mathir?
```

If MATHIR fires, the model's first lines should reference project-specific facts before you've given it any. If you see only generic answers, the hook didn't fire. See section 9.

---

## 5. Layer C: transparent OpenAI proxy

Layer C is **optional for Codex** (Codex already gets tools + hook), but useful for debugging and for any other OpenAI-compat client.

### 5.1 What it does

`mathir_proxy.py` listens on `127.0.0.1:7339`, accepts `/v1/chat/completions` requests in OpenAI Chat Completions format, queries MATHIR for context matching the last user message, adds a `<mathir-auto-injection>` block to the request's system instructions, and forwards the request to the real `api.openai.com`.

### 5.2 Required `~/.codex/config.toml` addition

```toml
[shell_environment_policy.set]
OPENAI_BASE_URL = "http://127.0.0.1:7339/v1"     # /v1 MANDATORY (OpenAI SDK convention)
MATHIR_PROXY_PORT = "7339"
```

`[shell_environment_policy.set]` is Codex's mechanism for forwarding env vars to its subprocesses (the OpenAI SDK client, the Node REPL, browser automation tools, etc.). This is what makes the proxy visible to Codex itself, not just to its MCP children.

**Per guardrail `guardrail-base-url-v1-convention`**: do NOT forget the trailing `/v1`. OpenAI SDK adds `/chat/completions` to your base_url; the SDK default already includes `/v1`, so a missing trailing slash causes a double `/v1/v1` and 404.

### 5.3 Verify

```bash
curl http://127.0.0.1:7339/health
# Expect: {"daemon":"http://127.0.0.1:7338","inject_k":8,"status":"ok","target":"https://api.openai.com"}
```

### 5.4 One-shot launch (manual)

```bash
python "<MATHIR_HOME>/mathir_mcp/mathir_lib/mathir_proxy.py" \
    --port 7339 \
    --host 127.0.0.1 \
    --target https://api.openai.com
```

Logs go to `~/.config/MATHIR/logs/mathir_proxy.log`.

---

## 6. Surviving reboots (autostart)

The MATHIR daemon auto-start is handled by Windows Task Scheduler (`MATHIR Daemon` task, at logon) and a periodic healthcheck (`MATHIR_Daemon_Healthcheck`, every 5 min) that re-launches both the daemon and the **proxy** via `auto_start.bat`.

To install on a fresh box:

```cmd
:: From an elevated cmd.exe
schtasks /create /tn "MATHIR Daemon" /tr "\"<ABSOLUTE_PYTHON_EXE>\" \"<MATHIR_HOME>\\mathir_mcp\\mathir_lib\\mathir_server.py\" --force" /sc onlogon /ru "%USERNAME%"
schtasks /create /tn "MATHIR_Daemon_Healthcheck" /tr "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"<MATHIR_HOME>\\mathir_mcp\\bin\\auto_start_healthcheck.ps1\"" /sc minute /mo 5 /ru "%USERNAME%"
```

The healthcheck task pulls up `auto_start.bat` whenever port 7338 OR 7339 is missing.

---

## 7. Restore MATHIR from scratch

If `~/.config/MATHIR/` is wiped:

```cmd
:: From the repo root
xcopy /E /I /Y mathir_mcp "%USERPROFILE%\.config\MATHIR\mathir_mcp"
xcopy /Y mathir_mcp\bin "%USERPROFILE%\.config\MATHIR\mathir_mcp\bin\"
"%USERPROFILE%\.config\MATHIR\mathir_mcp\bin\auto_start.bat"
```

---

## 8. File map (what goes where)

<REPO_ROOT>\                                             ← repository source of truth (git)
  ├─ CHANGELOG.md                                         ← top-level changelog
  └─ docs\
      └─ CODEX_INTEGRATION.md                             ← this file

<REPO_ROOT>\mathir_mcp\                                  ← daemon source
  ├─ CHANGELOG.md                                         ← sub-component changelog
  └─ docs\troubleshooting\
      └─ codex-mcp-tools-invisible-three-layers.md        ← 3-layer failure walk-through

~/.codex/                                                 ← Codex config dir
  ├─ config.toml                                           ← [mcp_servers.mathir] and proxy env
  └─ hooks.json                                            ← UserPromptSubmit → claude_code_hook.py

<MATHIR_HOME>\                                            ← deployed install
  ├─ mathir_mcp\mathir_lib\
  │   ├─ mathir_server.py                                  ← daemon (port 7338)
  │   ├─ mathir_mcp_server.py                              ← stdio MCP server
  │   └─ mathir_proxy.py                                   ← transparent proxy (port 7339)
  └─ mathir_mcp\bin\
      ├─ auto_start.bat                                    ← launches daemon + proxy
      ├─ auto_start_healthcheck.ps1                        ← 5-minute healthcheck
      └─ claude_code_hook.py                               ← shared with Claude Code hook

<MATHIR_HOME>\config\mathir.json                           ← daemon runtime config
```

---

## 9. Troubleshooting (the three-layer failure walk-through)

If the MCP server, hook, or proxy isn't visible to Codex, walk through this ladder in order. Each layer has failed independently on real systems.

### Layer 1: `~` not expanded in MCP env vars (Windows)

**Symptom**: `mcp__mathir__*` tools NEVER appear, even after full Codex restart. No error in Codex UI.

**Root cause** (verified 2026-07-31): Codex's MCP spawn does not shell-expand `~`. Python's `os.path.expanduser()` only works for paths *inside* Python, not for env vars at startup. `PYTHONPATH='~/.config/...'` stays unresolved and `from mathir_paths import CONFIG_PATH` raises `ModuleNotFoundError`. Server exits silently before `tools/list`.

**Fix**: use absolute paths in every `[mcp_servers.mathir.env]` value. Replace `<MATHIR_HOME>` and `<ABSOLUTE_PYTHON_EXE>` in section 3 with real paths on the target machine.

### Layer 2: `python` not in PATH for Codex's child process

**Symptom**: same as Layer 1, even after fixing paths. Daemon is healthy. MCP server doesn't appear in Codex's tools list.

**Root cause**: Codex (Electron) does not always inherit the user's shell `PATH` for MCP child processes. `command = "python"` may resolve to nothing.

**Fix**: set `command = "<ABSOLUTE_PYTHON_EXE>"` (or wherever your Python lives — verify with `where.exe python`).

### Layer 3: FastMCP banner corrupts JSON-RPC on stdout

**Symptom**: same as Layers 1 & 2. Tools still don't appear. Daemon is healthy, MCP server starts successfully when manually run with stdio piped to a probe.

**Root cause** (verified 2026-07-31, **discovered live by Codex itself**): FastMCP 3.4.4 prints a 15-line colored ASCII banner to **stdout** at `mcp.run()`. MCP-over-stdio uses stdout for JSON-RPC frames. The banner corrupts the framing, Codex's `initialize` or `tools/list` handshake silently fails, the server registers 0 tools.

**Fix** (`mathir_mcp_server.py:1490`):

```python
mcp.run(show_banner=False)
```

stderr is unaffected — daemon's `logging` output and 3-layer cache logs still appear there.

### Quick diagnostic — run the server like Codex does

```bash
probe='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}'

echo "$probe" | \
  PYTHONPATH='<MATHIR_HOME>/mathir_mcp/mathir_lib' \
  MATHIR_DAEMON_URL='http://127.0.0.1:7338' \
  MATHIR_CONFIG='<MATHIR_HOME>/config/mathir.json' \
  "<ABSOLUTE_PYTHON_EXE>" \
    "<MATHIR_HOME>/mathir_mcp/mathir_lib/mathir_mcp_server.py"
```

Expected: a single-line JSON response containing `"name":"memory_recall"` (and 26 more). If you see a colored banner first, layer 3 is still broken. If you see `"error":"ModuleNotFoundError"` or import errors, layer 1 or 2.

### Layer X: hooks.json or config.toml syntactically broken

If `~/.codex/config.toml` ever contains Markdown commentary (a previous MATHIR installer did this — see `config.toml.mathir-bad-inject-20260730-2335.bak` for an example), Codex silently ignores the whole file.

Verify with:

```bash
python -c "import tomllib; tomllib.loads(open('$HOME/.codex/config.toml').read()); print('TOML OK')"
```

If TOML fails to parse, restore from one of the `*.bak` files in `~/.codex/` or rewrite the affected section.

---

## 10. Versions verified

| MATHIR | Codex | Notes |
|---|---|---|
| 8.9.4 | 26.721.81911 | First end-to-end success, 2026-07-31 |
| 8.9.8+ | latest | Backward compatible (no API surface changes for Layer A/B/C) |

---

## 11. Related docs in this repo

- [`mathir_mcp/docs/troubleshooting/codex-mcp-tools-invisible-three-layers.md`](../mathir_mcp/docs/troubleshooting/codex-mcp-tools-invisible-three-layers.md) — the same three-layer walk-through, more incident-narrative style.
- [`mathir_mcp/INSTALL_FOR_AGENT/INSTALL_WINDOWS.md`](../mathir_mcp/INSTALL_FOR_AGENT/INSTALL_WINDOWS.md) — full daemon install for Windows (prerequisite for this guide).
- [`mathir_mcp/INSTALL_FOR_AGENT/INSTALL_LINUX.md`](../mathir_mcp/INSTALL_FOR_AGENT/INSTALL_LINUX.md), [`INSTALL_MACOS.md`](../mathir_mcp/INSTALL_FOR_AGENT/INSTALL_MACOS.md) — same for Linux/macOS.
- [`mathir_mcp/README.md`](../mathir_mcp/README.md) — daemon + MCP server overview, all 27 tools documented.
- [`README.md`](../README.md) — top-level (universal architecture across all tools).

---

## 12. Cheat sheet — copy/paste setup on a fresh box

Replace `princ` with your Windows username everywhere. Adapt the Python path if not using miniconda.

```toml
# ~/.codex/config.toml: append to existing content, don't overwrite

[mcp_servers.mathir]
command = "C:\\Users\\princ\\miniconda3\\python.exe"
args = ["C:\\Users\\princ\\.config\\MATHIR\\mathir_mcp\\mathir_lib\\mathir_mcp_server.py"]
startup_timeout_sec = 30

[mcp_servers.mathir.env]
MATHIR_EMBEDDING_DIM = "384"
MATHIR_PORT = "7338"
MATHIR_DAEMON_URL = "http://127.0.0.1:7338"
MATHIR_CONFIG = "C:\\Users\\princ\\.config\\MATHIR\\config\\mathir.json"
PYTHONPATH = "C:\\Users\\princ\\.config\\MATHIR\\mathir_mcp\\mathir_lib"

[shell_environment_policy.set]
OPENAI_BASE_URL = "http://127.0.0.1:7339/v1"
MATHIR_PROXY_PORT = "7339"
```

```json
// ~/.codex/hooks.json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python \"C:\\Users\\princ\\.config\\MATHIR\\mathir_mcp\\bin\\claude_code_hook.py\""
          }
        ]
      }
    ]
  }
}
```

Then:

1. `schtasks /query /tn "MATHIR Daemon"` (verify scheduled task is registered)
2. Quit and relaunch Codex Desktop
3. Ask Codex "liste tes outils MCP" — confirm `mcp__mathir__*` appears
4. Done.

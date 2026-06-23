# MATHIR Install — Windows 10 / 11 / Server 2019+

**Audience:** developers running OpenCode on Windows.
**Time:** ~10 minutes.
**Result:** `mathir_daemon.py` running on `127.0.0.1:7338`, restarting on logon, and registered as an MCP server in `opencode.json`.

> **What "install MATHIR" actually means here:**
> 1. A long-running **daemon** holds the embedding model in RAM (port 7338) — *this* is what you auto-start.
> 2. A short-lived **MCP server** (stdio JSON-RPC) is launched by OpenCode on demand — it just connects to the daemon.
> 3. The MCP server's path is added to your `opencode.json` under `mcp.mathir.command`.
>
> You are not "installing" the daemon's Python code per se — you're **copying the `mathir_lib/` package** to a stable location and registering it with the OS to start on logon.

---

## 0. Prerequisites

| Requirement | Check | Install |
|---|---|---|
| **Python 3.10+** | `python --version` | https://python.org (tick "Add to PATH") |
| **pip** | `python -m pip --version` | bundled with Python |
| **~1.5 GB free RAM** | Task Manager → Performance | embedding model is held in RAM |
| **~600 MB free disk** | `Get-PSDrive C` | for the model + sqlite-vec DB |
| **Port 7338 free** | `Test-NetConnection -ComputerName localhost -Port 7338 -InformationLevel Quiet` (should be `False`) | close the conflicting process, or set `MATHIR_PORT=7339` everywhere |
| **OpenCode already installed** | `opencode --version` | https://opencode.ai |

---

## 1. Choose a stable install path

OpenCode expands `~` to your user profile, so the convention is:

```
~  =  C:\Users\<YOU>\.config\
```

We will install to:

```
C:\Users\<YOU>\.config\opencode\bin\          (the daemon + helpers)
C:\Users\<YOU>\.config\opencode\config\        (mathir.json)
C:\Users\<YOU>\.config\opencode\data\          (sqlite-vec DBs, one per project)
```

Replace `<YOU>` with your actual Windows username (`echo %USERNAME%`).

**Cross-platform note:** OpenCode's `~` works on Windows the same way it does on Unix. The `opencode.json` snippet below uses `~`-relative paths so the same config travels with you to WSL/Linux/macOS.

---

## 2. Copy the MATHIR Python package

From the repo root (`D:\SECRET_PROJECT\MATHIR\mathir_mcp\`):

```powershell
$dest = "$env:USERPROFILE\.config\opencode\bin"
New-Item -ItemType Directory -Path $dest -Force | Out-Null
Copy-Item -Path ".\mathir_lib" -Destination $dest -Recurse -Force
Write-Output "Copied to $dest\mathir_lib\"
```

You should now have:

```
C:\Users\<YOU>\.config\opencode\bin\mathir_lib\
    mathir_daemon.py          ← the background process
    mathir_mcp_server.py      ← the MCP stdio server
    mathir_inject.py          ← template injection tool
    requirements.txt
    ... (other modules)
```

---

## 3. Install Python dependencies

The daemon uses `torch`, `sentence-transformers`, `sqlite-vec`, and `rank-bm25`. The first run downloads a ~120 MB embedding model.

```powershell
python -m pip install --upgrade pip
python -m pip install -r "$env:USERPROFILE\.config\opencode\bin\mathir_lib\requirements.txt"
```

If `torch` is too heavy for your box, see `docs/GPU_SETUP.md` for the ONNX fallback.

---

## 4. Start the daemon (manual, one-shot)

Verify the install works before wiring up auto-start:

```powershell
# Launch in a hidden window so it survives the shell closing
Start-Process python `
  -ArgumentList "$env:USERPROFILE\.config\opencode\bin\mathir_lib\mathir_daemon.py" `
  -WindowStyle Hidden
Start-Sleep -Seconds 5
```

**Verify:**

```powershell
Test-NetConnection -ComputerName localhost -Port 7338 -InformationLevel Quiet
```

Expected output: `True`. If `False`, wait 5 more seconds (model load is slow on first boot) and retry. Tail the daemon's log:

```powershell
Get-EventLog -LogName Application -Source "Python" -Newest 20  -ErrorAction SilentlyContinue
# Or just run the daemon in the foreground to see errors:
python "$env:USERPROFILE\.config\opencode\bin\mathir_lib\mathir_daemon.py"
```

**Stop the manual daemon** before setting up auto-start (to avoid port conflicts):

```powershell
Get-Process python -ErrorAction SilentlyContinue |
  Where-Object { $_.MainWindowTitle -eq "" -and $_.Path -like "*python*" } |
  Stop-Process -Force
```

Or more surgically:

```powershell
Get-NetTCPConnection -LocalPort 7338 -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object { Stop-Process -Id $_ -Force }
```

---

## 5. Auto-start: Scheduled Task (recommended)

Windows has no idiomatic user-level autostart for background daemons — Task Scheduler is the standard answer. It survives reboot, runs at logon with your user account, and restarts on failure.

**Option A — one-liner (recommended):**

```powershell
$action  = New-ScheduledTaskAction `
  -Execute "python" `
  -Argument "`"$env:USERPROFILE\.config\opencode\bin\mathir_lib\mathir_daemon.py`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
  -TaskName "MATHIR Daemon" `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -RunLevel Highest `
  -Description "MATHIR cognitive memory daemon (port 7338)" `
  -Force
```

**Option B — Task Scheduler GUI:**

1. `Win + R` → `taskschd.msc` → **Create Task…**
2. **General** tab → Name: `MATHIR Daemon` → "Run with highest privileges" ✓
3. **Triggers** tab → New… → "At log on" → your username
4. **Actions** tab → New… →
   - Program: `python`
   - Arguments: `"C:\Users\<YOU>\.config\opencode\bin\mathir_lib\mathir_daemon.py"`
5. **Conditions** tab → uncheck "Start only if on AC power"
6. **Settings** tab →
   - "Allow task to be run on demand" ✓
   - "If the task fails, restart every: 1 minute" → "Attempt to restart up to: 3 times"
7. **OK** → enter your password to confirm

**Verify the task:**

```powershell
Get-ScheduledTask -TaskName "MATHIR Daemon" | Select-Object TaskName, State
# Manually trigger it once to test
Start-ScheduledTask -TaskName "MATHIR Daemon"
Start-Sleep -Seconds 5
Test-NetConnection -ComputerName localhost -Port 7338 -InformationLevel Quiet
```

Expected: `State = Ready` and `Test-NetConnection` returns `True`.

**Uninstall later:**

```powershell
Unregister-ScheduledTask -TaskName "MATHIR Daemon" -Confirm:$false
```

---

## 6. Manual start: `mathir.bat` (no auto-start)

If you'd rather start the daemon by hand each session (e.g. on a shared dev box), drop a `mathir.bat` somewhere on your `PATH` (`C:\Users\<YOU>\.bin\` works):

```bat
@echo off
setlocal
set "DAEMON=%USERPROFILE%\.config\opencode\bin\mathir_lib\mathir_daemon.py"
if not exist "%DAEMON%" (
  echo MATHIR not installed at: %DAEMON%
  exit /b 1
)
echo Starting MATHIR daemon (Ctrl+C to stop)...
python "%DAEMON%"
endlocal
```

Then `Win + R` → `mathir.bat` → keep the window open.

---

## 7. Register the MCP server in `opencode.json`

The daemon is running, but OpenCode doesn't know about it yet. Edit `%USERPROFILE%\.config\opencode\opencode.json` and add (or merge) this block under the top-level `"mcp"` key:

```jsonc
"mcp": {
  // ...other MCP servers (tauri, playwright, ...)...

  "mathir": {
    "type": "local",
    "command": [
      "python",
      "~/.config/opencode/bin/mathir_lib/mathir_mcp_server.py"
    ],
    "environment": {
      "MATHIR_EMBEDDING_DIM": "384",
      "MATHIR_PORT": "7338",
      "MATHIR_CONFIG": "~/.config/opencode/config/mathir.json",
      "PYTHONPATH": "~/.config/opencode/bin/mathir_lib"
    },
    "enabled": true
  }
}
```

**Why `~` and not `C:\Users\<YOU>\.config\`?** OpenCode expands `~` at runtime, so the same JSON works in WSL, on a Raspberry Pi, and on a Mac without edits. Hardcoding the Windows path is exactly the bug this guide exists to prevent.

**Validate the JSON** (PowerShell 5.1):

```powershell
try {
  Get-Content "$env:USERPROFILE\.config\opencode\opencode.json" -Raw |
    ConvertFrom-Json | Out-Null
  Write-Output "[OK] opencode.json is valid JSON"
} catch {
  Write-Output "[FAIL] $_"
  exit 1
}
```

**Restart OpenCode** so it picks up the new MCP server.

---

## 8. End-to-end verification

```powershell
# 1. Daemon is up
Test-NetConnection -ComputerName localhost -Port 7338 -InformationLevel Quiet
# expected: True

# 2. JSON-RPC ping (the daemon speaks a tiny TCP protocol, not HTTP,
#    so this only confirms the port is bound; the real test is via OpenCode)
$client = New-Object System.Net.Sockets.TcpClient
$client.Connect("127.0.0.1", 7338)
Write-Output "Connected: $($client.Connected)"
$client.Close()

# 3. OpenCode picks it up — open the OpenCode TUI and ask the agent:
#    "Recall what you know about my last session."
#    If MCP is wired correctly, the agent will call memory_recall and
#    get a response. If not, you'll see "MCP server mathir failed to start"
#    in the OpenCode console.
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Test-NetConnection` returns `False` after 30s | Embedding model still loading (1st run) | Wait 30s more. The model is cached after the first run. |
| `OSError: [Errno 98] Address already in use` | Another process on port 7338 | `Get-NetTCPConnection -LocalPort 7338` → kill the PID. Or set `MATHIR_PORT=7339` in the task + opencode.json. |
| `ModuleNotFoundError: sentence_transformers` | Deps not installed | Re-run step 3. |
| `sqlite-vec` install fails on Python 3.13 | No prebuilt wheel for 3.13 yet | Use Python 3.11 or 3.12. `py -3.11 -m venv ...` |
| Scheduled task runs but daemon dies immediately | `python` not on `PATH` for the task user | Use full path: `C:\Users\<YOU>\AppData\Local\Programs\Python\Python311\python.exe` |
| OpenCode says "MCP server mathir failed to start" | `PYTHONPATH` missing or `~` not expanding | Use the full Windows path in `command` as a debugging step, then re-add `~`. |
| Daemon uses 100% CPU at idle | First run is indexing the model | Normal for ~30s on first launch; drops to ~0.3% after. |
| Want to nuke and reinstall | — | `Remove-Item -Recurse -Force "$env:USERPROFILE\.config\opencode\bin\mathir_lib"` then re-do steps 2-5. |

---

## 10. Unattended / scripted install (CI, provisioning)

```powershell
# install_mathir_windows.ps1 — run as the target user, not Administrator
$ErrorActionPreference = "Stop"
$repo       = "D:\SECRET_PROJECT\MATHIR\mathir_mcp"
$dest       = "$env:USERPROFILE\.config\opencode\bin"
$configRoot = "$env:USERPROFILE\.config\opencode"

# 1. Copy
New-Item -ItemType Directory -Path $dest -Force | Out-Null
Copy-Item -Path "$repo\mathir_lib" -Destination $dest -Recurse -Force

# 2. Deps
python -m pip install -r "$dest\mathir_lib\requirements.txt"

# 3. Scheduled task
$action  = New-ScheduledTaskAction -Execute "python" -Argument "`"$dest\mathir_lib\mathir_daemon.py`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "MATHIR Daemon" -Action $action -Trigger $trigger -Settings $settings -Force

# 4. Smoke test
Start-ScheduledTask -TaskName "MATHIR Daemon"
Start-Sleep -Seconds 8
$ok = Test-NetConnection -ComputerName localhost -Port 7338 -InformationLevel Quiet
if (-not $ok) { throw "MATHIR daemon did not bind port 7338" }
Write-Output "[OK] MATHIR daemon running on port 7338"
```

---

**Next:** see [INSTALL_LINUX.md](INSTALL_LINUX.md) or [INSTALL_MACOS.md](INSTALL_MACOS.md) for other platforms. The OpenCode MCP config (step 7) is identical across all three.

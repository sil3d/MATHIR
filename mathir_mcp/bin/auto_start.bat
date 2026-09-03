@echo off
REM ============================================================================
REM MATHIR Daemon + Universal Proxy Auto-Start (Windows) v2
REM ----------------------------------------------------------------------------
REM Idempotent launcher: checks ports 7338/7339 first and starts whatever is
REM missing. Safe to run at every logon, from Task Scheduler, from the Startup
REM folder, and from the 5-minute healthcheck -- concurrent invocations cannot
REM spawn duplicate daemons/proxies.
REM
REM v2 fixes (2026-08-18):
REM   - Port checks via `netstat` -> idempotent, no duplicate processes.
REM   - NO `>> log` redirects on the `start` lines anymore: when the daemon is
REM     already holding mathir_daemon.log open, cmd's redirection fails with
REM     "file in use by another process" and cmd SKIPS the command entirely.
REM     This is why the universal proxy silently never started since
REM     2026-08-12 despite the healthcheck retrying every 5 minutes. The
REM     daemon and the proxy both write their own logs via logging handlers.
REM   - HF offline probe kept (fast yes/no, cannot hang); hidden powershell.
REM
REM Usage:   auto_start.bat
REM Exit:    0 = everything required is up (or was launched)
REM          2 = python not found / daemon script missing
REM ============================================================================

setlocal

REM ---- Configuration ---------------------------------------------------------
REM Resolved from %USERPROFILE% so the script is portable across usernames
REM and matches the actual ~/.config/MATHIR install location.
set "BIN_DIR=%USERPROFILE%\.config\MATHIR\mathir_mcp\mathir_lib"
set "DAEMON_PATH=%BIN_DIR%\mathir_server.py"
set "LOG_PATH=%USERPROFILE%\.config\MATHIR\logs\mathir_daemon.log"
set "PORT=7338"
REM Universal proxy (see mathir_proxy.py) -- same lifecycle as the daemon so
REM it never sits dead while the daemon is healthy. Starting it is harmless
REM even if no tool is pointed at it yet: it just listens on 7339 idle.
set "PROXY_PATH=%BIN_DIR%\mathir_proxy.py"
set "PROXY_LOG_PATH=%USERPROFILE%\.config\MATHIR\logs\mathir_proxy.log"
set "PROXY_PORT=7339"

REM ---- Resolve Python dynamically (cmd-only, no hardcoded install path) -----
REM Priority: PATH (`where python`) > py launcher > common install locations
REM (miniconda/anaconda, WindowsApps, Programs\PythonXXX). This avoids the
REM historical bug where a hardcoded Python311 path silently broke auto-start
REM on machines using a different Python (e.g. Miniconda).
set "PYTHON_PATH="

for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYTHON_PATH set "PYTHON_PATH=%%P"
)

if not defined PYTHON_PATH (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_PATH=py"
)

if not defined PYTHON_PATH (
    for %%D in (
        "%USERPROFILE%\miniconda3\python.exe"
        "%USERPROFILE%\anaconda3\python.exe"
        "%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\python.exe"
        "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
        "%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"
        "%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe"
    ) do (
        if not defined PYTHON_PATH if exist %%D set "PYTHON_PATH=%%~D"
    )
)

REM ---- Sanity checks (cmd-only -- no PowerShell -Command embedded) -----------
if not defined PYTHON_PATH (
    echo [FATAL] No Python interpreter found ^(checked PATH, py launcher, common install dirs^)
    echo         Set PYTHON_PATH manually as an environment variable to override.
    endlocal & exit /b 2
)
if not exist "%DAEMON_PATH%" (
    echo [FATAL] Daemon script not found at: "%DAEMON_PATH%"
    endlocal & exit /b 2
)

REM ---- Network reachability probe --------------------------------------------
REM Decide whether huggingface.co is reachable BEFORE spawning the daemon.
REM - Online  -> HF_HUB_OFFLINE=0, daemon will (re)check / download model normally
REM - Offline -> HF_HUB_OFFLINE=1, daemon reads multilingual-e5-small from the local
REM             cache at %USERPROFILE%\.cache\huggingface\hub\. No panic, no
REM             5-retry timeout loop in the warmup thread, no hung startup.
REM The probe uses a 3-second TCP connect to huggingface.co:443 (the bare
REM minimum to know DNS + TLS are reachable). This is a fast yes/no -- no HTTP
REM request, no third-party dep, and it cannot hang the launcher.
set "HF_HUB_OFFLINE=0"
powershell -NoProfile -WindowStyle Hidden -Command "$ErrorActionPreference='SilentlyContinue'; try { $c = New-Object System.Net.Sockets.TcpClient; $iar = $c.BeginConnect('huggingface.co', 443, $null, $null); $ok = $iar.AsyncWaitHandle.WaitOne(3000); if ($ok) { $c.EndConnect($iar); $c.Close(); 'ONLINE' } else { 'OFFLINE' } } catch { 'OFFLINE' }" > "%TEMP%\mathir_probe.txt" 2>nul
set /p PROBE_RESULT=<"%TEMP%\mathir_probe.txt"
del "%TEMP%\mathir_probe.txt" 2>nul
if /I "%PROBE_RESULT%"=="OFFLINE" (
    set "HF_HUB_OFFLINE=1"
    echo [WARN] huggingface.co unreachable -- daemon will run OFFLINE ^(local embedder cache^).
)

REM ---- Port-state check (idempotence) ----------------------------------------
REM netstat + findstr is pure cmd -- no powershell needed. ERRORLEVEL 0 == the
REM port is already listening. This makes concurrent invocations no-ops.
netstat -ano | findstr "LISTENING" | findstr ":%PORT% " >nul 2>nul
set "DAEMON_UP=%ERRORLEVEL%"
netstat -ano | findstr "LISTENING" | findstr ":%PROXY_PORT% " >nul 2>nul
set "PROXY_UP=%ERRORLEVEL%"

REM ---- Launch daemon if down -------------------------------------------------
if "%DAEMON_UP%"=="0" (
    echo Daemon already listening on %PORT% -- nothing to do.
) else (
    start "MATHIR_DAEMON" /B "%PYTHON_PATH%" "%DAEMON_PATH%"
    echo Daemon launch requested on port %PORT% ^(see mathir_daemon.log^).
)

REM ---- Launch universal proxy if down (best-effort, non-fatal) ---------------
REM Missing script or missing flask/waitress must never block anything -- this
REM section only ever adds capability.
if "%PROXY_UP%"=="0" (
    echo Proxy already listening on %PROXY_PORT% -- nothing to do.
) else (
    if exist "%PROXY_PATH%" (
        start "MATHIR_PROXY" /B "%PYTHON_PATH%" "%PROXY_PATH%" --port %PROXY_PORT% --target https://api.openai.com
        echo Proxy launch requested on port %PROXY_PORT% ^(target api.openai.com, see mathir_proxy.log^).
    )
)

endlocal & exit /b 0

@echo off
REM ============================================================================
REM MATHIR Daemon Auto-Start (Windows)
REM ----------------------------------------------------------------------------
REM Minimal, robust launcher for the MATHIR cognitive memory daemon.
REM Compatible with: cmd.exe, PowerShell `& auto_start.bat`, Task Scheduler,
REM                  Startup folder (.vbs wrapper), and Git Bash.
REM
REM Design rule: NO embedded `powershell -Command` blocks. The .bat body must
REM consist only of cmd.exe primitives so PowerShell's `&` call operator
REM cannot misinterpret `>` / `>>` / `2>&1` as outer-level redirection.
REM
REM Usage:   auto_start.bat
REM Exit:    0 = daemon process was launched (or was already running)
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
REM Universal proxy (see mathir_proxy.py) — same lifecycle as the daemon so
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

REM ---- Sanity checks (cmd-only — no PowerShell -Command embedded) -----------
if not defined PYTHON_PATH (
    echo [FATAL] No Python interpreter found ^(checked PATH, py launcher, common install dirs^)
    echo         Set PYTHON_PATH manually as an environment variable to override.
    endlocal & exit /b 2
)
if not exist "%DAEMON_PATH%" (
    echo [FATAL] Daemon script not found at: "%DAEMON_PATH%"
    endlocal & exit /b 2
)

REM ---- Log banner (append, never clobber) -----------------------------------
echo. >> "%LOG_PATH%"
echo ================================================================================ >> "%LOG_PATH%"
echo [%date% %time%] auto_start.bat invoked (PID-launch only) >> "%LOG_PATH%"

REM ---- Launch daemon detached ------------------------------------------------
REM `start "" /B` starts without a new console window and detaches from this
REM process so the .bat can exit immediately. stdout+stderr are appended to the
REM log so cold-start failures are visible.
start "MATHIR_DAEMON" /B "%PYTHON_PATH%" "%DAEMON_PATH%" >> "%LOG_PATH%" 2>&1

echo [%date% %time%] Daemon launched (see log for startup progress) >> "%LOG_PATH%"
echo [%date% %time%] Use auto_start_helpers.ps1 to verify port %PORT% is open.
echo Daemon launch requested. PID will appear in mathir_daemon.log.
echo Log: "%LOG_PATH%"

REM ---- Launch universal proxy detached (best-effort, non-fatal) -------------
REM Missing script or missing flask/waitress must never block the daemon
REM launch above — this section only ever adds capability, never breaks it.
if exist "%PROXY_PATH%" (
    start "MATHIR_PROXY" /B "%PYTHON_PATH%" "%PROXY_PATH%" --port %PROXY_PORT% --target https://api.openai.com >> "%PROXY_LOG_PATH%" 2>&1
    echo [%date% %time%] Proxy launch requested (port %PROXY_PORT%, target=api.openai.com, see mathir_proxy.log) >> "%LOG_PATH%"
)

endlocal & exit /b 0

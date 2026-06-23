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
REM and matches the actual ~/.config/opencode/bin install location.
set "BIN_DIR=%USERPROFILE%\.config\opencode\bin"
set "PYTHON_PATH=%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"
set "DAEMON_PATH=%BIN_DIR%\mathir_daemon.py"
set "LOG_PATH=%BIN_DIR%\mathir_daemon.log"
set "PORT=7338"

REM ---- Sanity checks (cmd-only — no PowerShell -Command embedded) -----------
if not exist "%PYTHON_PATH%" (
    echo [FATAL] Python not found at: "%PYTHON_PATH%"
    echo         Update PYTHON_PATH in auto_start.bat
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

endlocal & exit /b 0

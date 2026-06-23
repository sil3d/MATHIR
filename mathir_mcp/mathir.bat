@echo off
title MATHIR Playground Dashboard
echo ==========================================
echo   MATHIR Playground - Stats Dashboard
echo ==========================================
echo.

REM Resolve the script's own directory (no hardcoded opencode path)
set "SCRIPT_DIR=%~dp0"

REM The stats server lives in mathir_lib/ subdirectory (portable)
set "STATS_SERVER=%SCRIPT_DIR%mathir_lib\mathir_stats_server.py"

if not exist "%STATS_SERVER%" (
    echo ERROR: stats server not found at:
    echo   %STATS_SERVER%
    echo.
    echo This script must be in the mathir_mcp/ root directory.
    pause
    exit /b 1
)

echo Starting MATHIR Stats Dashboard...
echo   Script dir:  %SCRIPT_DIR%
echo   Server file: %STATS_SERVER%
echo.

REM Launch the stats server in a new window (so Ctrl+C in this window doesn't kill it)
start "MATHIR Dashboard Server" /min python "%STATS_SERVER%"

REM Wait 3s for the server to bind port 7420
timeout /t 3 /nobreak >nul

REM Open the dashboard in the default browser
start http://127.0.0.1:7420

echo.
echo Dashboard running at http://127.0.0.1:7420
echo (The server runs in a minimized window. Close it from Task Manager when done.)
echo.
pause

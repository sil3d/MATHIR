@echo off
title MATHIR Smart Installer
cd /d "%~dp0"

echo.
echo ================================================================
echo.
echo   MATHIR Smart Installer - Install + Auto-Detect + Inject
echo   5-tier cognitive memory for 40+ coding agents
echo.
echo ================================================================
echo.

REM STEP 1: Install Python dependencies
echo [1/3] Installing Python dependencies...

REM Find requirements.txt (could be in mathir_mcp/ or repo root)
set "REQ_FILE=%~dp0..\requirements.txt"
if not exist "%REQ_FILE%" set "REQ_FILE=%~dp0..\..\requirements.txt"

if exist "%REQ_FILE%" (
    echo   Using: %REQ_FILE%
    python -m pip install --upgrade pip --quiet
    python -m pip install -r "%REQ_FILE%"
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install dependencies.
        echo Try manually: python -m pip install -r %REQ_FILE%
        pause
        exit /b 1
    )
    echo   Dependencies installed.
) else (
    echo   [WARNING] requirements.txt not found.
    echo   Install deps manually:
    echo     python -m pip install torch sentence-transformers usearch sqlite-vec numpy PyYAML
    pause
)
echo.

REM STEP 2: Auto-detect coding agents
echo [2/3] Detecting coding agents...
echo.
python install_smart.py
if errorlevel 1 (
    echo.
    echo [ERROR] Python not found or smart installer failed.
    echo Install Python from https://python.org
    pause
    exit /b 1
)
echo.

REM STEP 3: Auto-start setup
echo [3/3] Setting up auto-start...
python install_smart.py --autostart-only
echo.

echo ================================================================
echo   MATHIR installed!
echo   - Daemon on port 7338
echo   - Stats server on port 7420 (auto-started)
echo   - Auto-start configured (Task Scheduler / Startup folder)
echo   - Dashboard: http://localhost:7420
echo ================================================================
echo.
pause >nul

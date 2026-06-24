@echo off
title MATHIR Smart Installer
cd /d "%~dp0"

echo.
echo ================================================================
echo.
echo   MATHIR Smart Installer - Auto-Detect ^& Inject
echo   5-tier cognitive memory for 40+ coding agents
echo.
echo ================================================================
echo.
echo   IMPORTANT: This script configures your coding agent to USE MATHIR.
echo   It does NOT install MATHIR itself - it injects config files only.
echo.
echo   RECOMMENDED: Use an AI coding agent like OpenCode to install MATHIR.
echo   The agent will read the docs, detect your setup, and configure
echo   everything automatically. Much easier than doing it by hand!
echo.
echo   Download OpenCode: https://opencode.ai/
echo.
echo   If you prefer manual install, this script will detect your agents
echo   and inject the MATHIR config for each one.
echo.
echo ================================================================
echo.

python install_smart.py
echo.
echo Press any key to exit...
pause >nul

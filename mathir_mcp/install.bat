@echo off
title MATHIR Smart Installer
cd /d "%~dp0"
python install_smart.py
echo.
echo Press any key to exit...
pause >nul

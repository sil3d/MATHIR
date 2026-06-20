@echo off
title MATHIR Stats Dashboard
echo Starting MATHIR Stats Dashboard...
start /b python "%~dp0mathir_stats_server.py"
timeout /t 2 /nobreak >nul
start http://127.0.0.1:7420
echo Dashboard running at http://127.0.0.1:7420
echo Press Ctrl+C to stop.
pause >nul

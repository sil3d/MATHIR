@echo off
title MATHIR - Auto Launcher
cls
echo ================================================================
echo   MATHIR - FULL AUTOMATIC LAUNCH
echo ================================================================
echo.
echo This script will automatically open:
echo   1. Training Terminal (train_evolution.py)
echo   2. Visualization Dashboard (Streamlit)
echo.
echo Ensure that the "mathir_cuda" environment exists!
echo.
pause
echo.

echo [1/2] Opening Training Terminal...
start "MATHIR - Training" cmd /k "conda activate mathir_cuda && cd /d %~dp0 && python train_evolution.py"

echo [2/2] Waiting 5 seconds before dashboard...
timeout /t 5 /nobreak > nul

echo [2/2] Opening Live Brain Dashboard...
start "MATHIR - Live Brain" cmd /k "conda activate mathir_cuda && cd /d %~dp0 && streamlit run dashboard_live.py"

echo.
echo ================================================================
echo   SUCCESS! Two windows have been opened:
echo ================================================================
echo.
echo   [Window 1] MATHIR - Training
echo     -> Training in progress (train_evolution.py)
echo     -> Logs every 100 steps
echo     -> Checkpoints every 5000 steps
echo.
echo   [Window 2] MATHIR - Dashboard  
echo     -> Web interface in your browser
echo     -> Activate "Live Mode" in the sidebar
echo     -> "Brain Scan" tab to see weights evolving
echo.
echo ================================================================
echo   TIPS:
echo ================================================================
echo.
echo   - DO NOT CLOSE this window (to keep track)
echo   - To stop training: Ctrl+C in the Training window
echo   - The dashboard refreshes auto in Live Mode
echo   - Checkpoints saved in: checkpoints/
echo.
echo ================================================================
pause

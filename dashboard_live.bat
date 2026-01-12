@echo off
title MATHIR - Live Brain Dashboard
echo ====================================
echo   MATHIR - LIVE BRAIN DASHBOARD
echo ====================================
echo.
echo Ensure that the environment is active:
echo   conda activate mathir_cuda
echo.
echo This dashboard shows:
echo   - Real-time learning curves
echo   - Animated neural network (changing weights)
echo   - Instant metrics
echo.
streamlit run dashboard_live.py
pause

@echo off
title MATHIR - Dashboard
echo ====================================
echo   MATHIR - SCIENTIFIC DASHBOARD
echo ====================================
echo.
echo Ensure that the environment is active:
echo   conda activate mathir_cuda
echo.
echo Opening dashboard in browser...
echo (Activate Live Mode to see live training)
echo.
streamlit run final_report_streamlit.py
pause

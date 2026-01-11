@echo off
title MATHIR - Dashboard
echo ====================================
echo   MATHIR - DASHBOARD SCIENTIFIQUE
echo ====================================
echo.
echo Assurez-vous que l'environnement est actif:
echo   conda activate mathir_cuda
echo.
echo Ouverture du dashboard dans le navigateur...
echo (Activez le Mode Live pour voir l'entrainement en direct)
echo.
streamlit run final_report_streamlit.py
pause

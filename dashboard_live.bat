@echo off
title MATHIR - Live Brain Dashboard
echo ====================================
echo   MATHIR - LIVE BRAIN DASHBOARD
echo ====================================
echo.
echo Assurez-vous que l'environnement est actif:
echo   conda activate mathir_cuda
echo.
echo Ce dashboard montre:
echo   - Courbes d'apprentissage en temps reel
echo   - Reseau de neurones anime (poids qui changent)
echo   - Metriques instantanees
echo.
streamlit run dashboard_live.py
pause

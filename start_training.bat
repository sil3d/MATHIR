@echo off
title MATHIR - Lanceur Automatique
cls
echo ================================================================
echo   MATHIR - LANCEMENT AUTOMATIQUE COMPLET
echo ================================================================
echo.
echo Ce script va ouvrir automatiquement:
echo   1. Terminal d'entrainement (train_evolution.py)
echo   2. Dashboard de visualisation (Streamlit)
echo.
echo Assurez-vous que l'environnement "mathir_cuda" existe !
echo.
pause
echo.

echo [1/2] Ouverture du terminal d'entrainement...
start "MATHIR - Training" cmd /k "conda activate mathir_cuda && cd /d %~dp0 && python train_evolution.py"

echo [2/2] Attente de 5 secondes avant le dashboard...
timeout /t 5 /nobreak > nul

echo [2/2] Ouverture du dashboard Live Brain...
start "MATHIR - Live Brain" cmd /k "conda activate mathir_cuda && cd /d %~dp0 && streamlit run dashboard_live.py"

echo.
echo ================================================================
echo   SUCCES ! Deux fenetres ont ete ouvertes:
echo ================================================================
echo.
echo   [Fenetre 1] MATHIR - Training
echo     -> Entrainement en cours (train_evolution.py)
echo     -> Logs toutes les 100 steps
echo     -> Checkpoints tous les 5000 steps
echo.
echo   [Fenetre 2] MATHIR - Dashboard  
echo     -> Interface web dans votre navigateur
echo     -> Activez "Mode Live" dans la sidebar
echo     -> Onglet "Brain Scan" pour voir les poids evoluer
echo.
echo ================================================================
echo   TIPS:
echo ================================================================
echo.
echo   - NE FERMEZ PAS cette fenetre (pour garder la trace)
echo   - Pour arreter l'entrainement: Ctrl+C dans la fenetre Training
echo   - Le dashboard se rafraichit auto en Mode Live
echo   - Checkpoints sauvegardes dans: checkpoints/
echo.
echo ================================================================
pause

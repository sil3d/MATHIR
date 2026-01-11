@echo off
title MATHIR - Training
echo ====================================
echo   MATHIR V3.3 - TRAINING
echo ====================================
echo.
echo Assurez-vous que l'environnement est actif:
echo   conda activate mathir_cuda
echo.
python train_evolution.py
pause

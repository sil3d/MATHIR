@echo off
title MATHIR - Optimizer
echo ====================================
echo   MATHIR - HYPERPARAMETER OPTIMIZER
echo ====================================
echo.
echo Assurez-vous que l'environnement est actif:
echo   conda activate mathir_cuda
echo.
python optimize_mathir.py
pause

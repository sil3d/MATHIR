@echo off
title MATHIR - Optimizer
echo ====================================
echo   MATHIR - HYPERPARAMETER OPTIMIZER
echo ====================================
echo.
echo Ensure that the environment is active:
echo   conda activate mathir_cuda
echo.
python optimize_mathir.py
pause

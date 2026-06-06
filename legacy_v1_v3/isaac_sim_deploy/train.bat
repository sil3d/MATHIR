@echo off
title MATHIR - Training
echo ====================================
echo   MATHIR V3.3 - TRAINING
echo ====================================
echo.
echo Ensure that the environment is active:
echo   conda activate mathir_cuda
echo.
python train_evolution.py
pause

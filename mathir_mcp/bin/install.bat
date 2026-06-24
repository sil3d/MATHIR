@echo off
setlocal enabledelayedexpansion
title MATHIR Smart Installer
cd /d "%~dp0"

echo.
echo ================================================================
echo.
echo   MATHIR Smart Installer - Install + Auto-Detect + Inject
echo   5-tier cognitive memory for 40+ coding agents
echo.
echo ================================================================
echo.

REM STEP 0: Detect Python 3.10+
echo [0/3] Detecting Python 3.10+...
set "PYTHON="
for %%P in (python3.11 python3.10 py -3.11 py -3.10 python python3) do (
    %%P --version >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2" %%V in ('%%P --version 2^^>^&1') do (
            echo Found: %%P -- %%V
            set "PYTHON=%%P"
            goto :python_found
        )
    )
)
:python_found
if "%PYTHON%"=="" (
    echo [ERROR] Python 3.10+ not found in PATH.
    echo Install Python from https://python.org (tick 'Add to PATH' and 'Python 3.10+')
    pause
    exit /b 1
)
echo [OK] Using: %PYTHON%
echo.

REM STEP 0b: Detect NVIDIA GPU (for CUDA)
echo [0/3] Detecting GPU...
set "HAS_CUDA=0"
where nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    nvidia-smi >nul 2>&1
    if !errorlevel! equ 0 (
        set "HAS_CUDA=1"
        echo [INFO] NVIDIA GPU detected - will install CUDA PyTorch
    ) else (
        echo [INFO] nvidia-smi present but no GPU - will use CPU PyTorch
    )
) else (
    echo [INFO] No NVIDIA GPU detected - will use CPU PyTorch
)
echo.

REM STEP 1: Install Python dependencies
echo [1/3] Installing Python dependencies...

REM Find requirements.txt (could be in mathir_mcp/ or repo root)
set "REQ_FILE=%~dp0..\requirements.txt"
if not exist "%REQ_FILE%" set "REQ_FILE=%~dp0..\..\requirements.txt"

if not exist "%REQ_FILE%" (
    echo [ERROR] requirements.txt not found at %REQ_FILE%
    pause
    exit /b 1
)
echo   Using: %REQ_FILE%

%PYTHON% -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)

REM Install PyTorch with appropriate backend
echo   Installing PyTorch...
if "%HAS_CUDA%"=="1" (
    echo     Using CUDA 12.4 ^(NVIDIA GPU^)
    %PYTHON% -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
) else (
    echo     Using CPU-only ^(no NVIDIA GPU detected^)
    %PYTHON% -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
)
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch.
    pause
    exit /b 1
)

REM Install remaining deps from requirements.txt
echo   Installing remaining deps...
%PYTHON% -m pip install -r "%REQ_FILE%" --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    echo Try manually: %PYTHON% -m pip install -r %REQ_FILE%
    pause
    exit /b 1
)

echo   Dependencies installed.
echo.

REM STEP 2: Auto-detect coding agents
echo [2/3] Detecting coding agents...
echo.
%PYTHON% install_smart.py
if errorlevel 1 (
    echo [ERROR] Smart installer failed.
    pause
    exit /b 1
)
echo.

REM STEP 3: Auto-start setup
echo [3/3] Setting up auto-start...
%PYTHON% install_smart.py --autostart-only

echo.
echo ================================================================
echo   MATHIR installed!
echo   - Python: %PYTHON%
if "%HAS_CUDA%"=="1" (
    echo   - PyTorch: CUDA ^(GPU^)
) else (
    echo   - PyTorch: CPU
)
echo   - Daemon on port 7338
echo   - Stats server on port 7420 (auto-started)
echo   - Auto-start configured (Task Scheduler / Startup folder)
echo   - Dashboard: http://localhost:7420
echo ================================================================
echo.
pause >nul

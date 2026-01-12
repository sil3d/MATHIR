@echo off
REM Improved script to automatically detect conda
REM MATHIR Benchmark - CUDA Configuration

echo ================================================
echo   MATHIR + CUDA Environment Configuration
echo ================================================
echo.
echo ⚠️  IMPORTANT: This script should be run ONLY ONCE
echo     For daily runs, use:
echo       1. conda activate mathir_cuda
echo       2. train.bat / dashboard.bat / etc.
echo.
pause
echo.

REM Environment Name
set ENV_NAME=mathir_cuda

REM === INTELLIGENT CONDA DETECTION ===
echo [1/6] Detecting Conda...

REM First check if conda is in PATH
where conda >nul 2>nul
if not errorlevel 1 (
    echo ✓ Conda found in PATH
    set CONDA_EXE=conda
    goto conda_found
)

REM Search in standard locations
echo Conda not in PATH, searching in standard locations...

REM 1. Anaconda3 in User
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
    echo ✓ Found: Anaconda3 ^(User^)
    set CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe
    set CONDA_ROOT=%USERPROFILE%\anaconda3
    goto conda_found
)

REM 2. Miniconda3 in User
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
    echo ✓ Found: Miniconda3 ^(User^)
    set CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe
    set CONDA_ROOT=%USERPROFILE%\miniconda3
    goto conda_found
)

REM 3. Anaconda3 in ProgramData
if exist "%ProgramData%\anaconda3\Scripts\conda.exe" (
    echo ✓ Found: Anaconda3 ^(System^)
    set CONDA_EXE=%ProgramData%\anaconda3\Scripts\conda.exe
    set CONDA_ROOT=%ProgramData%\anaconda3
    goto conda_found
)

REM 4. Miniconda3 in ProgramData
if exist "%ProgramData%\miniconda3\Scripts\conda.exe" (
    echo ✓ Found: Miniconda3 ^(System^)
    set CONDA_EXE=%ProgramData%\miniconda3\Scripts\conda.exe
    set CONDA_ROOT=%ProgramData%\miniconda3
    goto conda_found
)

REM 5. In the project's .conda folder
if exist ".conda\Scripts\conda.exe" (
    echo ✓ Found: Local Conda ^(project^)
    set CONDA_EXE=%CD%\.conda\Scripts\conda.exe
    set CONDA_ROOT=%CD%\.conda
    goto conda_found
)

REM Conda not found
echo.
echo ❌ ERROR: Conda not found!
echo.
echo Checked locations:
echo - System PATH
echo - %USERPROFILE%\anaconda3
echo - %USERPROFILE%\miniconda3
echo - %ProgramData%\anaconda3
echo - %ProgramData%\miniconda3
echo - %CD%\.conda
echo.
echo Solutions:
echo.
echo Option 1: Add conda to PATH
echo   1. Search "Anaconda Prompt" in Start Menu
echo   2. Run this script from Anaconda Prompt
echo.
echo Option 2: Install Miniconda
echo   https://docs.conda.io/en/latest/miniconda.html
echo.
echo Option 3: Use pip directly ^(see CUDA_SETUP.md^)
echo.
pause
exit /b 1

:conda_found
echo.
echo Conda executable: %CONDA_EXE%
if defined CONDA_ROOT echo Conda root: %CONDA_ROOT%
echo.

REM Initialize conda for this session if necessary
if defined CONDA_ROOT (
    echo Initializing conda for this session...
    call "%CONDA_ROOT%\Scripts\activate.bat"
)

REM === GPU CHECK ===
echo [2/6] Checking NVIDIA GPU...
nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo ⚠️  nvidia-smi not available
    echo    Check that NVIDIA drivers are installed
    echo.
) else (
    echo ✓ NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo.
)

REM === ENVIRONMENT CHECK/CREATION ===
echo [3/6] Checking environment %ENV_NAME%...

REM Check if environment exists
"%CONDA_EXE%" env list | findstr /C:"%ENV_NAME%" >nul
if not errorlevel 1 (
    echo.
    echo Environment %ENV_NAME% already exists.
    echo.
    choice /C YN /M "Recreate environment (Y) or Use existing (N)"
    if errorlevel 2 (
        echo.
        echo Using existing environment
        goto activate_env
    )
    
    echo.
    echo Removing old environment...
    call "%CONDA_EXE%" env remove -n %ENV_NAME% -y
)

echo.
echo Creating environment %ENV_NAME% with Python 3.10...
call "%CONDA_EXE%" create -n %ENV_NAME% python=3.10 -y
if errorlevel 1 (
    echo.
    echo ❌ ERROR: Environment creation failed
    pause
    exit /b 1
)

:activate_env
echo.
echo [4/6] Activating environment %ENV_NAME%...
call "%CONDA_EXE%" activate %ENV_NAME%

REM === PYTORCH CUDA INSTALLATION ===
echo.
echo [5/6] Installing PyTorch with CUDA 12.1...
echo.
echo ⏳ Downloading in progress ^(may take 5-10 minutes^)...
echo.

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo.
    echo ❌ ERROR: PyTorch CUDA installation failed
    echo.
    echo Try manually:
    echo   conda activate %ENV_NAME%
    echo   pip install torch --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

REM === DEPENDENCIES INSTALLATION ===
echo.
echo [6/6] Installing MATHIR dependencies...
pip install streamlit plotly pandas numpy

echo.
echo ================================================
echo   ✅ CONFIGURATION COMPLETE!
echo ================================================
echo.

REM === VERIFICATION ===
echo PyTorch CUDA Verification:
echo.
python -c "import torch; print('✓ PyTorch:', torch.__version__); cuda_ok = torch.cuda.is_available(); print('✓ CUDA available:', cuda_ok); print('✓ GPU:', torch.cuda.get_device_name(0) if cuda_ok else 'N/A'); print('✓ VRAM:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if cuda_ok else 'N/A')"

echo.
echo ================================================
echo   USAGE
echo ================================================
echo.
echo In a NEW terminal:
echo.
echo 1. Activate environment:
echo    conda activate %ENV_NAME%
echo.
echo 2. Launch MATHIR:
echo    streamlit run app_streamlit.py
echo    OR
echo    python benchmark.py
echo.
echo ================================================
echo   VSCODE
echo ================================================
echo.
echo 1. Ctrl+Shift+P
echo 2. "Python: Select Interpreter"
echo 3. Select "%ENV_NAME%"
echo.
echo OR click on Python in bottom right
echo and select "%ENV_NAME%"
echo.
echo ================================================
echo.
echo 💡 TIP: Close and reopen VSCode to
echo     automatically detect the new
echo     environment!
echo.
echo ================================================

pause

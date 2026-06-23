@echo off
REM Alternative without conda - Direct installation with pip
REM MATHIR Benchmark - CUDA Configuration

echo ================================================
echo   MATHIR + PyTorch CUDA - pip Installation
echo ================================================
echo.
echo This method DOES NOT USE conda
echo Installing in current Python environment
echo.

REM Python Verification
echo [1/4] Checking Python...
python --version 2>nul
if errorlevel 1 (
    echo ❌ ERROR: Python not found
    echo.
    echo Install Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✓ Python detected
python --version
echo.

REM pip Verification
echo [2/4] Checking pip...
pip --version 2>nul
if errorlevel 1 (
    echo ❌ ERROR: pip not found
    pause
    exit /b 1
)

echo ✓ pip detected
echo.

REM GPU Verification
echo [3/4] Checking NVIDIA GPU...
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

REM Installation
echo [4/4] Installing packages...
echo.
echo ⏳ Installing PyTorch with CUDA 12.1...
echo    ^(may take 5-10 minutes^)
echo.

REM Uninstall torch CPU if present
pip uninstall -y torch torchvision torchaudio 2>nul

REM Install torch CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo.
    echo ❌ Failed to install PyTorch CUDA 12.1
    echo.
    echo Try CUDA 11.8:
    echo   pip install torch --index-url https://download.pytorch.org/whl/cu118
    echo.
    pause
    exit /b 1
)

echo.
echo ⏳ Installing MATHIR dependencies...
pip install streamlit plotly pandas numpy

echo.
echo ================================================
echo   ✅ INSTALLATION COMPLETE!
echo ================================================
echo.

REM Verification
echo Installation Verification:
echo.
python -c "import torch; print('✓ PyTorch:', torch.__version__); cuda_ok = torch.cuda.is_available(); print('✓ CUDA:', 'YES' if cuda_ok else 'NO'); print('✓ GPU:', torch.cuda.get_device_name(0) if cuda_ok else 'N/A')"

echo.
echo ================================================
echo   USAGE
echo ================================================
echo.
echo Run now:
echo   streamlit run app_streamlit.py
echo   OR
echo   python benchmark.py
echo.
echo ================================================
echo   VSCODE
echo ================================================
echo.
echo If using VSCode, verify that the
echo Python interpreter used is the one
echo that has PyTorch CUDA installed.
echo.
echo To verify:
echo   python -c "import torch; print(torch.cuda.is_available())"
echo.
echo Should display: True
echo.
echo ================================================

pause

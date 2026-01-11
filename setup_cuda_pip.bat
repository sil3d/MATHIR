@echo off
REM Alternative sans conda - Installation directe avec pip
REM MATHIR Benchmark - Configuration CUDA

echo ================================================
echo   MATHIR + PyTorch CUDA - Installation pip
echo ================================================
echo.
echo Cette methode N'UTILISE PAS conda
echo Installation dans l'environnement Python actuel
echo.

REM Verification Python
echo [1/4] Verification Python...
python --version 2>nul
if errorlevel 1 (
    echo ❌ ERREUR: Python non trouve
    echo.
    echo Installez Python depuis: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✓ Python detecte
python --version
echo.

REM Verification pip
echo [2/4] Verification pip...
pip --version 2>nul
if errorlevel 1 (
    echo ❌ ERREUR: pip non trouve
    pause
    exit /b 1
)

echo ✓ pip detecte
echo.

REM Verification GPU
echo [3/4] Verification GPU NVIDIA...
nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo ⚠️  nvidia-smi non disponible
    echo    Verifiez que les drivers NVIDIA sont installes
    echo.
) else (
    echo ✓ GPU NVIDIA detecte:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo.
)

REM Installation
echo [4/4] Installation packages...
echo.
echo ⏳ Installation PyTorch avec CUDA 12.1...
echo    ^(peut prendre 5-10 minutes^)
echo.

REM Desinstalle torch CPU si present
pip uninstall -y torch torchvision torchaudio 2>nul

REM Installe torch CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo.
    echo ❌ Echec installation PyTorch CUDA 12.1
    echo.
    echo Essayez CUDA 11.8:
    echo   pip install torch --index-url https://download.pytorch.org/whl/cu118
    echo.
    pause
    exit /b 1
)

echo.
echo ⏳ Installation dependances MATHIR...
pip install streamlit plotly pandas numpy

echo.
echo ================================================
echo   ✅ INSTALLATION TERMINEE!
echo ================================================
echo.

REM Verification
echo Verification installation:
echo.
python -c "import torch; print('✓ PyTorch:', torch.__version__); cuda_ok = torch.cuda.is_available(); print('✓ CUDA:', 'OUI' if cuda_ok else 'NON'); print('✓ GPU:', torch.cuda.get_device_name(0) if cuda_ok else 'N/A')"

echo.
echo ================================================
echo   UTILISATION
echo ================================================
echo.
echo Lancez maintenant:
echo   streamlit run app_streamlit.py
echo   OU
echo   python benchmark.py
echo.
echo ================================================
echo   VSCODE
echo ================================================
echo.
echo Si vous utilisez VSCode, verifiez que
echo l'interpreteur Python utilise est bien
echo celui qui a PyTorch CUDA installe.
echo.
echo Pour verifier:
echo   python -c "import torch; print(torch.cuda.is_available())"
echo.
echo Devrait afficher: True
echo.
echo ================================================

pause

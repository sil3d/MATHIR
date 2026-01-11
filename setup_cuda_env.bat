@echo off
REM Script ameliore pour detecter conda automatiquement
REM MATHIR Benchmark - Configuration CUDA

echo ================================================
echo   Configuration Environnement MATHIR + CUDA
echo ================================================
echo.
echo ⚠️  IMPORTANT: Ce script est a lancer UNE SEULE FOIS
echo     Pour les lancements quotidiens, utilisez:
echo       1. conda activate mathir_cuda
echo       2. train.bat / dashboard.bat / etc.
echo.
pause
echo.

REM Nom de l'environnement
set ENV_NAME=mathir_cuda

REM === DETECTION INTELLIGENTE DE CONDA ===
echo [1/6] Detection de Conda...

REM Verifie d'abord si conda est dans le PATH
where conda >nul 2>nul
if not errorlevel 1 (
    echo ✓ Conda trouve dans PATH
    set CONDA_EXE=conda
    goto conda_found
)

REM Cherche dans les emplacements standards
echo Conda pas dans PATH, recherche dans emplacements standards...

REM 1. Anaconda3 dans User
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
    echo ✓ Trouve: Anaconda3 ^(User^)
    set CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe
    set CONDA_ROOT=%USERPROFILE%\anaconda3
    goto conda_found
)

REM 2. Miniconda3 dans User
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
    echo ✓ Trouve: Miniconda3 ^(User^)
    set CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe
    set CONDA_ROOT=%USERPROFILE%\miniconda3
    goto conda_found
)

REM 3. Anaconda3 dans ProgramData
if exist "%ProgramData%\anaconda3\Scripts\conda.exe" (
    echo ✓ Trouve: Anaconda3 ^(System^)
    set CONDA_EXE=%ProgramData%\anaconda3\Scripts\conda.exe
    set CONDA_ROOT=%ProgramData%\anaconda3
    goto conda_found
)

REM 4. Miniconda3 dans ProgramData
if exist "%ProgramData%\miniconda3\Scripts\conda.exe" (
    echo ✓ Trouve: Miniconda3 ^(System^)
    set CONDA_EXE=%ProgramData%\miniconda3\Scripts\conda.exe
    set CONDA_ROOT=%ProgramData%\miniconda3
    goto conda_found
)

REM 5. Dans le dossier .conda du projet
if exist ".conda\Scripts\conda.exe" (
    echo ✓ Trouve: Conda local ^(projet^)
    set CONDA_EXE=%CD%\.conda\Scripts\conda.exe
    set CONDA_ROOT=%CD%\.conda
    goto conda_found
)

REM Conda non trouve
echo.
echo ❌ ERREUR: Conda introuvable!
echo.
echo Emplacements verifies:
echo - PATH systeme
echo - %USERPROFILE%\anaconda3
echo - %USERPROFILE%\miniconda3
echo - %ProgramData%\anaconda3
echo - %ProgramData%\miniconda3
echo - %CD%\.conda
echo.
echo Solutions:
echo.
echo Option 1: Ajouter conda au PATH
echo   1. Cherchez "Anaconda Prompt" dans le menu Demarrer
echo   2. Lancez ce script depuis Anaconda Prompt
echo.
echo Option 2: Installer Miniconda
echo   https://docs.conda.io/en/latest/miniconda.html
echo.
echo Option 3: Utiliser pip directement ^(voir CUDA_SETUP.md^)
echo.
pause
exit /b 1

:conda_found
echo.
echo Conda executable: %CONDA_EXE%
if defined CONDA_ROOT echo Conda root: %CONDA_ROOT%
echo.

REM Initialise conda pour cette session si necessaire
if defined CONDA_ROOT (
    echo Initialisation de conda pour cette session...
    call "%CONDA_ROOT%\Scripts\activate.bat"
)

REM === VERIFICATION GPU ===
echo [2/6] Verification GPU NVIDIA...
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

REM === VERIFICATION/CREATION ENVIRONNEMENT ===
echo [3/6] Verification environnement %ENV_NAME%...

REM Verifie si l'environnement existe
"%CONDA_EXE%" env list | findstr /C:"%ENV_NAME%" >nul
if not errorlevel 1 (
    echo.
    echo Environnement %ENV_NAME% existe deja.
    echo.
    choice /C YN /M "Recreer l'environnement (Y) ou Utiliser existant (N)"
    if errorlevel 2 (
        echo.
        echo Utilisation de l'environnement existant
        goto activate_env
    )
    
    echo.
    echo Suppression de l'ancien environnement...
    call "%CONDA_EXE%" env remove -n %ENV_NAME% -y
)

echo.
echo Creation environnement %ENV_NAME% avec Python 3.10...
call "%CONDA_EXE%" create -n %ENV_NAME% python=3.10 -y
if errorlevel 1 (
    echo.
    echo ❌ ERREUR: Echec creation environnement
    pause
    exit /b 1
)

:activate_env
echo.
echo [4/6] Activation environnement %ENV_NAME%...
call "%CONDA_EXE%" activate %ENV_NAME%

REM === INSTALLATION PYTORCH CUDA ===
echo.
echo [5/6] Installation PyTorch avec CUDA 12.1...
echo.
echo ⏳ Telechargement en cours ^(peut prendre 5-10 minutes^)...
echo.

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo.
    echo ❌ ERREUR: Echec installation PyTorch CUDA
    echo.
    echo Essayez manuellement:
    echo   conda activate %ENV_NAME%
    echo   pip install torch --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

REM === INSTALLATION DEPENDANCES ===
echo.
echo [6/6] Installation dependances MATHIR...
pip install streamlit plotly pandas numpy

echo.
echo ================================================
echo   ✅ CONFIGURATION TERMINEE!
echo ================================================
echo.

REM === VERIFICATION ===
echo Verification PyTorch CUDA:
echo.
python -c "import torch; print('✓ PyTorch:', torch.__version__); cuda_ok = torch.cuda.is_available(); print('✓ CUDA disponible:', cuda_ok); print('✓ GPU:', torch.cuda.get_device_name(0) if cuda_ok else 'N/A'); print('✓ VRAM:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if cuda_ok else 'N/A')"

echo.
echo ================================================
echo   UTILISATION
echo ================================================
echo.
echo Dans un NOUVEAU terminal:
echo.
echo 1. Activez l'environnement:
echo    conda activate %ENV_NAME%
echo.
echo 2. Lancez MATHIR:
echo    streamlit run app_streamlit.py
echo    OU
echo    python benchmark.py
echo.
echo ================================================
echo   VSCODE
echo ================================================
echo.
echo 1. Ctrl+Shift+P
echo 2. "Python: Select Interpreter"
echo 3. Selectionnez "%ENV_NAME%"
echo.
echo OU cliquez en bas a droite sur Python
echo et selectionnez "%ENV_NAME%"
echo.
echo ================================================
echo.
echo 💡 TIP: Fermez et rouvrez VSCode pour
echo     detecter automatiquement le nouvel
echo     environnement!
echo.
echo ================================================

pause

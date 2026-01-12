@echo off
echo ===================================================
echo      MATHIR RESEARCH PAPER COMPILATION
echo ===================================================
echo.
echo [1/3] Compiling MATHIR_Paper.tex...
pdflatex -output-directory=docs docs/MATHIR_Paper.tex

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Compilation failed. Please check if 'pdflatex' is installed (MiKTeX or TeX Live).
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Second pass for references...
pdflatex -output-directory=docs docs/MATHIR_Paper.tex > nul

echo.
echo [3/3] Cleanup aux files...
del docs\*.aux docs\*.log docs\*.out

echo.
echo ===================================================
echo [SUCCESS] Paper generated: docs/MATHIR_Paper.pdf
echo ===================================================
pause

@echo off
REM MATHIR Root Installer (Windows)
REM Convenience proxy — runs the smart installer from mathir_mcp\bin\

setlocal
set "SCRIPT_DIR=%~dp0"
set "INSTALLER=%SCRIPT_DIR%mathir_mcp\bin\install.bat"

if not exist "%INSTALLER%" (
    echo ERROR: MATHIR installer not found at %INSTALLER%
    echo Make sure you cloned the full repo ^(with subdirs^).
    exit /b 1
)

echo Delegating to mathir_mcp\bin\install.bat
echo.
call "%INSTALLER%" %*
endlocal
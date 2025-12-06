@echo off

REM === CONFIGURATION ===
set ENV_NAME=bearmind

echo.
echo ============================================
echo   BEARMiND Install Step 2: Pip Packages
echo   Environment: %ENV_NAME%
echo ============================================
echo.

REM Find conda environment path
echo Looking for %ENV_NAME% environment...

set CONDA_ENV=%USERPROFILE%\.conda\envs\%ENV_NAME%
if exist "%CONDA_ENV%\python.exe" goto :found

set CONDA_ENV=%USERPROFILE%\anaconda3\envs\%ENV_NAME%
if exist "%CONDA_ENV%\python.exe" goto :found

set CONDA_ENV=%USERPROFILE%\miniconda3\envs\%ENV_NAME%
if exist "%CONDA_ENV%\python.exe" goto :found

set CONDA_ENV=C:\ProgramData\anaconda3\envs\%ENV_NAME%
if exist "%CONDA_ENV%\python.exe" goto :found

set CONDA_ENV=C:\ProgramData\miniconda3\envs\%ENV_NAME%
if exist "%CONDA_ENV%\python.exe" goto :found

echo [FAILED] Cannot find %ENV_NAME% environment
echo Run install1_conda.bat first
pause
exit /b 1

:found
echo [OK] Found: %CONDA_ENV%
set PIP=%CONDA_ENV%\Scripts\pip.exe
set PYTHON=%CONDA_ENV%\python.exe

echo.
echo Upgrading pip...
"%PIP%" install --upgrade pip

echo.
echo [CRITICAL] Locking NumPy to 1.x (required for TensorFlow/CaImAn)...
"%PIP%" install "numpy>=1.24,<2.0"

echo.
echo Installing pip packages...
"%PIP%" install -r requirements-pip.txt

echo.
echo [CRITICAL] Re-locking NumPy to 1.x (in case pip upgraded it)...
"%PIP%" install "numpy>=1.24,<2.0"

echo.
echo ============================================
echo   Verifying installation...
echo ============================================
"%PYTHON%" -u test_env.py
echo.

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo To use: conda activate %ENV_NAME%
echo.
pause

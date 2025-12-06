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
echo Checking numpy installation...
"%PYTHON%" -c "import numpy; import numpy.core.multiarray; print('[OK] NumPy', numpy.__version__, 'is working')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo.
    echo [FAILED] NumPy is broken - DLL conflict detected
    echo.
    echo This happens when pip and conda numpy conflict.
    echo The environment must be recreated:
    echo.
    echo   1. conda env remove -n %ENV_NAME% -y
    echo   2. Run install1_conda.bat again
    echo   3. Run install2_pip.bat again
    echo.
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
"%PIP%" install --upgrade pip

echo.
echo Installing pip packages (preserving conda packages)...
echo Using --upgrade-strategy only-if-needed to prevent numpy conflicts
"%PIP%" install --upgrade-strategy only-if-needed -r requirements-pip.txt

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

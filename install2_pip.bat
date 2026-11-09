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

REM The folders of the launchers (run_bearmind.bat), in their order: the env found here is the one they run.
set "CONDA_ENV="
for %%P in ("%USERPROFILE%\anaconda3\envs\%ENV_NAME%" "%USERPROFILE%\miniconda3\envs\%ENV_NAME%" "%ProgramData%\anaconda3\envs\%ENV_NAME%" "%ProgramData%\miniconda3\envs\%ENV_NAME%" "%USERPROFILE%\.conda\envs\%ENV_NAME%" "%USERPROFILE%\miniforge3\envs\%ENV_NAME%") do if not defined CONDA_ENV if exist "%%~P\python.exe" set "CONDA_ENV=%%~P"
if defined CONDA_ENV goto :found

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
echo Checking pip installation...
"%PIP%" --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] pip is broken, reinstalling...
    "%PYTHON%" -m ensurepip --upgrade
    if %ERRORLEVEL% neq 0 (
        echo [FAILED] Could not reinstall pip
        echo Try: conda install -n %ENV_NAME% pip -y
        pause
        exit /b 1
    )
)

REM pip is not upgraded: the env keeps the pip conda installed (as in the lock and the student guide)
echo.
echo Installing pip packages: pip install -r requirements-pip.txt
echo (requirements-pip.txt pins NumPy to the conda version, so pip keeps the conda NumPy)
"%PYTHON%" -m pip install -r requirements-pip.txt

echo.
echo ============================================
echo   Verifying installation...
echo ============================================
"%PYTHON%" -u test_env.py
if errorlevel 1 (
    echo [FAILED] The environment check failed: see the list above.
    pause
    exit /b 1
)
echo.

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo To use: conda activate %ENV_NAME%
echo.
pause

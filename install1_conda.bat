@echo off
setlocal enabledelayedexpansion

REM === CONFIGURATION ===
set ENV_NAME=bearmind

echo.
echo ============================================
echo   BEARMiND Install Step 1: Conda Environment
echo   Environment: %ENV_NAME%
echo ============================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [FAILED] Conda not found in PATH
    echo Please install Anaconda or Miniconda first
    pause
    exit /b 1
)

REM Check for mamba
set CONDA_CMD=conda
where mamba >nul 2>nul
if %ERRORLEVEL%==0 (
    echo [OK] Using mamba for fast installation
    set CONDA_CMD=mamba
) else (
    echo [INFO] Using conda (install mamba for faster solving)
)

echo.
echo Checking for existing environment...
call conda deactivate 2>nul

conda env list | findstr /C:"%ENV_NAME%" >nul 2>nul
if %ERRORLEVEL%==0 (
    echo [INFO] Environment '%ENV_NAME%' already exists
    set /p REMOVE="Remove and recreate? (y/n): "
    if /i "!REMOVE!"=="y" (
        echo Removing existing environment...
        conda env remove -n %ENV_NAME% -y
    ) else (
        echo Skipping environment creation.
        goto :done
    )
)

echo.
echo Creating conda environment (5-15 minutes)...
echo Command: %CONDA_CMD% env create -f environment.yml
%CONDA_CMD% env create -f environment.yml

:done
echo.
echo ============================================
echo   Step 1 Complete
echo ============================================
echo.
echo Now run: install2_pip.bat
echo To activate: conda activate %ENV_NAME%
echo.
pause

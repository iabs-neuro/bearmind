@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================
echo   BEARMiND Environment Installer
echo   Windows 10+ / Python 3.10
echo ============================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [FAILED] Conda not found in PATH
    echo Please install Anaconda or Miniconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

REM Check for mamba (10x faster solver)
set CONDA_CMD=conda
where mamba >nul 2>nul
if %ERRORLEVEL%==0 (
    echo [OK] Using mamba for fast installation
    set CONDA_CMD=mamba
) else (
    echo [INFO] mamba not found - using conda (slower but works)
    echo [TIP] For faster installs, run: conda install -n base mamba -c conda-forge
)

echo.
echo Step 1/4: Checking for existing environment...

REM Deactivate any active environment first
call conda deactivate 2>nul

conda env list | findstr /C:"bearmind" >nul 2>nul
if %ERRORLEVEL%==0 (
    echo [INFO] Environment 'bearmind' already exists
    set /p REMOVE="Remove and recreate? (y/n): "
    if /i "!REMOVE!"=="y" (
        echo Removing existing environment...
        call conda deactivate 2>nul
        conda env remove -n bearmind -y
        if %ERRORLEVEL% neq 0 (
            echo [FAILED] Could not remove environment. Try manually:
            echo   conda deactivate
            echo   conda env remove -n bearmind -y
            pause
            exit /b 1
        )
    ) else (
        echo Aborting installation.
        pause
        exit /b 0
    )
)

echo.
echo Step 2/4: Creating conda environment (this may take 5-15 minutes)...
echo Command: %CONDA_CMD% env create -f environment.yml
%CONDA_CMD% env create -f environment.yml
if %ERRORLEVEL% neq 0 (
    echo.
    echo [FAILED] Conda environment creation failed
    echo.
    echo Troubleshooting:
    echo   1. Try installing mamba: conda install -n base mamba -c conda-forge
    echo   2. Try with strict channel priority:
    echo      conda config --set channel_priority strict
    echo   3. Check internet connection
    echo.
    pause
    exit /b 1
)

echo.
echo Step 3/4: Installing pip packages...

REM Get environment path reliably using conda run
for /f "tokens=*" %%i in ('conda run -n bearmind python -c "import sys; print(sys.prefix)"') do set ENV_PATH=%%i

if not exist "%ENV_PATH%\python.exe" (
    echo [WARNING] Could not detect environment path, trying fallback...
    for /f "tokens=*" %%i in ('conda info --base') do set CONDA_BASE=%%i
    set ENV_PATH=%USERPROFILE%\.conda\envs\bearmind
    if not exist "!ENV_PATH!\python.exe" (
        set ENV_PATH=!CONDA_BASE!\envs\bearmind
    )
)

echo Using environment: %ENV_PATH%

REM Upgrade pip first
echo Upgrading pip...
"%ENV_PATH%\python.exe" -m pip install --upgrade pip

REM Install pip packages using full path to ensure correct environment
echo.
echo Installing pip packages from requirements-pip.txt...
"%ENV_PATH%\python.exe" -m pip install -r requirements-pip.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Batch install had issues. Installing packages individually...
    echo.

    echo Installing core neuroscience packages...
    "%ENV_PATH%\python.exe" -m pip install driada

    echo Installing ML packages...
    "%ENV_PATH%\python.exe" -m pip install interpret sortedcontainers llvmlite

    echo Installing image/video packages...
    "%ENV_PATH%\python.exe" -m pip install opencv-python moviepy imageio-ffmpeg proglog decorator

    echo Installing visualization packages...
    "%ENV_PATH%\python.exe" -m pip install cmasher colorspacious

    echo Installing GPU visualization stack...
    "%ENV_PATH%\python.exe" -m pip install wgpu pygfx fastplotlib jupyter-rfb glfw freetype-py pylinalg uharfbuzz

    echo Installing Jupyter widgets...
    "%ENV_PATH%\python.exe" -m pip install sidecar

    echo Installing Qt6...
    "%ENV_PATH%\python.exe" -m pip install PySide6 shiboken6
)

echo.
echo Step 4/4: Verifying installation...
echo.

REM Run comprehensive test
if exist test_env.py (
    "%ENV_PATH%\python.exe" test_env.py
) else (
    echo Running basic verification...
    "%ENV_PATH%\python.exe" -W ignore -c "import caiman; print('[OK] caiman')"
    "%ENV_PATH%\python.exe" -W ignore -c "import driada; print('[OK] driada')"
    "%ENV_PATH%\python.exe" -W ignore -c "import bokeh; print('[OK] bokeh')"
    "%ENV_PATH%\python.exe" -W ignore -c "import tensorflow; print('[OK] tensorflow')"
    "%ENV_PATH%\python.exe" -W ignore -c "import sklearn; print('[OK] scikit-learn')"
    "%ENV_PATH%\python.exe" -W ignore -c "import interpret; print('[OK] interpret')"
    "%ENV_PATH%\python.exe" -W ignore -c "import cv2; print('[OK] opencv')"
    "%ENV_PATH%\python.exe" -W ignore -c "import fastplotlib; print('[OK] fastplotlib')"
    "%ENV_PATH%\python.exe" -W ignore -c "import PySide6; print('[OK] PySide6')"
)

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo To use BEARMiND:
echo   1. Open Anaconda Prompt
echo   2. Run: conda activate bearmind
echo   3. Navigate to your project folder
echo   4. Run: jupyter lab
echo.
echo Or run 'python test_env.py' to verify all packages.
echo.
pause

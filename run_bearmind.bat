@echo off
rem Student launcher: run_bearmind.bat <command> SERIES ...   (run_bearmind.bat --help lists commands)
set "PYTHONPATH=%~dp0"
set "CONDA_BAT=%CONDA_EXE%"
if not defined CONDA_BAT set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not exist "%CONDA_BAT%" (
    echo [FAILED] conda not found. Open Anaconda Prompt and run this command there, or ask for help.
    exit /b 1
)
call "%CONDA_BAT%" run --no-capture-output -n bearmind python -m bearmind %*
exit /b %ERRORLEVEL%

@echo off
setlocal
rem Student launcher of the window: bearmind_gui.bat [SERIES]   (no console window)
set "PYTHONPATH=%~dp0"
set "PY="
for %%P in ("%USERPROFILE%\anaconda3\envs\bearmind\pythonw.exe" "%USERPROFILE%\miniconda3\envs\bearmind\pythonw.exe" "%ProgramData%\anaconda3\envs\bearmind\pythonw.exe" "%ProgramData%\miniconda3\envs\bearmind\pythonw.exe") do if not defined PY if exist "%%~P" set "PY=%%~P"
if not defined PY (
    echo [FAILED] conda env "bearmind" not found. Ask for help.
    pause
    exit /b 1
)
start "" "%PY%" -m bearmind gui %*

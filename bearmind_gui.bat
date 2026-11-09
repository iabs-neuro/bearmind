@echo off
setlocal
rem Student launcher of the window: bearmind_gui.bat [SERIES]   (no console window)
set "PYTHONPATH=%~dp0"
rem The env folders of run_bearmind.bat; pythonw opens the window without a console.
set "PY="
for %%P in ("%USERPROFILE%\anaconda3\envs\bearmind\pythonw.exe" "%USERPROFILE%\miniconda3\envs\bearmind\pythonw.exe" "%ProgramData%\anaconda3\envs\bearmind\pythonw.exe" "%ProgramData%\miniconda3\envs\bearmind\pythonw.exe" "%USERPROFILE%\.conda\envs\bearmind\pythonw.exe") do if not defined PY if exist "%%~P" set "PY=%%~P"
if not defined PY goto conda_run
start "" "%PY%" -m bearmind gui %*
exit /b 0

:conda_run
rem As in run_bearmind.bat: conda finds the env in any other folder (CONDA_CMD, not CONDA_BAT: see there).
rem conda run waits for the window, so this console stays open until the window closes.
set "CONDA_CMD=%CONDA_EXE%"
if not defined CONDA_CMD set "CONDA_CMD=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not exist "%CONDA_CMD%" (
    echo [FAILED] conda env "bearmind" not found. Ask for help.
    pause
    exit /b 1
)
call "%CONDA_CMD%" run --no-capture-output -n bearmind python -m bearmind gui %*
set "CODE=%ERRORLEVEL%"
rem a failure (no env "bearmind" for conda either, a crash) stays on screen until a key is pressed
if not "%CODE%"=="0" pause
exit /b %CODE%

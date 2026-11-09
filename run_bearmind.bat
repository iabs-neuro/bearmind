@echo off
setlocal
rem Student launcher: run_bearmind.bat <command> SERIES ...   (run_bearmind.bat --help lists commands)
set "PYTHONPATH=%~dp0"
rem The env's python is called directly when found: `conda run` prints an extra
rem "ERROR conda.cli.main_run" line on every non-zero exit, which looks like a crash.
rem Env folders: per-user and all-users Anaconda and Miniconda, then %USERPROFILE%\.conda\envs, where conda puts
rem a user's envs when Anaconda is installed for all users. bearmind_gui.bat searches the same folders.
set "PY="
for %%P in ("%USERPROFILE%\anaconda3\envs\bearmind\python.exe" "%USERPROFILE%\miniconda3\envs\bearmind\python.exe" "%ProgramData%\anaconda3\envs\bearmind\python.exe" "%ProgramData%\miniconda3\envs\bearmind\python.exe" "%USERPROFILE%\.conda\envs\bearmind\python.exe") do if not defined PY if exist "%%~P" set "PY=%%~P"
if not defined PY goto conda_run
"%PY%" -m bearmind %*
exit /b %ERRORLEVEL%

:conda_run
rem conda finds the env in any other folder. Not named CONDA_BAT: `conda run` reads CONDA_BAT as the path of
rem conda.bat, and CONDA_EXE (conda.exe in Anaconda Prompt) there makes it fail with "Run 'conda init'".
set "CONDA_CMD=%CONDA_EXE%"
if not defined CONDA_CMD set "CONDA_CMD=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not exist "%CONDA_CMD%" (
    echo [FAILED] conda env "bearmind" not found. Open Anaconda Prompt and run this command there, or ask for help.
    exit /b 1
)
call "%CONDA_CMD%" run --no-capture-output -n bearmind python -m bearmind %*
exit /b %ERRORLEVEL%

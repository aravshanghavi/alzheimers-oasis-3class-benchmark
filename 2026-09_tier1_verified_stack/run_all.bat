@echo off
REM Unattended batch runner. Close the window afterwards; logs are on disk.
REM   run_all.bat            normal batch
REM   run_all.bat --resume   skip anything already COMPLETE
REM   run_all.bat --isolate  each run in its own subprocess (safest for VRAM)

REM Activate the project environment. Bare `python` on this machine has torch
REM but NOT torchvision, so running without this fails immediately.
REM Change the name here if your environment is called something else.
call conda activate alzheimers 2>nul
if errorlevel 1 echo [warn] could not activate conda env "alzheimers" - continuing with %~dp0 default python

REM Reduces CUDA allocator fragmentation across a long sequence of runs.
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Keep the machine awake for the duration.
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

python "%~dp0run_all.py" %*
echo.
echo Finished with exit code %ERRORLEVEL%. Logs are in %~dp0console_logs
pause

@echo off
REM One command for the whole analysis layer.
REM   run_analysis.bat                 everything
REM   run_analysis.bat --skip-gradcam  skip the optional GPU step
call conda activate alzheimers 2>nul
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python "%~dp0run_analysis.py" %*
echo.
echo Exit code %ERRORLEVEL%. Outputs in %~dp0outputs
pause

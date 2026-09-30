@echo off
setlocal
title SREP round-2  MUSTs

rem =====================================================================
rem  MUST-PRIORITY WORK, ONE FILE.
rem  Dry runs are NOT in here. Do those yourself first.
rem
rem      run_musts.bat                GPU C1, then the MUST analyses
rem      run_musts.bat readout-only   skip the GPU, just re-run the analyses
rem      run_musts.bat gpu-only       GPU C1 only, no analyses
rem
rem  Contents:
rem      C1   exp15 unweighted reference on the primary cohort, 5 runs, 1.3 h
rem      A1   a25 incremental value of the images over MMSE and metadata
rem      A2   a26 paired AUC differences, nWBV against every network
rem      A3   a27 corrected metadata reference rows for Table 4
rem      A9   a28 unweighted reference on the full cohort, existing runs
rem      plus a23 and a24 as a re-verification pass
rem
rem  C1 carries --resume, so killing this window and running the file again
rem  skips whatever already finished.
rem =====================================================================

set ROOT=C:\Users\aravs\PycharmProjects\Alzheimers
set INSTR=%ROOT%\2026-09_instrumented_arms
set LOG=%INSTR%\analysis\outputs\_queue_log_musts.txt

set DOGPU=1
set DOREAD=1
if /i "%~1"=="readout-only" set DOGPU=0
if /i "%~1"=="gpu-only"     set DOREAD=0

echo.
echo ================================================================
echo  MUSTs  started %DATE% %TIME%
echo  log  %LOG%
echo ================================================================
echo.
echo [%DATE% %TIME%] ==== MUSTS START ==== >> "%LOG%"

call conda activate alzheimers 2>nul
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
echo Sleep and hibernate disabled on AC. Keep the laptop plugged in.
echo.

if "%DOGPU%"=="0" goto READOUT

echo ---------------- C1  exp15 unweighted reference, 5 runs, about 1.3 h
echo [%DATE% %TIME%] START C1 exp15 >> "%LOG%"
cd /d "%INSTR%"
python experiments\exp15_age60_unweighted\run.py --isolate --resume --console progress
call :LOGRC "C1 exp15 unweighted reference"

:READOUT
if "%DOREAD%"=="0" goto SUMMARY

echo.
echo ================================================================
echo  MUST ANALYSES. CPU only. About 9 minutes.
echo ================================================================
echo [%DATE% %TIME%] START must readout >> "%LOG%"
cd /d "%INSTR%"

python analysis\discover.py --check
call :LOGRC "discover --check"

rem The new arms are opt-in. Without this variable every table comes out
rem identical to the tables already on disk from 28 and 29 September.
set SREP_INCLUDE_NEW_ARMS=1

python run_musts.py
call :LOGRC "run_musts.py"

set SREP_INCLUDE_NEW_ARMS=
echo.
echo SREP_INCLUDE_NEW_ARMS cleared for this window.

:SUMMARY
echo.
echo ================================================================
echo  MUSTs DONE %DATE% %TIME%
echo ================================================================
echo [%DATE% %TIME%] ==== MUSTS END ==== >> "%LOG%"
echo.
type "%LOG%"
echo.
echo Tables are in %INSTR%\analysis\outputs
echo Next: run_shoulds.bat
echo.
goto END

:LOGRC
if errorlevel 1 (
  echo [%DATE% %TIME%] FAIL %~1 >> "%LOG%"
  echo.
  echo *** FAILED: %~1
  echo *** Continuing. Check the log before trusting any table.
  echo.
) else (
  echo [%DATE% %TIME%] OK   %~1 >> "%LOG%"
)
exit /b 0

:END
endlocal
echo Press any key to close.
pause >nul

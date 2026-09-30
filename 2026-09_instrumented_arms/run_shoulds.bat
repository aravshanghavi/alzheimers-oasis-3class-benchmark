@echo off
setlocal
title SREP round-2  SHOULDs

rem =====================================================================
rem  EVERYTHING THAT IS NOT A MUST, ONE FILE.
rem  Dry runs are NOT in here. Do those yourself first.
rem  Run run_musts.bat before this one.
rem
rem      run_shoulds.bat                GPU C3a, C2, C3b, then the analyses
rem      run_shoulds.bat with-c4        same, plus exp17 before the analyses
rem      run_shoulds.bat readout-only   skip the GPU, just re-run the analyses
rem      run_shoulds.bat gpu-only       GPU only, no analyses
rem
rem  GPU order and cost, measured from existing manifests:
rem      C3a  permutation null seeds 6 to 10,   25 runs,  5.9 h
rem      C2   exp16 initialisation variance,    20 runs,  5.3 h
rem      C3b  permutation null seeds 11 to 19,  45 runs, 10.7 h
rem      C4   exp17 multi-slice,                10 runs,  3.2 h   opt in only
rem
rem  C3b is the killable tail. Everything before it is banked.
rem  A permutation seed counts only when all five of its folds finish, so if
rem  the clock runs out, close the window at a seed boundary and then run
rem  run_shoulds.bat readout-only
rem
rem  Every GPU step carries --resume, so running this file again skips
rem  whatever already completed.
rem =====================================================================

set ROOT=C:\Users\aravs\PycharmProjects\Alzheimers
set INSTR=%ROOT%\2026-09_instrumented_arms
set PERM=%ROOT%\2026-09_permutation_null
set LOG=%INSTR%\analysis\outputs\_queue_log_shoulds.txt

set WITHC4=0
set DOGPU=1
set DOREAD=1
if /i "%~1"=="with-c4"      set WITHC4=1
if /i "%~1"=="readout-only" set DOGPU=0
if /i "%~1"=="gpu-only"     set DOREAD=0

echo.
echo ================================================================
echo  SHOULDs  started %DATE% %TIME%
echo  log  %LOG%
echo ================================================================
echo.
echo [%DATE% %TIME%] ==== SHOULDS START ==== >> "%LOG%"

call conda activate alzheimers 2>nul
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
echo Sleep and hibernate disabled on AC. Keep the laptop plugged in.
echo.

if "%DOGPU%"=="0" goto READOUT

echo ---------------- C3a  permutation null seeds 6 to 10, 25 runs, about 5.9 h
echo [%DATE% %TIME%] START C3a seeds 6-10 >> "%LOG%"
cd /d "%PERM%"
python run_permnull.py --first-seed 6 --seeds 5 --resume
call :LOGRC "C3a permutation null seeds 6 to 10"

echo.
echo ---------------- C2  exp16 initialisation variance, 20 runs, about 5.3 h
echo [%DATE% %TIME%] START C2 exp16 >> "%LOG%"
cd /d "%INSTR%"
python experiments\exp16_age60_init_variance\run.py --isolate --resume --console progress
call :LOGRC "C2 exp16 initialisation variance"

echo.
echo ---------------- C3b  permutation null seeds 11 to 19, 45 runs, about 10.7 h
echo ----------------
echo ---------------- KILLABLE TAIL. Everything above is banked.
echo ---------------- Close at a seed boundary, then: run_shoulds.bat readout-only
echo.
echo [%DATE% %TIME%] START C3b seeds 11-19 >> "%LOG%"
cd /d "%PERM%"
python run_permnull.py --first-seed 11 --seeds 9 --resume
call :LOGRC "C3b permutation null seeds 11 to 19"

if "%WITHC4%"=="0" goto READOUT

echo.
echo ---------------- C4  exp17 multi-slice, 10 runs, about 3.2 h
echo [%DATE% %TIME%] START C4 exp17 >> "%LOG%"
cd /d "%INSTR%"
python experiments\exp17_age60_multislice\run.py --isolate --resume --console progress
call :LOGRC "C4 exp17 multi-slice"

:READOUT
if "%DOREAD%"=="0" goto SUMMARY

echo.
echo ================================================================
echo  SHOULD, COULD AND ADDITION ANALYSES. CPU only. About 37 minutes.
echo  b01 reports SKIPPED unless the Desktop image folder is reachable.
echo ================================================================
echo [%DATE% %TIME%] START shoulds readout >> "%LOG%"
cd /d "%INSTR%"

python analysis\discover.py --check
call :LOGRC "discover --check"
python analysis\discover.py --summary
call :LOGRC "discover --summary"

rem The new arms are opt-in. Without this variable every table comes out
rem identical to the tables already on disk from 28 and 29 September.
set SREP_INCLUDE_NEW_ARMS=1

python run_shoulds.py
call :LOGRC "run_shoulds.py"

set SREP_INCLUDE_NEW_ARMS=
echo.
echo SREP_INCLUDE_NEW_ARMS cleared for this window.

:SUMMARY
echo.
echo ================================================================
echo  SHOULDs DONE %DATE% %TIME%
echo ================================================================
echo [%DATE% %TIME%] ==== SHOULDS END ==== >> "%LOG%"
echo.
type "%LOG%"
echo.
echo Tables are in %INSTR%\analysis\outputs
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

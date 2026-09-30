@echo off
REM Mechanism control for FA-FL plus the canonical focal baseline.
REM 10 runs, roughly 5 hours. Safe to leave unattended.
call conda activate alzheimers
python experiments\exp07_focal_gamma137\run.py > console_logs\F_focal_gamma137.txt 2>&1
python experiments\exp08_focal_gamma200\run.py > console_logs\G_focal_gamma200.txt 2>&1
python analysis\a14_gamma_control.py

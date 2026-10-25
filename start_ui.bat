@echo off
setlocal
cd /d "%~dp0"
title TDD-Net PCB Defect Detection v2 (phase 9 - pcb_30300)

if not exist "python\python.exe" (
  echo [LOI] Khong thay python\python.exe
  pause
  exit /b 1
)
if not exist "output\trained_weights\FPN_Res101_0117_OHEM\checkpoint" (
  echo [LOI] Chua co file checkpoint. Hay chay setup_venv.bat truoc.
  pause
  exit /b 1
)

echo Dang khoi dong (build graph mat ~40 giay, lan dau lau hon)...
echo Giu nguyen cua so den nay. Mo Chrome: http://127.0.0.1:5000
echo.
"python\python.exe" launcher.py

echo.
echo Server da dung.
pause

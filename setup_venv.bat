@echo off
setlocal
cd /d "%~dp0"
title TDD-Net-PCB v2 - Setup
echo ================================================
echo  TDD-Net-PCB-portable-v2 - SETUP (chay 1 lan)
echo ================================================
echo.
if not exist "python\python.exe" (
  echo [LOI] Khong thay python\python.exe
  echo Giai nen thieu folder "python" di kem.
  pause
  exit /b 1
)
if not exist ".venv\Lib\site-packages\flask" (
  echo [LOI] Khong thay .venv\Lib\site-packages
  echo Giai nen thieu folder ".venv" di kem.
  pause
  exit /b 1
)
echo [1/2] Viet lai file checkpoint theo may nay...
"python\python.exe" tools\fix_checkpoint.py
if errorlevel 1 (
  echo [LOI] checkpoint that bai.
  pause
  exit /b 1
)
echo [2/2] Xong.
echo.
echo  LUU Y: neu giai nen o duong dan co dau tieng Viet
echo  ma bao loi restore checkpoint, hay chuyen ca folder sang
echo  D:\TDD-Net-PCB-portable-v2 roi chay lai file nay.
echo.
echo  Tiep theo: chay start_ui.bat, doi hien
echo  "Model ready. Open http://127.0.0.1:5000" roi mo Chrome.
echo.
pause

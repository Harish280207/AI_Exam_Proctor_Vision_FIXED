@echo off
cd /d "%~dp0"
echo PROCTOR VISION SETUP
py -3.13 -m venv .venv
if errorlevel 1 (echo Python 3.13 is required.&pause&exit /b 1)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo SETUP COMPLETE
pause

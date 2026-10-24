@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Virtual environment not found. Run setup.bat first.&pause&exit /b 1)
.venv\Scripts\python.exe main.py
echo.
echo Application exited with code %ERRORLEVEL%
pause

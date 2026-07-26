@echo off
set PATH=D:\Develop\MyPythonLibs\nvidia\cublas\bin;%PATH%
chcp 65001 >nul
title VoxSub Launcher

set DIR=%~dp0
set PORT=8768

echo Killing old process on port %PORT%...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo Starting VoxSub server (loading model, please wait)...
echo.

cd /d "%DIR%"
start "VoxSub" python -u server.py

echo Waiting for model to load...
timeout /t 8 /nobreak >nul
start http://127.0.0.1:%PORT%
echo Done.


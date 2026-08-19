@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   Trading Bot - one-click start
echo ============================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python venv not found at .venv\Scripts\python.exe
    echo Create it first:  py -3.13 -m venv .venv
    echo Then install deps: .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "dashboard\.next\BUILD_ID" (
    echo Building dashboard (first run, takes ~1 min)...
    pushd dashboard
    call npm run build
    popd
    if errorlevel 1 (
        echo [ERROR] Dashboard build failed.
        pause
        exit /b 1
    )
)

echo Starting backend (port 8000) and dashboard (port 3001)...
start "TradingBot Backend" cmd /k ".venv\Scripts\python.exe scripts\run_live.py"
start "TradingBot Dashboard" cmd /k "cd /d dashboard && node node_modules\next\dist\bin\next start -p 3001"

timeout /t 5 /nobreak >nul
start "" "http://localhost:3001/theater"
echo.
echo Bot started. Close both windows (or Ctrl+C in each) to stop.
echo Note: port 3000 is not used - it belongs to another app.
endlocal
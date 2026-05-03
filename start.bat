@echo off
setlocal

title MT5 AI Analyst Bot

echo ============================================================
echo   MT5 AI Analyst Bot - Starting Server
echo ============================================================
echo.

:: Create required directories if missing
if not exist "data" mkdir data
if not exist "logs" mkdir logs

:: Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating .venv ...
    call .venv\Scripts\activate.bat
    goto :env_done
)
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating venv ...
    call venv\Scripts\activate.bat
    goto :env_done
)
echo [!] No virtual environment found - using system Python.

:env_done

:: Verify .env exists
if not exist ".env" (
    echo.
    echo [ERROR] .env file not found.
    echo         Copy .env.example to .env and fill in your API keys.
    echo.
    pause
    exit /b 1
)

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11+ and add it to PATH.
    pause
    exit /b 1
)

:: Launch server
echo [*] Starting FastAPI server on http://127.0.0.1:8000
echo [*] API docs: http://127.0.0.1:8000/docs
echo [*] Press Ctrl+C to stop.
echo.

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

echo.
echo [*] Server stopped.
pause

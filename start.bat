@echo off
setlocal

echo.
echo ============================================================
echo   Video Caption Suite
echo ============================================================
echo.

cd /d "%~dp0"

REM Check if venv exists, if not run install
if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found. Running installation...
    echo.

    REM Check if Python is available
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is not installed or not in PATH
        echo Please install Python 3.10+ from https://python.org
        pause
        exit /b 1
    )

    echo [1/3] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )

    echo [2/3] Installing pip...
    venv\Scripts\python.exe -m pip install --upgrade pip -q

    echo [3/3] Installing Python dependencies...
    venv\Scripts\pip.exe install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )

    echo.
    echo Installation complete!
    echo.
)

echo Starting server on http://localhost:8000
echo.
echo Press Ctrl+C to stop
echo.

REM Run uvicorn directly from venv Python (not relying on activate)
venv\Scripts\python.exe -m uvicorn backend.api:app --host 0.0.0.0 --port 8000

pause

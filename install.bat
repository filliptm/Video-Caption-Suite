@echo off
setlocal

echo.
echo ============================================================
echo   Video Caption Suite - Installation
echo ============================================================
echo.

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment already exists
)

echo [2/4] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip -q

echo [3/4] Installing Python dependencies...
venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some packages may have failed to install
)

REM Check if Node.js is available for frontend build
node --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [4/4] Skipping frontend build (Node.js not installed)
    echo       Frontend will use pre-built version if available
    echo       To rebuild frontend, install Node.js from https://nodejs.org
) else (
    echo [4/4] Building frontend...
    cd frontend
    if exist "package.json" (
        call npm install -q 2>nul
        call npm run build -q 2>nul
        if errorlevel 1 (
            echo WARNING: Frontend build may have failed
        ) else (
            echo       Frontend built successfully
        )
    )
    cd ..
)

echo.
echo ============================================================
echo   Installation Complete!
echo ============================================================
echo.
echo To start the application, run: start.bat
echo.
pause

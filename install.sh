#!/bin/bash

echo ""
echo "============================================================"
echo "  Video Caption Suite - Installation"
echo "============================================================"
echo ""

cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.10+ first"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
else
    echo "[1/4] Virtual environment already exists"
fi

echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo "[3/4] Installing Python dependencies..."
python -m pip install --upgrade pip -q
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "WARNING: Some packages may have failed to install"
fi

# Check if Node.js is available for frontend build
if ! command -v node &> /dev/null; then
    echo ""
    echo "[4/4] Skipping frontend build (Node.js not installed)"
    echo "      Frontend will use pre-built version if available"
    echo "      To rebuild frontend, install Node.js from https://nodejs.org"
else
    echo "[4/4] Building frontend..."
    cd frontend
    if [ -f "package.json" ]; then
        npm install -q 2>/dev/null
        npm run build -q 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "WARNING: Frontend build may have failed"
        else
            echo "      Frontend built successfully"
        fi
    fi
    cd ..
fi

echo ""
echo "============================================================"
echo "  Installation Complete!"
echo "============================================================"
echo ""
echo "To start the application, run: ./start.sh"
echo ""

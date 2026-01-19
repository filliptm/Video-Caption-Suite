#!/bin/bash

echo ""
echo "============================================================"
echo "  Video Caption Suite"
echo "============================================================"
echo ""

cd "$(dirname "$0")"

# Check if venv exists, if not run install
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    echo "Virtual environment not found. Running installation..."
    echo ""

    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: Python 3 is not installed"
        echo "Please install Python 3.10+ first"
        exit 1
    fi

    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi

    echo "[2/3] Activating virtual environment..."
    source venv/bin/activate

    echo "[3/3] Installing Python dependencies..."
    python -m pip install --upgrade pip -q
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi

    echo ""
    echo "Installation complete!"
    echo ""
else
    # Activate existing venv
    source venv/bin/activate
fi

echo "Starting server on http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000

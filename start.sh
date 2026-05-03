#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "  MT5 AI Analyst Bot — Starting Server"
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# Create required directories if missing
# ---------------------------------------------------------------------------
mkdir -p data logs

# ---------------------------------------------------------------------------
# Activate virtual environment (if present)
# ---------------------------------------------------------------------------
if [ -f ".venv/bin/activate" ]; then
    echo "[*] Activating virtual environment (.venv)..."
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "[*] Activating virtual environment (venv)..."
    # shellcheck disable=SC1091
    source venv/bin/activate
else
    echo "[!] No virtual environment found — using system Python."
fi

# ---------------------------------------------------------------------------
# Verify .env exists
# ---------------------------------------------------------------------------
if [ ! -f ".env" ]; then
    echo
    echo "[ERROR] .env file not found."
    echo "        Copy .env.example to .env and fill in your API keys."
    echo
    exit 1
fi

# ---------------------------------------------------------------------------
# Check Python
# ---------------------------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.11+ first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Launch server
# ---------------------------------------------------------------------------
echo "[*] Starting FastAPI server on http://127.0.0.1:8000"
echo "[*] Press Ctrl+C to stop."
echo

python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

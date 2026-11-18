#!/usr/bin/env bash

# SIH25099: Canopy Stress Scout - One-Click Launcher

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================="
echo "🌱 Starting Canopy Stress Scout (Backend + Frontend)"
echo "=========================================================="

# 1. Start Python Backend API (Port 8000)
echo "-> Starting Backend API on http://localhost:8000..."
python3 server.py &
BACKEND_PID=$!

cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill -9 $BACKEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

sleep 2

# Verify backend responded
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo " Backend API is healthy and listening on port 8000!"
else
    echo "⚠️ Warning: Backend did not respond on port 8000 yet. Continuing..."
fi

# 2. Start Frontend Dev Server (Port 5173)
if [ -d "frontend_repo" ]; then
    echo "-> Starting Frontend on http://localhost:5173..."
    cd frontend_repo
    npx vite --port 5173 --host
else
    echo "Error: frontend_repo directory not found!"
    exit 1
fi

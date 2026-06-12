#!/bin/bash
cd "$(dirname "$0")"
PORT=8765
echo "Stopping old server on port $PORT..."
lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null
sleep 0.5
echo "Starting server..."
python3 runner/server.py

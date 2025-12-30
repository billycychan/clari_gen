#!/bin/bash

# Script to stop the vLLM servers and their monitors

LOG_DIR="../vllm_logs"

echo "Stopping servers..."

# 1. Kill the monitor processes (the loops)
# We look for the monitor pid files
for pidfile in "$LOG_DIR"/*-monitor.pid; do
    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        echo "Killing monitor process $PID ($pidfile)..."
        kill $PID 2>/dev/null || echo "Monitor $PID not running."
        rm "$pidfile"
    fi
done

# 2. Kill the main serve_models.sh process if running
# It might be waited on by start_models_nohup.sh or just matching "serve_models.sh"
pkill -f "bash ./serve_models.sh" 2>/dev/null

# 3. Kill the actual vLLM processes
# We look for the model pid files
for pidfile in "$LOG_DIR"/*.pid; do
    if [[ "$pidfile" != *"-monitor.pid" ]] && [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        echo "Killing model process $PID ($pidfile)..."
        kill $PID 2>/dev/null || echo "Model $PID not running."
        rm "$pidfile"
    fi
done

# Fallback: Kill by name
echo "Cleaning up any remaining vLLM processes..."
pkill -f vllm

echo "Servers stopped."

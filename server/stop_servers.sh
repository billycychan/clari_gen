#!/bin/bash

# Script to stop vLLM servers

LOG_DIR="../vllm_logs"

echo "Stopping vLLM servers..."

# Stop meta-llama/Llama-3.2-3B
if [ -f "$LOG_DIR/llama-3.2-3b.pid" ]; then
  PID=$(cat "$LOG_DIR/llama-3.2-3b.pid")
  if ps -p $PID > /dev/null 2>&1; then
    echo "Stopping meta-llama/Llama-3.2-3B (PID: $PID)..."
    kill $PID
    rm "$LOG_DIR/llama-3.2-3b.pid"
    echo "Stopped."
  else
    echo "meta-llama/Llama-3.2-3B is not running."
    rm "$LOG_DIR/llama-3.2-3b.pid"
  fi
else
  echo "No PID file found for meta-llama/Llama-3.2-3B"
fi

# Stop nvidia/Llama-3.3-70B-Instruct-FP8
if [ -f "$LOG_DIR/llama-3.3-70b-fp8.pid" ]; then
  PID=$(cat "$LOG_DIR/llama-3.3-70b-fp8.pid")
  if ps -p $PID > /dev/null 2>&1; then
    echo "Stopping nvidia/Llama-3.3-70B-Instruct-FP8 (PID: $PID)..."
    kill $PID
    rm "$LOG_DIR/llama-3.3-70b-fp8.pid"
    echo "Stopped."
  else
    echo "nvidia/Llama-3.3-70B-Instruct-FP8 is not running."
    rm "$LOG_DIR/llama-3.3-70b-fp8.pid"
  fi
else
  echo "No PID file found for nvidia/Llama-3.3-70B-Instruct-FP8"
fi

echo ""
echo "All servers stopped."

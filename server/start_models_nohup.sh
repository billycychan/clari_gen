#!/bin/bash

# Alternative startup script for systems without tmux/screen installed.
# Uses nohup to detach the process from the terminal session.

LOG_FILE="../vllm_logs/server_manager.log"
mkdir -p "../vllm_logs"

echo "Starting serve_models.sh in the background..."
echo "Output will be redirected to $LOG_FILE"

# Run serve_models.sh with nohup, redirecting all output
nohup ./serve_models.sh > "$LOG_FILE" 2>&1 &
MANAGER_PID=$!

echo "Manager process started with PID: $MANAGER_PID"
echo "To check status:"
echo "  tail -f $LOG_FILE"
echo "To stop the server, you will need to find and kill the PIDs listed in the logs or use 'pkill -f vllm'"

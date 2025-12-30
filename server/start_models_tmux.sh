#!/bin/bash

# Name of the tmux session
SESSION_NAME="vllm_service"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    echo "Starting new tmux session: $SESSION_NAME"
    # Create new session in detached mode
    tmux new-session -d -s $SESSION_NAME
    
    # Send commands to the session
    # 1. Activate conda environment (assuming 'vllm' is the name)
    # Uses 'source' or 'conda activate' depending on shell init, 
    # but simplest is often just to run the command if conda is in path.
    # We'll allow the user's .bashrc to handle conda init if possible,
    # or explicitly try to activate.
    tmux send-keys -t $SESSION_NAME "source ~/.bashrc" C-m
    tmux send-keys -t $SESSION_NAME "conda activate vllm" C-m
    
    # 2. Run the serve script
    tmux send-keys -t $SESSION_NAME "cd $SCRIPT_DIR" C-m
    tmux send-keys -t $SESSION_NAME "./serve_models.sh" C-m
    
    echo "Session started. Models are booting up."
    echo "Attach to view logs with: tmux attach -t $SESSION_NAME"
else
    echo "Session $SESSION_NAME already exists."
    echo "Attach with: tmux attach -t $SESSION_NAME"
fi

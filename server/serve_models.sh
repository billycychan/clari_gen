#!/bin/bash

# Script to serve multiple models with vLLM in the background
# Models to serve:
# 1. meta-llama/Llama-3.2-3B (port 8368)
# 2. nvidia/Llama-3.3-70B-Instruct-FP8 (port 8369)

LOG_DIR="../vllm_logs"
mkdir -p "$LOG_DIR"

echo "Starting vLLM servers for multiple models..."

# Wait to ensure clean GPU state
sleep 2

# Start meta-llama/Llama-3.2-3B on port 8368 (single GPU 0 only)
echo "Starting meta-llama/Llama-3.2-3B on port 8368..."
CUDA_VISIBLE_DEVICES=0 nohup vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --port 8368 \
  --dtype auto \
  --api-key token-abc123 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 65536 \
  > "$LOG_DIR/llama-3.2-3b.log" 2>&1 &

LLAMA_3_2_PID=$!
echo "Started meta-llama/Llama-3.2-3B (PID: $LLAMA_3_2_PID)"

# Wait for first model to initialize and claim its GPU
sleep 5

# Start nvidia/Llama-3.3-70B-Instruct-FP8 on port 8369 (GPUs 2,3 only)
echo "Starting nvidia/Llama-3.3-70B-Instruct-FP8 on port 8369..."
CUDA_VISIBLE_DEVICES=2,3 nohup vllm serve nvidia/Llama-3.3-70B-Instruct-FP8 \
  --port 8369 \
  --dtype auto \
  --api-key token-abc123 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  > "$LOG_DIR/llama-3.3-70b-fp8.log" 2>&1 &

LLAMA_3_3_PID=$!
echo "Started nvidia/Llama-3.3-70B-Instruct-FP8 (PID: $LLAMA_3_3_PID)"

# Save PIDs to a file for easy management
echo "$LLAMA_3_2_PID" > "$LOG_DIR/llama-3.2-3b.pid"
echo "$LLAMA_3_3_PID" > "$LOG_DIR/llama-3.3-70b-fp8.pid"

echo ""
echo "===================================================================="
echo "vLLM servers started successfully!"
echo "===================================================================="
echo ""
echo "Server details:"
echo "  1. meta-llama/Llama-3.2-3B-Instruct"
echo "     - Port: 8368"
echo "     - PID: $LLAMA_3_2_PID"
echo "     - Log: $LOG_DIR/llama-3.2-3b.log"
echo "     - API endpoint: http://localhost:8368/v1"
echo ""
echo "  2. nvidia/Llama-3.3-70B-Instruct-FP8"
echo "     - Port: 8369"
echo "     - PID: $LLAMA_3_3_PID"
echo "     - Log: $LOG_DIR/llama-3.3-70b-fp8.log"
echo "     - API endpoint: http://localhost:8369/v1"
echo ""
echo "===================================================================="
echo ""
echo "To check logs:"
echo "  tail -f $LOG_DIR/llama-3.2-3b.log"
echo "  tail -f $LOG_DIR/llama-3.3-70b-fp8.log"
echo ""
echo "To stop servers:"
echo "  ./stop_servers.sh"
echo ""
echo "To test the APIs:"
echo "  curl http://localhost:8368/v1/models -H 'Authorization: Bearer token-abc123'"
echo "  curl http://localhost:8369/v1/models -H 'Authorization: Bearer token-abc123'"
echo ""

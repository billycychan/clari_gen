# vLLM Model Serving Scripts

[← Back to Root](../README.md)

This directory contains scripts to serve multiple LLMs using vLLM's OpenAI-compatible server.

## Models

1. **meta-llama/Llama-3.1-8B-Instruct** - Served on port 8368
2. **nvidia/Llama-3.3-70B-Instruct-FP8** - Served on port 8369

## Prerequisites

```bash
# Install vLLM
pip install vllm

# Install OpenAI client for testing (optional)
pip install openai
```

## Usage

### Start the Servers

```bash
cd server
chmod +x serve_models.sh
./serve_models.sh
```

This will:
- Start both models in the background
- Create logs in `../vllm_logs/` directory
- Save PIDs for easy management
- Display server information

### Stop the Servers

```bash
cd server
chmod +x stop_servers.sh
./stop_servers.sh
```

### Check Logs

```bash
# View logs for Llama 3.1 8B
tail -f ../vllm_logs/llama-3.1-8b.log

# View logs for Llama 3.3 70B FP8
tail -f ../vllm_logs/llama-3.3-70b-fp8.log
```

### Test the Servers

Using curl:
```bash
# List models on port 8368
curl http://localhost:8368/v1/models \
  -H "Authorization: Bearer token-abc123"

# List models on port 8369
curl http://localhost:8369/v1/models \
  -H "Authorization: Bearer token-abc123"

# Chat completion example
curl http://localhost:8368/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

Using Python:
```bash
cd server
chmod +x test_servers.py
python test_servers.py
```

## Configuration

### API Key
The default API key is `token-abc123`. You can change it in `serve_models.sh` by modifying the `--api-key` parameter.

### Tensor Parallelism
The 70B model uses `--tensor-parallel-size 2` for multi-GPU inference on GPUs 2-3. The 8B model runs on GPU 0. This configuration is optimized for 4x H100 GPUs.

### Memory and Performance
Additional optional parameters you can add to `serve_models.sh`:

```bash
--max-model-len 4096          # Maximum sequence length
--gpu-memory-utilization 0.9  # GPU memory utilization (0.0-1.0)
--max-num-seqs 256           # Maximum number of sequences per iteration
--enforce-eager              # Disable CUDA graphs (useful for debugging)
```

## OpenAI Client Usage

Example using the official OpenAI Python client:

```python
from openai import OpenAI

# Client for Llama 3.1 8B
client = OpenAI(
    api_key="token-abc123",
    base_url="http://localhost:8368/v1",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    max_tokens=200,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

## Troubleshooting

### Server won't start
- Check if ports 8368 and 8369 are already in use
- Verify you have sufficient GPU memory
- Check the log files in `../vllm_logs/`

### Out of memory errors
- Reduce `--max-model-len`
- Reduce `--gpu-memory-utilization`
- For 70B model, ensure you have enough GPUs and adjust `--tensor-parallel-size`

### Model not found
- Ensure you have access to the models on Hugging Face
- Set `HF_TOKEN` environment variable if needed:
  ```bash
  export HF_TOKEN=your_huggingface_token
  ```

## API Endpoints

Both servers expose the following OpenAI-compatible endpoints:

- `/v1/models` - List available models
- `/v1/completions` - Text completions
- `/v1/chat/completions` - Chat completions
- `/v1/embeddings` - Embeddings (if model supports)

Full API documentation: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/

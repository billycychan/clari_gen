# Ambiguity Detection and Clarification System

A multi-model pipeline using vLLM-served LLMs to detect query ambiguity, classify ambiguity types, generate clarifying questions, and reformulate queries.

## Overview

This system uses two models in a coordinated pipeline:
- **Llama-3.1-8B-Instruct** (small model) - Fast ambiguity detection
- **Llama-3.3-70B-Instruct-FP8** (large model) - Classification, clarification generation, validation, and reformulation

### How It Works

1. **Ambiguity Detection**: User submits a query → 3B model determines if it's ambiguous
2. **Non-Ambiguous Path**: If clear → return original query
3. **Ambiguous Path**: If ambiguous → 70B model:
   - Classifies the ambiguity type
   - Generates a clarifying question
   - Validates user's clarification (ensures it resolves the ambiguity)
   - Reformulates the query with the clarification

### Ambiguity Types

The system recognizes 8 types of ambiguity:

| Type | Explanation | Example |
|------|-------------|---------|
| **UNFAMILIAR** | Query contains unfamiliar entities or facts | "Find the price of Samsung Chromecast." |
| **CONTRADICTION** | Query contains self-contradictions | Complex contradictory instructions |
| **LEXICAL** | Query contains terms with multiple meanings | "Tell me about the source of Nile." |
| **SEMANTIC** | Query lacks context leading to multiple interpretations | "When did he land on the moon?" |
| **WHO** | Missing personal elements | "Suggest me some gifts for my mother." |
| **WHEN** | Missing temporal elements | "How many goals did Argentina score in the World Cup?" |
| **WHERE** | Missing spatial elements | "Tell me how to reach New York." |
| **WHAT** | Missing task-specific elements | "Real name of gwen stacy in spiderman?" |

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Running vLLM servers** (see `server/README.md`)

### Setup

```bash
cd clari_gen

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m clari_gen.main --test
```

## Usage

### Interactive Mode

Run the system interactively to process queries and provide clarifications:

```bash
python -m clari_gen.main
```

Example session:
```
Enter your query: When did he land on the moon?

📊 Processing query...

======================================================================
CLARIFICATION NEEDED
======================================================================

Who are you referring to? Please specify the person's name.

Your answer: Neil Armstrong

======================================================================
RESULT
======================================================================

✓ Original query: When did he land on the moon?
✓ Ambiguity type: SEMANTIC
✓ Your clarification: Neil Armstrong

🎯 Reformulated query: When did Neil Armstrong land on the moon?
```

### Batch Mode

Process a single query non-interactively:

```bash
# Simple output
python -m clari_gen.main --query "What is the capital of France?"

# JSON output (useful for scripting)
python -m clari_gen.main --query "Tell me about the source" --json
```

### Test Server Connections

```bash
python -m clari_gen.main --test
```

### Configuration

Set log level and other parameters:

```bash
# Debug mode
python -m clari_gen.main --log-level DEBUG

# Custom max clarification attempts
python -m clari_gen.main --max-attempts 5
```

## Examples

Run the demo scripts:

```bash
# Simple demo (non-interactive)
python examples/simple_demo.py

# Full interactive demo with simulated clarifications
python examples/interactive_demo.py
```

## Testing

Run the test suite:

```bash
# Unit tests (no servers required)
pytest tests/test_orchestrator.py -v

# Integration tests (requires running servers)
export SKIP_INTEGRATION_TESTS=false
pytest tests/test_integration.py -v

# All tests
pytest tests/ -v
```

## Architecture

```
clari_gen/
├── models/              # Data models and ambiguity taxonomy
│   ├── ambiguity_types.py
│   ├── query.py
│   └── conversation.py
├── clients/             # vLLM client wrappers
│   ├── base_client.py
│   ├── small_model_client.py (3B, temp=0.3)
│   └── large_model_client.py (70B, temp=0.3/0.7)
├── prompts/             # Prompt engineering
│   ├── ambiguity_detection.py
│   ├── ambiguity_classification.py
│   ├── clarification_generation.py
│   ├── clarification_validation.py
│   └── query_reformulation.py
├── orchestrator/        # Pipeline coordination
│   └── ambiguity_pipeline.py
├── utils/               # Utilities
│   └── logger.py
├── config.py            # Configuration management
└── main.py             # CLI interface
```

## API Usage

Use the pipeline programmatically:

```python
from clari_gen.orchestrator import AmbiguityPipeline
from clari_gen.clients import SmallModelClient, LargeModelClient

# Initialize clients
small_client = SmallModelClient()
large_client = LargeModelClient()

# Create pipeline
pipeline = AmbiguityPipeline(
    small_model_client=small_client,
    large_model_client=large_client,
)

# Process without clarification
result = pipeline.process_query("What is 2+2?")
print(result.get_final_output())  # "What is 2+2?"

# Process with clarification callback
def get_clarification(question):
    return input(f"{question}\nYour answer: ")

result = pipeline.process_query(
    "When did he land on the moon?",
    clarification_callback=get_clarification
)

print(result.get_final_output())  # Reformulated query
```

## Temperature Settings

The system uses different temperatures for different tasks:
- **Ambiguity Detection** (3B): `0.3` - Low for consistent binary classification
- **Classification** (70B): `0.3` - Low for consistent type identification  
- **Clarification Generation** (70B): `0.7` - Higher for natural question generation
- **Validation** (70B): `0.3` - Low for consistent validation
- **Reformulation** (70B): `0.7` - Higher for natural query reformulation

## Environment Variables

Customize behavior via environment variables:

```bash
# Model endpoints (defaults to localhost)
export SMALL_MODEL_URL="http://localhost:8368/v1"
export LARGE_MODEL_URL="http://localhost:8369/v1"

# API authentication
export VLLM_API_KEY="token-abc123"

# Pipeline configuration
export MAX_CLARIFICATION_ATTEMPTS="3"
export LOG_LEVEL="INFO"

# Testing
export SKIP_INTEGRATION_TESTS="false"
```

## Server Management

The vLLM model servers are managed separately. See `server/README.md` for details.

```bash
# Start servers
cd server && ./serve_models.sh

# Stop servers
cd server && ./stop_servers.sh

# Check logs
tail -f ../vllm_logs/llama-3.1-8b.log
```

## Troubleshooting

### Connection errors
- Ensure vLLM servers are running: `cd server && ./serve_models.sh`
- Test connections: `python -m clari_gen.main --test`

### Import errors
- Install dependencies: `pip install -r requirements.txt`
- Ensure you're in the correct directory

### Model not responding
- Check server logs in `vllm_logs/`
- Verify GPU availability
- Check port availability (8368, 8369)

## License

This project is part of the clari_gen research system.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional ambiguity type detection
- Multi-turn clarification dialogs
- Confidence scoring for ambiguity detection
- Support for additional model backends

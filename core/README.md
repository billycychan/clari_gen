# ClariGen Core Library

This is the core library for ClariGen, containing the fundamental components for ambiguity detection and clarification.

## Components

- **`orchestrator/`** - Pipeline orchestration and workflow management
- **`clients/`** - LLM client interfaces (small and large models)
- **`models/`** - Data models and schemas
- **`prompts/`** - Prompt templates for LLMs
- **`utils/`** - Utilities and logging
- **`config.py`** - Configuration management

## Installation

From the project root:

```bash
cd core
pip install -e .
```

## Usage

```python
from clari_gen.config import Config
from clari_gen.clients import SmallModelClient, LargeModelClient
from clari_gen.orchestrator import AmbiguityPipeline

# Initialize clients
small_client = SmallModelClient(base_url="http://localhost:8368/v1")
large_client = LargeModelClient(base_url="http://localhost:8369/v1")

# Create pipeline
pipeline = AmbiguityPipeline(
    small_model_client=small_client,
    large_model_client=large_client
)

# Process a query
result = pipeline.process_query("Your query here")
```

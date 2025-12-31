# ClariGen: Ambiguity Detection and Clarification System

ClariGen is a multi-model pipeline for detecting ambiguous queries and generating clarifications. It leverages both small and large language models to identify ambiguity, classify its type, and generate clarifying questions or reformulations. The system is designed for research and practical applications in information retrieval, conversational AI, and question answering.

## Features
- **Ambiguity Detection:** Uses a small, efficient LLM to quickly detect if a query is ambiguous.
- **Ambiguity Classification & Clarification:** Simultaneously classifies the ambiguity type and generates a clarifying question using a large LLM.
- **Query Reformulation:** Reformulates the original query based on the user's response.
- **Interactive Confirmation:** Allows users to review and confirm the reformulated query.
- **Flexible Strategies:** Supports multiple prompting strategies for clarification (Standard, Chain-of-Thought, Vanilla).
- **Interactive Web UI:** Run interactively via a Streamlit web app.
- **API Access:** FastAPI backend for programmatic access.

## Project Structure

The project is organized into clearly separated components:

```
clari_gen/
├── core/                          # Core library
│   ├── clari_gen/                # Main package
│   │   ├── orchestrator/         # Pipeline orchestration
│   │   ├── clients/              # LLM client interfaces
│   │   ├── models/               # Data models & schemas
│   │   ├── prompts/              # Prompt templates
│   │   ├── utils/                # Utilities & logging
│   │   └── config.py             # Configuration
│   ├── setup.py
│   ├── requirements.txt
│   └── README.md
│
├── apps/                          # Applications
│   ├── api/                      # FastAPI backend
│   └── frontend/                 # Streamlit web UI
│
├── evaluation/                    # Evaluation scripts & datasets
│   ├── scripts/                  # Evaluation scripts
│   ├── results/                  # Results
│   ├── data/                     # Datasets
│   └── README.md
│
├── deployment/                    # Deployment configs
│   ├── docker/                   # Docker files
│   └── README.md
│
├── llm_hosting/                   # LLM server management
│   ├── serve_models.sh
│   ├── stop_models.sh
│   └── README.md
│
├── docs/                          # Documentation
│   └── ARCHITECTURE.md
│
└── tests/                         # Tests
```

See component-specific READMEs for detailed documentation:
- [Core Library](core/README.md)
- [API Application](apps/api/README.md)
- [Frontend Application](apps/frontend/README.md)
- [Evaluation](evaluation/README.md)
- [Deployment](deployment/README.md)
- [LLM Hosting](llm_hosting/README.md)
- [Environment Configuration](docs/ENVIRONMENT.md)

## Configuration

ClariGen uses environment variables for configuration. 

**Quick setup:**
```bash
cp .env.example .env
# Edit .env with your settings
```

See the [Environment Configuration Guide](docs/ENVIRONMENT.md) for detailed information about all available settings.

**Key variables:**
- `SMALL_MODEL_URL` - Small model server URL (default: `http://localhost:8368/v1`)
- `LARGE_MODEL_URL` - Large model server URL (default: `http://localhost:8369/v1`)
- `API_URL` - API backend URL for frontend (default: `http://localhost:8370/v1`)
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `CLARIFICATION_STRATEGY` - Strategy: `at_standard`, `at_cot`, or `vanilla`


## Installation

### 1. Install Core Library

```bash
cd core
pip install -e .
```

### 2. Install Component Dependencies

For specific components, install their requirements:

```bash
# API
pip install -r apps/api/requirements.txt

# Frontend
pip install -r apps/frontend/requirements.txt

# Evaluation
pip install -r evaluation/requirements.txt

# LLM Hosting
pip install -r llm_hosting/requirements.txt
```

## Quick Start

### 1. Start Model Servers (vLLM)

Before running the system, start the vLLM model servers:

```bash
cd llm_hosting
./serve_models.sh
```

This starts:
- **Llama 3.1 8B** on port 8368
- **Llama 3.3 70B FP8** on port 8369

See [LLM Hosting README](llm_hosting/README.md) for details.

### 2. Run Applications
**API Server:**
```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8370
```

**Web UI:**
```bash
streamlit run apps/frontend/app.py --server.port 8501
```

See component READMEs for detailed usage instructions.

## Docker Deployment

```bash
cd deployment/docker
docker-compose up -d
```

This starts:
- API on port 8370
- Frontend on port 8371

See [Deployment README](deployment/README.md) for details.

## Evaluation

Run evaluation scripts to test system performance:

```bash
cd evaluation
python scripts/evaluate_ambiguity_classification.py --dataset data/ambignq_preprocessed.tsv
```

See [Evaluation README](evaluation/README.md) for all available evaluation scripts.

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system architecture and flow diagrams.

## Requirements

- Python >= 3.8
- vLLM >= 0.6.0 (for model serving)
- See component-specific `requirements.txt` files for dependencies

## Contributing

Contributions are welcome! Please open issues or pull requests.


## License

Copyright © 2025, Wireless System Research Group (WiSeR), McMaster University.
This work is licensed under the terms of a Creative Commons Attribution–
NonCommercial 4.0 International License (CC BY-NC 4.0).

You are free to:
- **Share** — copy and redistribute the material in any medium or format.
- **Adapt** — remix, transform, and build upon the material.

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. Credit should be given as: “Wireless System Research Group (WiSeR), McMaster University. Used with permission.”
- **NonCommercial** — You may not use the material for commercial purposes.
- **Permission for Commercial Use** — Any commercial use requires prior written permission from Rong Zheng (rzheng@mcmaster.ca).

**Disclaimer**: This code and related artifacts are provided “as is,” without warranties or conditions of any kind, either express or implied.


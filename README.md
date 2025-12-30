# ClariGen: Ambiguity Detection and Clarification System

ClariGen is a multi-model pipeline for detecting ambiguous queries and generating clarifications. It leverages both small and large language models to identify ambiguity, classify its type, and generate clarifying questions or reformulations. The system is designed for research and practical applications in information retrieval, conversational AI, and question answering.

## Features
- **Ambiguity Detection:** Uses a small, efficient LLM to quickly detect if a query is ambiguous.
- **Ambiguity Classification & Clarification:** Simultaneously classifies the ambiguity type and generates a clarifying question using a large LLM.
- **Query Reformulation:** Reformulates the original query based on the user's response.
- **Interactive Confirmation:** Allows users to review and confirm the reformulated query.
- **Flexible Strategies:** Supports multiple prompting strategies for clarification (Standard, Chain-of-Thought, Vanilla).
- **Interactive CLI and Web UI:** Run interactively in the terminal or via a Streamlit web app.
- **API Access:** FastAPI backend for programmatic access.

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd clari_gen
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Install in editable mode:**
   ```bash
   pip install -e .
   ```

### Model Servers (vLLM)
Before running the system, you need to start the vLLM model servers. A script is provided to manage this:

```bash
cd server
./serve_models.sh
```

This will start:
- **Llama 3.1 8B** on port 8368
- **Llama 3.3 70B FP8** on port 8369

To stop the servers:
```bash
./stop_servers.sh
```

### API (FastAPI)
The backend API handles the orchestration of the ambiguity pipeline.

```bash
uvicorn clari_gen.api.main:app --port 8000 --reload
```

### Web UI (Streamlit)
The Streamlit interface provides a user-friendly way to interact with the system.

```bash
streamlit run clari_gen/frontend/app.py --server.port 8501 
```

### CLI
Run the main interface in interactive mode:
```bash
python -m clari_gen
```

Process a single query:
```bash
python -m clari_gen --query "Who won the championship?"
```

Process with a specific clarification strategy (at_standard, at_cot, or vanilla):
```bash
python -m clari_gen --query "..." --clarification-strategy at_cot
```

Test model server connections:
```bash
python -m clari_gen --test
```

## Project Structure
- `clari_gen/` - Main package
  - `ARCHITECTURE.md` - System architecture and flow diagrams
  - `main.py` - CLI entry point
  - `config.py` - Configuration management
  - `orchestrator/ambiguity_pipeline.py` - Main pipeline logic
  - `clients/` - Model client interfaces
  - `frontend/app.py` - Streamlit web app
  - `api/main.py` - FastAPI backend
  - `models/` - Data models and schemas
  - `prompts/` - Prompt templates for LLMs
  - `utils/` - Utilities and logging
- `eval/` - Evaluation scripts and results
- `data/` - Example datasets
- `tests/` - Unit and integration tests

## Requirements
See `requirements.txt` for all dependencies. Key packages:
- openai, vllm, fastapi, streamlit, torch, pandas, requests

## Contributing
Contributions are welcome! Please open issues or pull requests.

## License
[MIT License](LICENSE)

## Contact
For questions or collaboration, contact the project author or open an issue.

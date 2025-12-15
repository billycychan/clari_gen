# ClariGen: Ambiguity Detection and Clarification System

ClariGen is a multi-model pipeline for detecting ambiguous queries and generating clarifications. It leverages both small and large language models to identify ambiguity, classify its type, and generate clarifying questions or reformulations. The system is designed for research and practical applications in information retrieval, conversational AI, and question answering.

## Features
- **Ambiguity Detection:** Uses a small LLM to detect if a query is ambiguous.
- **Ambiguity Classification:** Classifies the type of ambiguity using a large LLM.
- **Clarification Generation:** Generates clarifying questions or reformulated queries.
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

## Usage
### CLI
Run the main interface:
```bash
python -m clari_gen
```
Or use the installed command:
```bash
clari-gen
```

### Web UI
Start the Streamlit app:
```bash
streamlit run clari_gen/frontend/app.py
```

### API
Start the FastAPI server:
```bash
uvicorn clari_gen.api.main:app --reload
```

## Project Structure
- `clari_gen/` - Main package
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

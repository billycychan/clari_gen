# ClariGen Streamlit Frontend

Interactive web UI for the ClariGen ambiguity detection and clarification system.

## Installation

Ensure the core library is installed:
```bash
cd ../../core
pip install -e .
```

Install frontend dependencies:
```bash
pip install streamlit requests pandas
```

## Running

```bash
streamlit run apps/frontend/app.py --server.port 8501
```

## Configuration

Set environment variables:
- `API_URL` - URL for the FastAPI backend (default: http://localhost:8000/v1)

## Features

- **Query Selection** - Load example queries from a TSV file
- **Interactive Analysis** - Submit queries and receive real-time clarification
- **Conversation History** - View the full conversation flow
- **API Inspector** - See detailed API requests and responses
- **Context Debugging** - Inspect the internal context blob

## Usage

1. Start the API backend first (see `apps/api/README.md`)
2. Launch the Streamlit app
3. Enter or select a query
4. Click "Analyze Query"
5. If ambiguous, provide clarification when prompted
6. Confirm or modify the reformulated query

# ClariGen FastAPI Backend

REST API service for the ClariGen ambiguity detection and clarification system.

## Installation

Ensure the core library is installed:
```bash
cd ../../core
pip install -e .
```

Install API dependencies:
```bash
pip install fastapi uvicorn
```

## Running

```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8370
```

Or with hot reload:
```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8370 --reload
```

## API Endpoints

### POST /v1/query
Process a new query.

**Request:**
```json
{
  "text": "Who won the championship?"
}
```

**Response:**
- If clear: `{"status": "completed", "reformulated_query": "..."}`
- If ambiguous: `{"status": "clarification_needed", "clarifying_question": "...", "context": {...}}`

### POST /v1/clarify
Process a clarification response. Requires context blob from `/v1/query`.

**Request:**
```json
{
  "answer": "NBA championship",
  "context": {...}
}
```

**Response:**
```json
{
  "status": "confirmation_needed",
  "reformulated_query": "Who won the NBA championship?",
  "context": {...}
}
```

### POST /v1/confirm
Confirm the reformulated query.

**Request:**
```json
{
  "confirmation": "yes",
  "context": {...}
}
```

Or with alternative:
```json
{
  "confirmation": "no",
  "alternative_query": "Who won the NBA Finals in 2024?",
  "context": {...}
}
```

**Response:**
```json
{
  "status": "completed",
  "confirmed_query": "..."
}
```

### GET /health
Health check endpoint.

## Configuration

Set environment variables:
- `SMALL_MODEL_URL` - URL for the small model server (default: http://localhost:8368/v1)
- `LARGE_MODEL_URL` - URL for the large model server (default: http://localhost:8369/v1)
- `VLLM_API_KEY` - API key for vLLM servers
- `LOG_LEVEL` - Logging level (default: INFO)

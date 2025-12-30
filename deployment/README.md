# ClariGen Deployment

Docker deployment configurations for ClariGen.

## Prerequisites

- Docker and Docker Compose installed
- vLLM model servers running (see `../llm_hosting/README.md`)

## Quick Start

From the project root:

```bash
cd deployment/docker
docker-compose up -d
```

This will start:
- **API Service** on port 8370
- **Frontend Service** on port 8371

## Configuration

### Environment Variables

Edit `docker-compose.yml` to configure:

```yaml
environment:
  - SMALL_MODEL_URL=http://130.113.68.24:8368/v1
  - LARGE_MODEL_URL=http://130.113.68.24:8369/v1
  - VLLM_API_KEY=token-abc123
  - LOG_LEVEL=INFO
```

Update model URLs to point to your vLLM servers.

## Services

### API (port 8370)
FastAPI backend service handling ambiguity detection pipeline.

**Dockerfile:** `Dockerfile.api`

**Endpoints:**
- `http://localhost:8370/v1/query` - Process queries
- `http://localhost:8370/v1/clarify` - Submit clarifications
- `http://localhost:8370/v1/confirm` - Confirm reformulations
- `http://localhost:8370/health` - Health check

### Frontend (port 8371)
Streamlit web UI for interactive usage.

**Dockerfile:** `Dockerfile.frontend`

**Access:** Open `http://localhost:8371` in your browser
C
## Docker Commands

### Start services
```bash
docker-compose up -d
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f frontend
```

### Stop services
```bash
docker-compose down
```

### Rebuild images
```bash
docker-compose build --no-cache
docker-compose up -d
```

### Restart a service
```bash
docker-compose restart api
docker-compose restart frontend
```

## Development Mode

For development with hot reload:

1. Mount source code as volume in `docker-compose.yml`:
```yaml
services:
  api:
    volumes:
      - ../../core:/app/core
      - ../../apps:/app/apps
```

2. Enable reload in commands:
```yaml
command: uvicorn apps.api.main:app --host 0.0.0.0 --port 8370 --reload
```

## Troubleshooting

### Container won't start
```bash
# Check container logs
docker-compose logs api
docker-compose logs frontend

# Check container status
docker-compose ps
```

### Cannot connect to model servers
Ensure the `SMALL_MODEL_URL` and `LARGE_MODEL_URL` are reachable from within the Docker containers. Use host IPs, not `localhost`.

### Port conflicts
If ports 8370 or 8371 are in use, modify the port mappings in `docker-compose.yml`:
```yaml
ports:
  - "9000:8370"  # Map to different host port
```

## Production Deployment

For production:

1. Use production-ready web server for API (e.g., gunicorn with uvicorn workers)
2. Add nginx reverse proxy
3. Enable HTTPS with SSL certificates
4. Configure proper logging and monitoring
5. Set up health checks and auto-restart policies
6. Use Docker secrets for sensitive environment variables

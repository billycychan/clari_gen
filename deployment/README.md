# ClariGen Deployment

[← Back to Root](../README.md)

This directory contains configurations for deploying ClariGen using Docker. It supports both local-only setups and configurations that connect to remote models via secure SSH tunnels.

## Deployment Options

### 1. Local Deployment
Ideal for development when you have model servers running on your local machine or an accessible network.

### 2. Remote Deployment with SSH Tunnel
Required when models (Llama-3.1-8B, Llama-3.3-70B) are hosted on a remote server (e.g., `grace.cas.mcmaster.ca`) that requires SSH access.

> [!IMPORTANT]
> For SSH Tunneling, you must follow the [SSH Setup Guide](docker/README.md) to configure your `id_rsa` key.

---

## Quick Start (Docker Compose)

1. **Configure Environment**:
   Ensure your `.env` file in the root is set up correctly (see [.env.example](../.env.example)).

2. **Start Services**:
   ```bash
   cd deployment/docker
   docker-compose up -d
   ```

3. **Verify**:
   - **Frontend**: [http://localhost:8371](http://localhost:8371)
   - **API Backend**: [http://localhost:8370](http://localhost:8370)

---

## Service Overview

### [FastAPI Backend](docker/Dockerfile.api)
The brain of the operation, handling the ambiguity detection and clarification pipeline.
- **Port**: 8370
- **Health Check**: `GET /health`

### [Streamlit Frontend](docker/Dockerfile.frontend)
Interactive dashboard for testing and visualizing the system.
- **Port**: 8371

### [SSH Tunnel](docker/README.md) (Optional)
A dedicated container that establishes an SSH tunnel to forward remote model ports to the local Docker network.

---

## Common Commands

```bash
# View all logs
docker-compose logs -f

# Logs for a specific service
docker-compose logs -f api

# Restart everything
docker-compose down && docker-compose up -d

# Rebuild images after code changes
docker-compose build --no-cache
```

## Management & Troubleshooting

See the [Docker README](docker/README.md) for detailed troubleshooting steps regarding SSH connections, port mapping, and volume management.

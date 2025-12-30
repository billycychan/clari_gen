# Environment Configuration Guide

ClariGen uses environment variables for configuration. This allows you to easily customize settings for different environments (development, production, Docker, etc.) without modifying code.

## Quick Setup

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your settings:**
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **The `.env` file is automatically loaded** when you import the ClariGen config.

## Environment Variables

### Model Server Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SMALL_MODEL_URL` | URL for the small model (8B) server | `http://localhost:8368/v1` |
| `SMALL_MODEL_NAME` | Name of the small model | `meta-llama/Llama-3.1-8B-Instruct` |
| `LARGE_MODEL_URL` | URL for the large model (70B) server | `http://localhost:8369/v1` |
| `LARGE_MODEL_NAME` | Name of the large model | `nvidia/Llama-3.3-70B-Instruct-FP8` |
| `VLLM_API_KEY` | API key for vLLM servers | `token-abc123` |

### Application Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `API_URL` | URL for the FastAPI backend (used by frontend) | `http://localhost:8370/v1` |

### Pipeline Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_CLARIFICATION_ATTEMPTS` | Maximum clarification attempts before giving up | `3` |
| `CLARIFICATION_STRATEGY` | Strategy: `at_standard`, `at_cot`, or `vanilla` | `at_standard` |

### Logging Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |
| `LOG_FILE` | Path to log file (empty = console only) | _(empty)_ |

### Testing Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SKIP_INTEGRATION_TESTS` | Skip integration tests (`true`/`false`) | `false` |

## Usage Examples

### Local Development

For local development with models running on localhost:

```bash
# .env
SMALL_MODEL_URL=http://localhost:8368/v1
LARGE_MODEL_URL=http://localhost:8369/v1
API_URL=http://localhost:8370/v1
LOG_LEVEL=DEBUG
```

### Remote Model Servers

If your model servers are on a different machine:

```bash
# .env
SMALL_MODEL_URL=http://192.168.1.100:8368/v1
LARGE_MODEL_URL=http://192.168.1.100:8369/v1
VLLM_API_KEY=your-secure-api-key
```

### Docker Deployment

For Docker, the `.env` file is automatically loaded by docker-compose:

```bash
# .env
SMALL_MODEL_URL=http://130.113.68.24:8368/v1
LARGE_MODEL_URL=http://130.113.68.24:8369/v1
API_URL=http://api:8370/v1  # Use service name for inter-container communication
```

Then run:
```bash
cd deployment/docker
docker-compose up -d
```

### Different Clarification Strategies

To test different clarification strategies:

```bash
# .env
CLARIFICATION_STRATEGY=at_cot  # Use chain-of-thought reasoning
LOG_LEVEL=DEBUG                # Enable debug logging to see prompts
```

## How It Works

1. **Automatic Loading**: The `core/clari_gen/config.py` module automatically loads the `.env` file when imported.

2. **Fallback to Defaults**: If a variable is not set in `.env`, the system uses the default value defined in `config.py`.

3. **Override Priority**: 
   - Environment variables set in your shell take highest priority
   - Then `.env` file values
   - Finally, default values in code

4. **Docker Integration**: Docker Compose reads the `.env` file and passes variables to containers.

## Security Notes

⚠️ **Important:**
- The `.env` file is in `.gitignore` and will NOT be committed to version control
- Never commit sensitive API keys or credentials
- Use `.env.example` as a template for sharing configuration structure
- For production, consider using proper secret management (e.g., Docker secrets, Kubernetes secrets)

## Troubleshooting

### Variables Not Loading

If your environment variables aren't being picked up:

1. **Check file location**: The `.env` file should be in the project root (`/u40/chanc187/source/clari_gen/.env`)

2. **Check file format**: Each line should be `KEY=value` with no spaces around `=`

3. **Restart your application**: Changes to `.env` require restarting the application

4. **Check for typos**: Variable names are case-sensitive

### Docker Issues

If Docker isn't picking up your `.env` file:

1. **Check path in docker-compose.yml**: Should be `../../.env` relative to the docker-compose file

2. **Rebuild containers**: 
   ```bash
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

3. **Verify variables inside container**:
   ```bash
   docker exec clari_gen_api env | grep MODEL
   ```

## Example Configurations

### Development Setup
```bash
# .env - Local development
SMALL_MODEL_URL=http://localhost:8368/v1
LARGE_MODEL_URL=http://localhost:8369/v1
API_URL=http://localhost:8370/v1
LOG_LEVEL=DEBUG
CLARIFICATION_STRATEGY=at_standard
```

### Production Setup
```bash
# .env - Production
SMALL_MODEL_URL=https://models.example.com:8368/v1
LARGE_MODEL_URL=https://models.example.com:8369/v1
API_URL=https://api.example.com/v1
VLLM_API_KEY=prod-secure-key-here
LOG_LEVEL=WARNING
LOG_FILE=/var/log/clari_gen/app.log
MAX_CLARIFICATION_ATTEMPTS=5
```

### Testing Setup
```bash
# .env - Testing
SMALL_MODEL_URL=http://test-server:8368/v1
LARGE_MODEL_URL=http://test-server:8369/v1
LOG_LEVEL=ERROR
SKIP_INTEGRATION_TESTS=false
```

# Docker Setup Guide

## Problem: Missing Data in Docker vs Local Execution

### Symptoms
When running the trading LLM system:
- **Local execution** (`python -m trading_llm.main --pretty`): ✅ Works perfectly - returns ticker data, news, and AI analysis
- **Docker execution** (via FastAPI at port 8080): ❌ Returns empty news and zero ticker data, but AI responses work

### Root Cause

The issue has **two parts**:

#### Part 1: OpenBB Extension Build System

OpenBB Platform uses a modular architecture where extensions (like `equity`, `news`, etc.) must be "built" after installation:

1. When you `pip install openbb`, it installs the core and extension packages
2. **But** OpenBB needs to run a build step to generate static assets that map extensions to the API
3. This build creates files like `/usr/local/lib/python3.13/site-packages/openbb_core/app/static/package/__extensions__.py`
4. Without this build, `obb.equity` and `obb.news` attributes don't exist → "Failed to import extensions"

**In Docker**: The build must happen AFTER copying application code to avoid conflicts.

#### Part 2: OpenBB Credentials (Secondary Issue)

OpenBB Platform also stores API credentials in `~/.openbb_platform/` directory:
- `user_settings.json` - API keys for data providers (FMP, Tiingo, etc.)
- `system_settings.json` - System configuration

**Note**: For basic functionality with yfinance provider, credentials are optional. However, for premium providers and higher rate limits, credentials should be mounted as a volume.

## Solution

### 1. Update Dockerfile (CRITICAL FIX)

The Dockerfile now includes the OpenBB build step **AFTER** copying application code:

```dockerfile
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build OpenBB extensions AFTER copying application code
# This generates the static assets needed for OpenBB to discover installed extensions
RUN python -c "from openbb_core.app.static.package_builder import PackageBuilder; PackageBuilder().build()" \
    && find /usr/local/lib/python3.13/site-packages/openbb* -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Create OpenBB platform directory for credentials (optional, for premium providers)
RUN mkdir -p /root/.openbb_platform
```

**Why this order matters**:
- Building BEFORE `COPY . .` doesn't work because the build might get overwritten
- Building AFTER ensures extensions are properly registered
- Clearing `__pycache__` prevents stale import caches

### 2. Mount OpenBB Credentials as Volume (Optional)

For premium data providers, mount your local OpenBB credentials:

```bash
docker run --rm -p 8080:8080 \
    --env-file .env \
    -e PORT=8080 \
    -v ~/.openbb_platform:/root/.openbb_platform:ro \
    ritamk/brok
```

**Note**: The volume mount is optional. Basic functionality works with yfinance provider without credentials.

### 3. Use Updated Scripts

The `scripts.sh` file has been updated:

```bash
# Rebuild with the fix
./scripts.sh build_docker_image

# Run the container
./scripts.sh run_docker_image
```

## For Production Deployment (Google Cloud Run)

### Quick Deploy

The Dockerfile fix is already in place. Simply deploy the updated image:

```bash
# Using the deployment script
./scripts.sh deploy_cloud_run

# Or manually
gcloud builds submit --tag gcr.io/PROJECT_ID/brok-api
gcloud run deploy brok-api --image gcr.io/PROJECT_ID/brok-api --region asia-south2
```

**Important**: The OpenBB extension build step now runs automatically during the Docker build, so no additional configuration is needed for basic functionality.

### Environment Variables

Set these in Cloud Run for full functionality:

**Required:**
- `OPENAI_API_KEY` - Your OpenAI API key

**Optional (with defaults):**
- `NEWS_ENABLED=1`
- `NEWS_LIMIT=10`
- `TIMEFRAMES=1D,15m`

### OpenBB Credentials (Optional)

For premium data providers, you have several options:

### Option 1: Environment Variables (Recommended)

Set OpenBB credentials as environment variables and configure OpenBB programmatically:

```python
# In your startup code or config
from openbb import obb

# Configure credentials from environment
obb.user.credentials.fmp_api_key = os.getenv("FMP_API_KEY")
obb.user.credentials.tiingo_api_key = os.getenv("TIINGO_API_KEY")
# ... other providers
```

### Option 2: Build Credentials into Image (Less Secure)

Copy a sanitized `user_settings.json` into the Docker image:

```dockerfile
# In Dockerfile
COPY .openbb_platform/user_settings.json /root/.openbb_platform/user_settings.json
```

**⚠️ Warning**: Never commit actual credentials to version control. Use build secrets or CI/CD variables.

### Option 3: Secret Management

Use cloud provider secret management:
- **Google Cloud**: Secret Manager
- **AWS**: Secrets Manager
- **Azure**: Key Vault

Mount secrets as files or inject as environment variables at runtime.

## Verification

After applying the fix, verify the setup works:

1. **Build the image**:
   ```bash
   ./scripts.sh build_docker_image
   ```

2. **Run with volume mount**:
   ```bash
   ./scripts.sh run_docker_image
   ```

3. **Test the API**:
   ```bash
   curl -X POST http://localhost:8080/v1/tickers/data \
     -H "Content-Type: application/json" \
     -d '{"tickers": ["RELIANCE.NS"], "timeframes": ["1D"]}'
   ```

4. **Verify the response includes**:
   - ✅ Non-zero price data
   - ✅ News headlines
   - ✅ Technical analysis
   - ✅ AI-generated trade recommendations

## Troubleshooting

### Still Getting Empty Data?

1. **Check OpenBB credentials exist locally**:
   ```bash
   ls -la ~/.openbb_platform/
   cat ~/.openbb_platform/user_settings.json
   ```

2. **Verify volume mount in container**:
   ```bash
   docker run --rm -it \
     -v ~/.openbb_platform:/root/.openbb_platform:ro \
     ritamk/brok \
     bash -c "ls -la /root/.openbb_platform/"
   ```

3. **Check OpenBB can access credentials in container**:
   ```bash
   docker run --rm -it \
     --env-file .env \
     -v ~/.openbb_platform:/root/.openbb_platform:ro \
     ritamk/brok \
     python -c "from openbb import obb; print(obb.user.credentials)"
   ```

### Provider-Specific Issues

If specific providers fail:
- Check API key validity in `~/.openbb_platform/user_settings.json`
- Verify provider rate limits haven't been exceeded
- Test provider directly: `obb.news.company(symbol='AAPL', provider='fmp')`

## Summary

The key difference between local and Docker execution was the missing OpenBB credentials directory. By mounting `~/.openbb_platform` as a volume, the Docker container gains access to the same credentials as local execution, ensuring consistent behavior across both environments.


# syntax=docker/dockerfile:1.7

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
       ca-certificates \
       tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build OpenBB extensions AFTER copying application code
# This generates the static assets needed for OpenBB to discover installed extensions
# Must be done after COPY to ensure no file conflicts
RUN python -c "from openbb_core.app.static.package_builder import PackageBuilder; print('Building OpenBB extensions...'); pb = PackageBuilder(); pb.build(); print('OpenBB extensions built successfully')" && \
    find /usr/local/lib/python3.13/site-packages/openbb* -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    echo "Verifying build..." && \
    python -c "from openbb import obb; assert hasattr(obb, 'equity'), 'equity extension not found'; assert hasattr(obb, 'news'), 'news extension not found'; print('✓ OpenBB extensions verified')"

# Create OpenBB platform directory for credentials
# This directory should be mounted as a volume at runtime with the host's ~/.openbb_platform
RUN mkdir -p /root/.openbb_platform

ENV PORT=8080

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
# Multi-stage build for fks_training Python service
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies (scipy, numpy, etc. need fortran and lapack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel (better caching with BuildKit)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel

# Copy dependency files (for better layer caching)
COPY requirements.txt requirements.dev.txt* ./

# Install numpy first as a build dependency for scipy and other packages
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --user --no-warn-script-location numpy

# Install Python dependencies with BuildKit cache mount
# Only install requirements.txt, dev dependencies are optional
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --user --no-warn-script-location -r requirements.txt

# Runtime stage
FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app \
    SERVICE_NAME=training \
    SERVICE_TYPE=training \
    SERVICE_PORT=8005 \
    TRAINING_SERVICE_PORT=8005 \
    PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user first (before copying files)
RUN useradd -u 1000 -m -s /bin/bash appuser

# Copy Python packages from builder with correct ownership
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application source with correct ownership
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser entrypoint.sh* ./

# Make entrypoint executable if it exists
RUN if [ -f entrypoint.sh ]; then chmod +x entrypoint.sh; fi

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request,sys;port=os.getenv('SERVICE_PORT','8005');u=f'http://localhost:{port}/health';\
import urllib.error;\
try: urllib.request.urlopen(u,timeout=3);\
except Exception: sys.exit(1)" || exit 1

# Expose the service port
EXPOSE 8005

# Use entrypoint script if available, otherwise run directly
CMD if [ -f entrypoint.sh ]; then ./entrypoint.sh; else python src/main.py; fi

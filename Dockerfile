# Multi-stage build for fks_training Python service
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt requirements.dev.txt* ./

# Install Python dependencies
RUN python -m pip install --upgrade pip wheel setuptools && \
    python -m pip install --user --no-cache-dir -r requirements.txt || true

# Runtime stage
FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
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

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application source
COPY src/ ./src/
COPY entrypoint.sh* ./

# Make entrypoint executable if it exists
RUN if [ -f entrypoint.sh ]; then chmod +x entrypoint.sh; fi

# Create non-root user
RUN useradd -u 1000 -m appuser && \
    chown -R appuser:appuser /app /home/appuser/.local
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

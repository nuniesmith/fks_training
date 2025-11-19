# Optimized Dockerfile for fks_training - Uses base image directly to reduce size
# Uses GPU base image with all dependencies pre-installed
FROM nuniesmith/fks:docker-gpu

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app \
    SERVICE_NAME=training \
    SERVICE_TYPE=training \
    SERVICE_PORT=8011 \
    TRAINING_SERVICE_PORT=8011 \
    PATH=/usr/local/bin:$PATH

# Install only service-specific packages
# Base image already has: PyTorch, Transformers, MLflow, WandB, langchain, chromadb, sentence-transformers, TA-Lib, numpy, pandas
COPY requirements.txt requirements.dev.txt* ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip cache purge || true \
    && rm -rf /root/.cache/pip/* /tmp/pip-* 2>/dev/null || true

# Create non-root user if it doesn't exist
RUN id -u appuser 2>/dev/null || useradd -u 1000 -m -s /bin/bash appuser

# Copy application source
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser entrypoint.sh* ./

# Make entrypoint executable if it exists
RUN if [ -f entrypoint.sh ]; then chmod +x entrypoint.sh; fi

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request,sys;port=os.getenv('SERVICE_PORT','8011');u=f'http://localhost:{port}/health';\
import urllib.error;\
try: urllib.request.urlopen(u,timeout=3);\
except Exception: sys.exit(1)" || exit 1

# Expose the service port
EXPOSE 8011

# Use entrypoint script if available, otherwise run directly
CMD if [ -f entrypoint.sh ]; then ./entrypoint.sh; else python src/main.py; fi

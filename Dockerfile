FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .
WORKDIR /app/src

# Create non-root user
RUN useradd -r -u 1088 appuser
USER appuser

ENV SERVICE_NAME=fks-training \
    SERVICE_TYPE=training \
    SERVICE_PORT=8005 \
    TRAINING_SERVICE_PORT=8005

EXPOSE 8005

# Use explicit path to main to avoid module path ambiguity
CMD ["python", "src/main.py"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

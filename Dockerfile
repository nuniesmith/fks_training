FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt requirements.dev.txt* ./
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip wheel setuptools && \
    (pip install -r requirements.txt || true)

COPY . /app/

ENV PYTHONPATH=/app/src \
    SERVICE_NAME=training \
    SERVICE_TYPE=training \
    SERVICE_PORT=8005 \
    TRAINING_SERVICE_PORT=8005

EXPOSE 8005

RUN adduser --disabled-password --gecos "" appuser || useradd -m appuser || true
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

CMD ["python", "src/main.py"]

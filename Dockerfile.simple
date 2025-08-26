## Simplified Python Service Dockerfile
## Use this for standard CPU Python services (FastAPI, Celery, Workers, etc.)
## For advanced / GPU / hybrid builds, use the unified master Dockerfile at shared/shared_docker/Dockerfile

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    APP_HOME=/app \
    PATH="/opt/venv/bin:$PATH"

WORKDIR ${APP_HOME}

# System deps kept minimal; extend via BUILD_PACKAGES if needed
ARG BUILD_PACKAGES="build-essential gcc libpq-dev"
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $BUILD_PACKAGES curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Dependency layer (copy only requirement manifests for caching)
COPY requirements*.txt ./
COPY pyproject.toml poetry.lock* ./ 2>/dev/null || true

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip wheel setuptools \
 && if [ -f requirements.txt ]; then /opt/venv/bin/pip install -r requirements.txt; fi \
 && if [ -f requirements.prod.txt ]; then /opt/venv/bin/pip install -r requirements.prod.txt; fi

# Copy application source
COPY src ./src

# Optional: copy migrations / scripts if present
COPY migrations ./migrations 2>/dev/null || true
COPY scripts ./scripts 2>/dev/null || true

ENV PYTHONPATH=${APP_HOME}/src:${APP_HOME}

EXPOSE 8000

# Smart default entrypoint: try to detect FastAPI/Uvicorn app or fallback to module
ENV APP_MODULE="${APP_MODULE:-app.main:app}" \
    APP_COMMAND="${APP_COMMAND:-}" \
    SERVICE_PORT=8000

ENTRYPOINT ["/bin/bash","-c"]
CMD ["if [ -n \"$APP_COMMAND\" ]; then exec $APP_COMMAND; fi; \
if command -v uvicorn >/dev/null 2>&1; then \
  TARGET=${APP_MODULE}; \
  if [ -z \"$TARGET\" ]; then \
    CAND=$(grep -R --include=*.py -l 'FastAPI(' src 2>/dev/null | head -1 | sed 's#^src/##; s#/__init__\.py$##; s#/#.#g; s#\.py$##'); \
    [ -n \"$CAND\" ] && TARGET="$CAND:app"; \
  fi; \
  [ -n \"$TARGET\" ] && exec uvicorn $TARGET --host 0.0.0.0 --port ${SERVICE_PORT}; \
fi; echo 'No uvicorn target detected; launching shell'; exec bash"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import socket,os; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', int(os.environ.get('SERVICE_PORT',8000)))); s.close()" || exit 1

## To extend:
##   docker build -f shared/shared_docker/templates/python.Dockerfile -t myimage .
##   docker run -e APP_COMMAND="python -m tasks.worker" myimage

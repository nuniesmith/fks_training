#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[training] Stopping existing containers..."
docker compose down

echo "[training] Rebuilding images..."
docker compose build

echo "[training] Starting containers in detached mode..."
docker compose up -d

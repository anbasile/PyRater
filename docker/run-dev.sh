#!/usr/bin/env bash
# Run a command inside the minimal pyrater-dev podman container, with the repo
# bind-mounted and a persistent named volume for the uv cache (so `uv sync`
# doesn't re-download wheels on every invocation).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
podman run --rm \
  -v "$(pwd)":/app:z \
  -v pyrater-uv-cache:/uv-cache:z \
  -v pyrater-venv:/venv:z \
  -e UV_PROJECT_ENVIRONMENT=/venv \
  -w /app \
  pyrater-dev "$@"

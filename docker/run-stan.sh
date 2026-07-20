#!/usr/bin/env bash
# Run a command inside the CmdStan-capable podman container: same isolation
# as run-dev.sh (repo bind-mounted, venv/uv-cache in named volumes), plus a
# named volume for ~/.cmdstan so CmdStan itself is compiled once and reused
# across runs instead of rebuilt from scratch every time.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
podman run --rm \
  -v "$(pwd)":/app:z \
  -v pyrater-uv-cache:/uv-cache:z \
  -v pyrater-venv:/venv:z \
  -v pyrater-cmdstan:/root/.cmdstan:z \
  -v pyrater-stan-cache:/root/.cache/pyrater:z \
  -e UV_PROJECT_ENVIRONMENT=/venv \
  -w /app \
  pyrater-stan-test "$@"

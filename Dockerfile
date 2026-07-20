# Multi-stage build: the C++ toolchain needed to compile CmdStan and the six
# bundled Stan models never makes it into the final image, only the compiled
# results do -- the whole point of shipping this image is to avoid asking
# users to install a C++ toolchain themselves (see the paper's own stated
# motivation for a Docker image).
FROM python:3.12-slim AS builder

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends g++ make \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG CMDSTAN_VERSION=2.39.0

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    CMDSTAN=/opt/cmdstan/cmdstan-${CMDSTAN_VERSION} \
    XDG_CACHE_HOME=/opt/cache

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# --no-editable: install pyrater as a real package inside /opt/venv rather
# than an editable link back to /app/src, since the final stage below only
# copies /opt/venv (not /app).
RUN uv sync --frozen --no-dev --no-editable

# `install_cmdstan --dir X` unpacks into `X/cmdstan-<version>/`, not `X`
# itself -- CMDSTAN above points at that versioned subdirectory to match.
RUN /opt/venv/bin/python -m cmdstanpy.install_cmdstan \
    --dir /opt/cmdstan --version ${CMDSTAN_VERSION}

# Pre-compile every bundled model so the runtime image starts ready to
# sample immediately, with no first-run compile latency.
RUN /opt/venv/bin/python -c "\
import pathlib; \
from pyrater.stan import get_compiled_model; \
[get_compiled_model(p.name) for p in sorted(pathlib.Path('src/pyrater/stan').glob('*.stan'))]"

# ---------------------------------------------------------------------------

FROM python:3.12-slim

ARG CMDSTAN_VERSION=2.39.0

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/cmdstan /opt/cmdstan
COPY --from=builder /opt/cache /opt/cache

ENV PATH=/opt/venv/bin:$PATH \
    CMDSTAN=/opt/cmdstan/cmdstan-${CMDSTAN_VERSION} \
    XDG_CACHE_HOME=/opt/cache

ENTRYPOINT ["pyrater"]
CMD ["--help"]

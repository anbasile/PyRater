# Installation

```bash
pip install pyrater
```

CmdStan (the C++ sampler PyRater fits models with) is *not* installed by
`pip install pyrater` — it's a large, compiled toolchain most users only
need once. The first time you call `.fit()` on a Stan-backed model without
CmdStan installed, PyRater raises a `CmdStanNotFoundError` telling you
exactly what to run:

```bash
python -m cmdstanpy.install_cmdstan
```

## Optional extras

```bash
pip install "pyrater[docs]"      # build these docs locally
pip install "pyrater[examples]"  # sentence-transformers/datasets, for examples/
```

## Docker

A prebuilt image ships CmdStan and all six bundled models precompiled, so
there's no first-run compile latency and no C++ toolchain to install:

```bash
docker build -t pyrater .
docker run --rm pyrater --list-models
```

## Development

This repository's own development workflow runs entirely inside podman —
see the top-level `README.md` for `docker/run-dev.sh` and `docker/run-stan.sh`.

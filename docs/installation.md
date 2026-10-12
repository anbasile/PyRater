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
pip install "pyrater[jax]"       # JAX and NumPyro, for the np_* models
pip install "pyrater[docs]"      # build these docs locally
pip install "pyrater[examples]"  # sentence-transformers/datasets, for examples/
```

The `jax` extra is what the [NumPyro-backed models](models/numpyro.md)
(`np_d&s`, `np_mace`, `np_hier_d&s`, `np_nested`) run on. They need no
CmdStan. Everything else in PyRater works without the extra: those four
models are registered either way, and fitting one without it raises an
`ImportError` that names the extra. The extra installs the CPU build of
JAX; for a GPU, install the JAX build for your hardware first, following
[JAX's own instructions](https://docs.jax.dev/en/latest/installation.html).

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

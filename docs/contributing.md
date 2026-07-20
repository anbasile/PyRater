# Contributing

All development commands run inside podman — nothing is installed on the
host. See the top-level `README.md` and `docker/run-dev.sh` /
`docker/run-stan.sh`.

```bash
./docker/run-dev.sh uv sync --extra dev
./docker/run-dev.sh uv run pytest -m "not stan"     # fast unit suite
./docker/run-stan.sh uv run pytest -m stan          # Stan integration suite
./docker/run-dev.sh uvx ruff format .
./docker/run-dev.sh uvx ruff check .
./docker/run-dev.sh uvx ty check --python /venv src tests scripts
```

## Conventions

- Tests before code: every module was built test-first.
- 100% type-checked with `ty`; every new module should stay that way.
- Unit tests must never require CmdStan; anything that compiles/samples a
  real Stan model belongs under `tests/integration/`, marked
  `@pytest.mark.stan`.
- The bundled `.stan` files under `src/pyrater/stan/` are **not** original
  to this library or to the Basile, Franco-Salvador & Rosso (2024) paper
  PyRater implements — they're copies of Paun, Carpenter, Chamberlain,
  Hovy, Kruschwitz & Poesio (2018), "Comparing Bayesian Models of
  Annotation" (TACL, CC-BY 4.0)'s supplementary material, migrated to
  modern Stan array syntax (`array[N] int y` instead of `int y[N]`) — see
  the note at the top of each file. `LRE-NC.stan` has one additional,
  purely-additive `generated quantities` block beyond that; see
  [Logistic Random Effects](models/logistic_random_effects.md).

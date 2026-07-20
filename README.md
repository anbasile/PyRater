# PyRater

A Python toolkit for annotation analysis with Bayesian models of annotation:
given a dataset labeled by multiple annotators, PyRater estimates the most
likely true label per item (Q1), how difficult each item was to annotate
(Q2), and how reliable each annotator is (Q3).

Six models are included — Dawid & Skene (`d&s`), Hierarchical Dawid & Skene
(`hier_d&s`), MACE (`mace`), a pooled Multinomial baseline (`multinomial`),
Item Difficulty (`item_difficulty`), and Logistic Random Effects (`lre`) —
plus two non-Bayesian baselines (`majority_voting`,
`kappa_majority_voting`), all behind one consistent API and fit via
[Stan](https://mc-stan.org/)/[CmdStanPy](https://mc-stan.org/cmdstanpy/).

```python
import pyrater as pyr

data = pyr.load_dataset("synthetic_small")
model = pyr.PyRaterModel.get("d&s")
model.fit(data)

model.labels           # Q1: predicted true label per item
model.difficulty        # Q2: how hard each item was to annotate
model.credibilities      # Q3: how trustworthy each annotator was
```

```bash
pyrater -m d&s annotations.csv -o results/
```

See `docs/` for the full guide (`docs/quickstart.md`,
`docs/models/overview.md` for choosing a model, `docs/user_guide/` for data
loading, diagnostics, plotting, and the CLI).

## Installation

```bash
pip install pyrater
python -m cmdstanpy.install_cmdstan   # once, to install the Stan sampler
```

Or use the Docker image, which ships CmdStan and all six models
precompiled:

```bash
docker build -t pyrater .
docker run --rm pyrater --list-models
```

## Development

All commands run inside podman — nothing is installed on the host. Two
images are used: a minimal one for fast unit tests/lint/type-checking, and
a heavier one (with a C++ toolchain) for the Stan integration suite:

```bash
podman build -t pyrater-dev -f docker/Dockerfile.dev .
podman build -t pyrater-stan-test -f docker/Dockerfile.stan-test .

./docker/run-dev.sh uv sync --extra dev
./docker/run-dev.sh uv run pytest -m "not stan"     # fast unit suite (~2s)
./docker/run-dev.sh uvx ruff format .
./docker/run-dev.sh uvx ruff check .
./docker/run-dev.sh uvx ty check --python /venv src tests scripts examples

./docker/run-stan.sh uv run python -m cmdstanpy.install_cmdstan  # once
./docker/run-stan.sh uv run pytest -m stan          # Stan integration suite
```

Dependencies, the virtualenv, and CmdStan itself live only in podman-managed
named volumes (`pyrater-uv-cache`, `pyrater-venv`, `pyrater-cmdstan`) —
nothing is installed on the host beyond the repo's own source files.
`uvx ty` needs `--python /venv` to see the project's installed dependencies
(it otherwise runs in an isolated tool environment with no access to
numpy/pandas/etc.). The `/app` bind mount and shared named volumes use `:z`
(shared SELinux labeling) rather than `:Z` (private/exclusive), since both
containers may run concurrently against the same paths.

Built test-first throughout: every module has unit tests written before its
implementation, 100% type-checked with `ty`, and the six bundled Stan
models are additionally verified end-to-end (compile + sample) in the
`-m stan` integration suite.

## Credits

PyRater implements the toolkit described in Basile, Franco-Salvador &
Rosso (2024), ["PyRater: A Python Toolkit for Annotation
Analysis"](https://aclanthology.org/2024.lrec-main.1169/) (LREC-COLING
2024).

The six bundled Stan models (`d&s`, `hier_d&s`, `mace`, `multinomial`,
`item_difficulty`, `lre`) are **not original to this library or to the
paper above** — their implementations are taken from the supplementary
material of Paun, Carpenter, Chamberlain, Hovy, Kruschwitz & Poesio
(2018), ["Comparing Bayesian Models of
Annotation"](https://aclanthology.org/Q18-1040/) (TACL, CC-BY 4.0), only
migrated to modern Stan array syntax. See
[Contributing](docs/contributing.md) for details on what changed and what
didn't.

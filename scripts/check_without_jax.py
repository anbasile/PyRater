"""Check that pyrater works when the optional `jax` extra is not installed.

Run in an environment built without the extra (`uv sync`, no `--extra`); CI's
`without-jax-extra` job does. Exits non-zero, naming each failed check, if JAX
or NumPyro turn out to be installed, or to be needed for anything other than
fitting a NumPyro-backed model.
"""

from __future__ import annotations

import importlib.util
import sys

# Importing pyrater without the extra is itself the first check.
import pyrater

NUMPYRO_MODELS = ["np_d&s", "np_mace", "np_hier_d&s", "np_nested"]
BACKEND = ("jax", "numpyro")


def failures() -> list[str]:
    found: list[str] = []

    installed = [name for name in BACKEND if importlib.util.find_spec(name)]
    if installed:
        return [f"{installed} installed; this check needs an environment without them."]

    available = pyrater.PyRaterModel.list_available()
    missing = [name for name in NUMPYRO_MODELS if name not in available]
    if missing:
        found.append(f"not registered without the extra: {missing}")

    # A Stan-free code path, end to end.
    data = pyrater.load_dataset("synthetic_small")
    model = pyrater.PyRaterModel.get("majority_voting").fit(data)
    if model.labels.shape != (data.n_items,) or model.credibilities.shape != (data.n_annotators,):
        found.append("majority_voting did not produce labels and credibilities.")

    # Fitting a NumPyro-backed model is the one thing that needs the extra,
    # and it has to say so.
    try:
        pyrater.PyRaterModel.get("np_d&s").fit(data, method="map")
    except ImportError as error:
        if "pyrater[jax]" not in str(error):
            found.append(f"the ImportError does not name the extra: {error}")
    else:
        found.append("fitting np_d&s without JAX did not raise ImportError.")

    leaked = [name for name in BACKEND if name in sys.modules]
    if leaked:
        found.append(f"imported although not installed: {leaked}")
    return found


if __name__ == "__main__":
    problems = failures()
    for problem in problems:
        print(f"FAIL: {problem}")
    if not problems:
        n_models = len(pyrater.PyRaterModel.list_available())
        print(f"OK: pyrater {pyrater.__version__} without the jax extra; {n_models} models.")
    sys.exit(1 if problems else 0)

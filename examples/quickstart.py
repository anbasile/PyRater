"""A minimal end-to-end PyRater workflow.

Requires CmdStan (`python -m cmdstanpy.install_cmdstan`, once).
Run with: uv run python examples/quickstart.py
"""

from __future__ import annotations

import pyrater as pyr


def main() -> None:
    data = pyr.load_dataset("synthetic_small")
    print(data)

    model = pyr.PyRaterModel.get("d&s")
    model.fit(data)

    print("labels:", model.labels)
    print("difficulty (hardest 3):", model.difficulty.argsort()[-3:])
    print("credibilities:", model.credibilities)
    print("converged:", model.convergence_.ok if model.convergence_ else "not checked")


if __name__ == "__main__":
    main()

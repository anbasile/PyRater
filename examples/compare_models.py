"""Compare every registered model's labels against the majority-voting
baseline on the bundled synthetic dataset.

Requires CmdStan. Run with: uv run python examples/compare_models.py
"""

from __future__ import annotations

import pyrater as pyr


def main() -> None:
    data = pyr.load_dataset("synthetic_small")
    baseline = pyr.PyRaterModel.get("majority_voting").fit(data)

    for name in pyr.PyRaterModel.list_available():
        if name in {"majority_voting", "kappa_majority_voting"}:
            continue
        model = pyr.PyRaterModel.get(name).fit(data, check_rhat=False)
        agreement = (model.labels == baseline.labels).mean()
        print(f"{name:20s} agrees with majority voting on {agreement:.1%} of items")


if __name__ == "__main__":
    main()

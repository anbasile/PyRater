"""Convergence checking via rank-normalized split-R-hat.

Wraps :func:`arviz.rhat` (Vehtari et al. 2021's rank-normalized split-R-hat,
the current best-practice diagnostic) rather than the older R-hat computed
by rstan's classic ``monitor()``. Convergence is reported, never enforced:
a bad R-hat means "trust these results less," not "these results don't
exist" — callers decide what to do with a non-``ok`` report.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import arviz as az
import numpy as np


@dataclasses.dataclass(frozen=True)
class ConvergenceReport:
    """The result of a convergence check."""

    rhat: dict[str, np.ndarray]
    threshold: float
    n_above_threshold: dict[str, int]

    @property
    def ok(self) -> bool:
        return all(count == 0 for count in self.n_above_threshold.values())

    def summary(self) -> str:
        lines = []
        for name, above in self.n_above_threshold.items():
            total = self.rhat[name].size
            lines.append(f"{name} - {above}/{total} parameters above R-hat {self.threshold}")
        return "\n".join(lines)


def check_convergence(
    idata: az.InferenceData,
    *,
    var_names: Sequence[str] | None = None,
    threshold: float = 1.01,
    method: str = "rank",
) -> ConvergenceReport:
    """Compute rank-normalized split-R-hat for each parameter and compare it
    against `threshold`, per individual parameter entry (e.g. each of a
    ``simplex[K]`` array's ``K`` components counts separately)."""
    rhat_dataset = az.rhat(
        idata, var_names=list(var_names) if var_names is not None else None, method=method
    )

    rhat: dict[str, np.ndarray] = {}
    n_above_threshold: dict[str, int] = {}
    for name in rhat_dataset.data_vars:
        values = rhat_dataset[str(name)].to_numpy()
        rhat[str(name)] = values
        n_above_threshold[str(name)] = int(np.sum(values >= threshold))

    return ConvergenceReport(rhat=rhat, threshold=threshold, n_above_threshold=n_above_threshold)

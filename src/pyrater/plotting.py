"""Plotting utilities: a Figure-1-style jittered credibility strip, and
generic histograms for a fitted model's `difficulty`/`credibilities`.

Both functions return a plain `(Figure, Axes)` pair -- callers decide
whether to save, show, or keep customizing; nothing here ever calls
`plt.show()`.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyrater.models.base import PyRaterModel


def credibility_plot(
    model: PyRaterModel,
    *,
    label_fn: Callable[[int], str] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[Figure, Axes]:
    """A jittered strip plot of annotator credibility, one point per
    annotator, in the style of the paper's Figure 1.

    The best and worst annotators are highlighted; if `label_fn` is given,
    they're also annotated with its return value for that annotator's
    internal index (e.g. their name, or -- in the zero-shot prompt-ranking
    case study -- the prompt text itself).
    """
    credibilities = model.credibilities  # raises NotFittedError if not fitted
    if rng is None:
        rng = np.random.default_rng()

    n = len(credibilities)
    x = rng.uniform(-0.05, 0.05, size=n)

    best = int(np.argmax(credibilities))
    worst = int(np.argmin(credibilities))
    colors = np.full(n, "tab:gray", dtype=object)
    colors[best] = "tab:green"
    colors[worst] = "tab:red"

    fig, ax = plt.subplots()
    ax.scatter(x, credibilities, c=list(colors), alpha=0.8)

    if label_fn is not None:
        for idx in (best, worst):
            ax.annotate(label_fn(idx), (x[idx], credibilities[idx]))

    ax.set_ylabel("Credibility")
    ax.set_xticks([])
    return fig, ax


_QUANTITIES: dict[str, Callable[[PyRaterModel], npt.NDArray[np.float64]]] = {
    "difficulty": lambda m: m.difficulty,
    "credibilities": lambda m: m.credibilities,
}


def posterior_plot(model: PyRaterModel, quantity: str) -> tuple[Figure, Axes]:
    """A histogram of a fitted model's `difficulty` or `credibilities`."""
    try:
        getter = _QUANTITIES[quantity]
    except KeyError:
        raise ValueError(
            f"Unknown quantity {quantity!r}. Expected one of {list(_QUANTITIES)}."
        ) from None
    values = getter(model)  # raises NotFittedError if not fitted

    fig, ax = plt.subplots()
    ax.hist(values, bins="auto")
    ax.set_xlabel(quantity)
    ax.set_ylabel("count")
    return fig, ax

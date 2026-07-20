"""Shared pytest fixtures and test-data builders for the pyrater test suite."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded RNG, for deterministic randomized tests."""
    return np.random.default_rng(0)


def one_hot_ish(
    labels: npt.NDArray[np.int64], n_classes: int, n_draws: int, *, high: float = 0.9
) -> npt.NDArray[np.float64]:
    """Build a ``(draw, item, class)`` posterior array concentrated on `labels`.

    Not a fixture (it's parametrized per call site) — a plain builder shared
    across test modules that need a fake, deterministic posterior.
    """
    low = (1 - high) / (n_classes - 1)
    base = np.full((len(labels), n_classes), low)
    base[np.arange(len(labels)), labels] = high
    return np.tile(base, (n_draws, 1, 1))


@pytest.fixture
def one_hot_posterior() -> Callable[..., npt.NDArray[np.float64]]:
    """Fixture form of :func:`one_hot_ish`, for tests that prefer dependency
    injection over a direct cross-module import."""
    return one_hot_ish


@pytest.fixture
def long_annotations_df() -> pd.DataFrame:
    """A small long-format annotation table, deliberately not alphabetically
    ordered, to exercise first-appearance ordering."""
    return pd.DataFrame(
        {
            "item_id": ["b", "b", "a", "a", "c", "c"],
            "annotator_id": ["w2", "w1", "w2", "w1", "w1", "w2"],
            "label": ["yes", "no", "yes", "yes", "no", "no"],
        }
    )

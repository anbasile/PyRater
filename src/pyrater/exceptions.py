"""Exceptions and warnings raised by pyrater."""

from __future__ import annotations


class NotFittedError(RuntimeError):
    """Raised when a fitted-only property is accessed before ``.fit()``."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"This '{model_name}' model has not been fit yet. Call `.fit(data)` first."
        )


class ModelCapabilityError(RuntimeError):
    """Raised when a requested quantity cannot be computed for the given model/data."""


class CmdStanNotFoundError(RuntimeError):
    """Raised when CmdStan is required but not installed."""

    def __init__(self) -> None:
        super().__init__(
            "CmdStan is not installed. Run this once:\n"
            "  python -m cmdstanpy.install_cmdstan\n"
            "(or use the pyrater Docker image, which ships it preinstalled)"
        )


class ConvergenceWarning(UserWarning):
    """Emitted when a fit may not have converged: an R-hat above the configured
    threshold, or, for a NumPyro-backed model fitted with ``method="map"``, a
    loss that was still moving at the last optimisation step."""

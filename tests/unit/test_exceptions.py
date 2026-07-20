import warnings

import pytest

from pyrater.exceptions import (
    CmdStanNotFoundError,
    ConvergenceWarning,
    ModelCapabilityError,
    NotFittedError,
)


def test_not_fitted_error_is_a_runtime_error() -> None:
    assert issubclass(NotFittedError, RuntimeError)


def test_not_fitted_error_message_mentions_fit() -> None:
    err = NotFittedError("d&s")
    assert "fit" in str(err).lower()
    assert "d&s" in str(err)


def test_model_capability_error_is_a_runtime_error() -> None:
    assert issubclass(ModelCapabilityError, RuntimeError)


def test_model_capability_error_message_is_informative() -> None:
    err = ModelCapabilityError("annotator identifiers are not available")
    assert "annotator identifiers are not available" in str(err)


def test_cmdstan_not_found_error_message_names_the_fix() -> None:
    err = CmdStanNotFoundError()
    assert "install_cmdstan" in str(err)


def test_convergence_warning_is_a_user_warning() -> None:
    assert issubclass(ConvergenceWarning, UserWarning)


def test_convergence_warning_is_filterable() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("R-hat above threshold", ConvergenceWarning, stacklevel=1)
    assert len(caught) == 1
    assert issubclass(caught[0].category, ConvergenceWarning)


def test_exceptions_are_distinct_types() -> None:
    with pytest.raises(NotFittedError):
        raise NotFittedError("d&s")
    with pytest.raises(ModelCapabilityError):
        raise ModelCapabilityError("no annotator ids")

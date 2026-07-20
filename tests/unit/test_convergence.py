import arviz as az
import numpy as np
import pytest

from pyrater.diagnostics.convergence import ConvergenceReport, check_convergence

N_DRAWS = 200


@pytest.fixture
def well_mixed_idata(rng: np.random.Generator) -> az.InferenceData:
    """Two chains sampling the same distribution: R-hat should be close to 1."""
    pi = rng.normal(loc=0.0, scale=1.0, size=(2, N_DRAWS, 3))
    return az.from_dict(posterior={"pi": pi})


@pytest.fixture
def poorly_mixed_idata(rng: np.random.Generator) -> az.InferenceData:
    """Two chains stuck at very different, non-overlapping locations: R-hat
    should be large."""
    chain0 = rng.normal(loc=0.0, scale=0.1, size=(N_DRAWS, 3))
    chain1 = rng.normal(loc=10.0, scale=0.1, size=(N_DRAWS, 3))
    pi = np.stack([chain0, chain1])
    return az.from_dict(posterior={"pi": pi})


class TestCheckConvergence:
    def test_well_mixed_chains_are_ok(self, well_mixed_idata: az.InferenceData) -> None:
        report = check_convergence(well_mixed_idata, threshold=1.2)
        assert report.ok
        assert report.n_above_threshold["pi"] == 0

    def test_poorly_mixed_chains_are_not_ok(self, poorly_mixed_idata: az.InferenceData) -> None:
        report = check_convergence(poorly_mixed_idata, threshold=1.01)
        assert not report.ok
        assert report.n_above_threshold["pi"] > 0

    def test_threshold_is_configurable(self, poorly_mixed_idata: az.InferenceData) -> None:
        lenient = check_convergence(poorly_mixed_idata, threshold=1000.0)
        assert lenient.ok

    def test_returns_a_convergence_report(self, well_mixed_idata: az.InferenceData) -> None:
        report = check_convergence(well_mixed_idata)
        assert isinstance(report, ConvergenceReport)
        assert report.threshold == pytest.approx(1.01)

    def test_var_names_filters_which_parameters_are_checked(
        self, well_mixed_idata: az.InferenceData
    ) -> None:
        report = check_convergence(well_mixed_idata, var_names=["pi"])
        assert set(report.n_above_threshold) == {"pi"}


class TestConvergenceReportSummary:
    def test_summary_mentions_variable_names_and_threshold(
        self, well_mixed_idata: az.InferenceData
    ) -> None:
        report = check_convergence(well_mixed_idata, threshold=1.2)
        text = report.summary()
        assert "pi" in text
        assert "1.2" in text

    def test_summary_reports_counts_out_of_total(
        self, poorly_mixed_idata: az.InferenceData
    ) -> None:
        report = check_convergence(poorly_mixed_idata, threshold=1.01)
        text = report.summary()
        assert "/3 parameters" in text

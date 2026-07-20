import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from pyrater.data.annotation_data import AnnotationData
from pyrater.exceptions import NotFittedError
from pyrater.models.baselines import MajorityVotingModel
from pyrater.plotting import credibility_plot, posterior_plot


@pytest.fixture
def fitted_model() -> MajorityVotingModel:
    df = pd.DataFrame(
        {
            "item_id": ["i0", "i0", "i0", "i1", "i1", "i1", "i2", "i2", "i2"],
            "annotator_id": ["a0", "a1", "a2"] * 3,
            "label": ["yes", "yes", "no", "no", "no", "yes", "yes", "no", "yes"],
        }
    )
    data = AnnotationData.from_long_df(df)
    return MajorityVotingModel().fit(data, check_rhat=False)


class TestCredibilityPlot:
    def test_returns_figure_and_axes(self, fitted_model: MajorityVotingModel) -> None:
        fig, ax = credibility_plot(fitted_model)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plots_one_point_per_annotator(self, fitted_model: MajorityVotingModel) -> None:
        _, ax = credibility_plot(fitted_model)
        (collection,) = ax.collections
        offsets = np.asarray(collection.get_offsets())
        assert len(offsets) == fitted_model.credibilities.shape[0]

    def test_y_values_match_credibilities(self, fitted_model: MajorityVotingModel) -> None:
        _, ax = credibility_plot(fitted_model)
        (collection,) = ax.collections
        offsets = np.asarray(collection.get_offsets())
        y_values = sorted(offsets[:, 1])
        assert y_values == pytest.approx(sorted(fitted_model.credibilities))

    def test_label_fn_annotates_best_and_worst(self, fitted_model: MajorityVotingModel) -> None:
        _, ax = credibility_plot(fitted_model, label_fn=lambda i: f"annotator #{i}")
        assert len(ax.texts) == 2  # best + worst

    def test_no_label_fn_means_no_annotations(self, fitted_model: MajorityVotingModel) -> None:
        _, ax = credibility_plot(fitted_model)
        assert len(ax.texts) == 0


class TestPosteriorPlot:
    def test_difficulty_returns_figure_and_axes(self, fitted_model: MajorityVotingModel) -> None:
        fig, ax = posterior_plot(fitted_model, "difficulty")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_credibilities_returns_figure_and_axes(self, fitted_model: MajorityVotingModel) -> None:
        fig, ax = posterior_plot(fitted_model, "credibilities")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_unknown_quantity_raises(self, fitted_model: MajorityVotingModel) -> None:
        with pytest.raises(ValueError, match="difficulty"):
            posterior_plot(fitted_model, "not-a-real-quantity")

    def test_histogram_uses_all_values(self, fitted_model: MajorityVotingModel) -> None:
        _, ax = posterior_plot(fitted_model, "difficulty")
        (container,) = ax.containers
        assert isinstance(container, BarContainer)
        assert len(container.patches) > 0
        total = sum(p.get_height() for p in container.patches)
        assert total == pytest.approx(len(fitted_model.difficulty))


def test_credibility_plot_requires_a_fitted_model() -> None:
    with pytest.raises(NotFittedError):
        credibility_plot(MajorityVotingModel())


def test_returned_arrays_are_not_accidentally_mutated(fitted_model: MajorityVotingModel) -> None:
    before = fitted_model.credibilities.copy()
    credibility_plot(fitted_model)
    np.testing.assert_array_equal(before, fitted_model.credibilities)

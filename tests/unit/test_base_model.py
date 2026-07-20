import warnings
from collections.abc import Callable
from typing import Any, ClassVar

import arviz as az
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from pyrater.data.annotation_data import AnnotationData
from pyrater.diagnostics.convergence import ConvergenceReport
from pyrater.exceptions import ConvergenceWarning, ModelCapabilityError, NotFittedError
from pyrater.models.base import PyRaterModel

PosteriorBuilder = Callable[..., npt.NDArray[np.float64]]


class FakeModel(PyRaterModel):
    """A minimal concrete model that skips Stan entirely: `_sample` just
    returns whatever InferenceData the test built by hand."""

    name = "fake"
    class_indexed_params: ClassVar[dict[str, int]] = {"q_z": 1}

    def __init__(self, idata_to_return: az.InferenceData) -> None:
        super().__init__()
        self._idata_to_return = idata_to_return
        self.sample_kwargs_received: dict[str, Any] = {}

    def _sample(
        self,
        data: AnnotationData,
        *,
        chains: int,
        warmup: int,
        iters: int,
        seed: int | None,
        **kwargs: Any,
    ) -> az.InferenceData:
        self.sample_kwargs_received = {
            "chains": chains,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
            **kwargs,
        }
        return self._idata_to_return


class FakeModelNoAlignment(FakeModel):
    """Mimics MACE: no class-indexed parameters, so no alignment is possible
    or necessary."""

    name = "fake-no-alignment"
    class_indexed_params: ClassVar[dict[str, int]] = {}


@pytest.fixture
def toy_data() -> AnnotationData:
    df = pd.DataFrame(
        {
            "item_id": ["i0", "i0", "i1", "i1", "i2", "i2", "i3", "i3"],
            "annotator_id": ["a0", "a1", "a0", "a1", "a0", "a1", "a0", "a1"],
            "label": ["yes", "yes", "no", "no", "yes", "no", "no", "no"],
        }
    )
    return AnnotationData.from_long_df(df)


@pytest.fixture
def confident_idata(one_hot_posterior: PosteriorBuilder) -> az.InferenceData:
    """q_z confidently predicts internal labels [0, 1, 0, 1] (yes, no, yes, no)
    for the 4 items in `toy_data`, with 2 chains that already agree."""
    labels = np.array([0, 1, 0, 1])
    q_z = np.stack(
        [
            one_hot_posterior(labels, n_classes=2, n_draws=20, high=0.95),
            one_hot_posterior(labels, n_classes=2, n_draws=20, high=0.95),
        ]
    )
    return az.from_dict(posterior={"q_z": q_z})


class TestRegistry:
    def test_register_and_get(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(PyRaterModel, "_registry", {})
        PyRaterModel.register("fake")(FakeModel)
        assert "fake" in PyRaterModel.list_available()

    def test_list_available_preserves_registration_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(PyRaterModel, "_registry", {})
        PyRaterModel.register("second-to-register")(FakeModel)
        PyRaterModel.register("first-to-register")(FakeModelNoAlignment)
        assert PyRaterModel.list_available() == ["second-to-register", "first-to-register"]

    def test_duplicate_registration_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(PyRaterModel, "_registry", {})
        PyRaterModel.register("fake")(FakeModel)
        with pytest.raises(ValueError, match="fake"):
            PyRaterModel.register("fake")(FakeModelNoAlignment)

    def test_get_unknown_model_raises_with_available_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(PyRaterModel, "_registry", {})
        PyRaterModel.register("fake")(FakeModel)
        with pytest.raises(KeyError, match="fake"):
            PyRaterModel.get("nonexistent")


class TestNotFittedBeforeFit:
    def test_labels_raises(self, confident_idata: az.InferenceData) -> None:
        model = FakeModel(confident_idata)
        with pytest.raises(NotFittedError):
            _ = model.labels

    def test_labels_proba_raises(self, confident_idata: az.InferenceData) -> None:
        model = FakeModel(confident_idata)
        with pytest.raises(NotFittedError):
            _ = model.labels_proba

    def test_difficulty_raises(self, confident_idata: az.InferenceData) -> None:
        model = FakeModel(confident_idata)
        with pytest.raises(NotFittedError):
            _ = model.difficulty

    def test_credibilities_raises(self, confident_idata: az.InferenceData) -> None:
        model = FakeModel(confident_idata)
        with pytest.raises(NotFittedError):
            _ = model.credibilities

    def test_idata_raises(self, confident_idata: az.InferenceData) -> None:
        model = FakeModel(confident_idata)
        with pytest.raises(NotFittedError):
            _ = model.idata

    def test_convergence_is_none_before_fit(self, confident_idata: az.InferenceData) -> None:
        model = FakeModel(confident_idata)
        assert model.convergence_ is None


class TestFitOrchestration:
    def test_fit_returns_self(
        self, toy_data: AnnotationData, confident_idata: az.InferenceData
    ) -> None:
        model = FakeModel(confident_idata)
        assert model.fit(toy_data, check_rhat=False) is model

    def test_sampling_kwargs_are_forwarded(
        self, toy_data: AnnotationData, confident_idata: az.InferenceData
    ) -> None:
        model = FakeModel(confident_idata)
        model.fit(toy_data, chains=2, warmup=111, iters=222, seed=7, check_rhat=False)
        assert model.sample_kwargs_received == {
            "chains": 2,
            "warmup": 111,
            "iters": 222,
            "seed": 7,
        }

    def test_labels_after_fit(
        self, toy_data: AnnotationData, confident_idata: az.InferenceData
    ) -> None:
        model = FakeModel(confident_idata)
        model.fit(toy_data, check_rhat=False)
        assert list(model.labels) == ["yes", "no", "yes", "no"]

    def test_labels_proba_shape(
        self, toy_data: AnnotationData, confident_idata: az.InferenceData
    ) -> None:
        model = FakeModel(confident_idata)
        model.fit(toy_data, check_rhat=False)
        assert model.labels_proba.shape == (4, 2)
        np.testing.assert_allclose(model.labels_proba.sum(axis=1), 1.0, atol=1e-6)


class TestDefaultDifficulty:
    def test_entropy_fallback_ranks_uncertain_items_as_harder(
        self, toy_data: AnnotationData, one_hot_posterior: PosteriorBuilder
    ) -> None:
        labels = np.array([0, 1, 0, 1])
        confident = one_hot_posterior(labels, n_classes=2, n_draws=20, high=0.99)
        uncertain = one_hot_posterior(labels, n_classes=2, n_draws=20, high=0.55)
        # item 0 confident in both chains, item 1 uncertain in both chains.
        mixed = confident.copy()
        mixed[:, 1, :] = uncertain[:, 1, :]
        q_z = np.stack([mixed, mixed])
        idata = az.from_dict(posterior={"q_z": q_z})

        model = FakeModel(idata)
        model.fit(toy_data, check_rhat=False)

        assert model.difficulty[0] < model.difficulty[1]
        assert (model.difficulty >= 0).all()
        assert (model.difficulty <= 1).all()


class TestCredibilities:
    def test_uniform_post_inference_agreement_rate(
        self, toy_data: AnnotationData, one_hot_posterior: PosteriorBuilder
    ) -> None:
        # Inferred internal labels: i0=yes(0), i1=no(1), i2=yes(0), i3=no(1).
        # a0 annotated: yes,no,yes,no -> agrees with all 4 -> credibility 1.0
        # a1 annotated: yes,no,no,no -> agrees with 3/4 (disagrees on i2) -> 0.75
        labels = np.array([0, 1, 0, 1])
        q_z = np.stack(
            [
                one_hot_posterior(labels, n_classes=2, n_draws=10, high=0.9),
                one_hot_posterior(labels, n_classes=2, n_draws=10, high=0.9),
            ]
        )
        idata = az.from_dict(posterior={"q_z": q_z})

        model = FakeModel(idata)
        model.fit(toy_data, check_rhat=False)

        credibilities = model.credibilities
        assert toy_data.annotator_ids is not None
        a0 = list(toy_data.annotator_ids).index("a0")
        a1 = list(toy_data.annotator_ids).index("a1")
        assert credibilities[a0] == pytest.approx(1.0)
        assert credibilities[a1] == pytest.approx(0.75)

    def test_raises_when_no_annotator_identifiers(
        self, toy_data: AnnotationData, confident_idata: az.InferenceData
    ) -> None:
        object.__setattr__(toy_data, "jj", None)
        object.__setattr__(toy_data, "annotator_ids", None)
        model = FakeModel(confident_idata)
        model.fit(toy_data, check_rhat=False)
        with pytest.raises(ModelCapabilityError, match="annotator"):
            _ = model.credibilities

    def test_works_for_models_with_no_native_annotator_dimension(
        self, toy_data: AnnotationData, one_hot_posterior: PosteriorBuilder
    ) -> None:
        # Even a model whose Stan file never used `jj` (like Multinomial or
        # ItemDiff) still gets credibilities: it's computed post-inference
        # from the raw data, not from any model-specific parameter.
        labels = np.array([0, 1, 0, 1])
        q_z = np.stack(
            [
                one_hot_posterior(labels, n_classes=2, n_draws=10, high=0.9),
                one_hot_posterior(labels, n_classes=2, n_draws=10, high=0.9),
            ]
        )
        idata = az.from_dict(posterior={"q_z": q_z})
        model = FakeModelNoAlignment(idata)
        model.fit(toy_data, check_rhat=False)
        assert model.credibilities.shape == (2,)


class TestAlignmentAndGoldWiring:
    def test_gold_from_data_is_used_automatically(
        self, one_hot_posterior: PosteriorBuilder
    ) -> None:
        df = pd.DataFrame(
            {
                "item_id": ["i0", "i0", "i1", "i1", "i2", "i2"],
                "annotator_id": ["a0", "a1", "a0", "a1", "a0", "a1"],
                "label": ["yes", "yes", "no", "no", "yes", "yes"],
            }
        )
        data = AnnotationData.from_long_df(df, gold={"i0": "yes", "i1": "no", "i2": "yes"})

        # "yes" is class 0 and "no" is class 1 (first-appearance order), so the
        # correct internal labeling matching gold is [0, 1, 0]. The fake
        # model's raw (unaligned) posterior deliberately uses the opposite,
        # equally-valid-but-wrong numbering [1, 0, 1] — `fit()` should notice
        # via `data.gold` and correct it automatically through align_to_gold.
        wrong_numbering = np.array([1, 0, 1])
        q_z = np.stack(
            [
                one_hot_posterior(wrong_numbering, n_classes=2, n_draws=10, high=0.95),
                one_hot_posterior(wrong_numbering, n_classes=2, n_draws=10, high=0.95),
            ]
        )
        idata = az.from_dict(posterior={"q_z": q_z})

        model = FakeModel(idata)
        model.fit(data, check_rhat=False)

        assert list(model.labels) == ["yes", "no", "yes"]

    def test_class_indexed_params_empty_skips_alignment_without_error(
        self, toy_data: AnnotationData, confident_idata: az.InferenceData
    ) -> None:
        model = FakeModelNoAlignment(confident_idata)
        model.fit(toy_data, check_rhat=False)
        assert list(model.labels) == ["yes", "no", "yes", "no"]


class TestConvergenceWiring:
    def test_check_rhat_true_populates_convergence_report(
        self, toy_data: AnnotationData, confident_idata: az.InferenceData
    ) -> None:
        model = FakeModel(confident_idata)
        model.fit(toy_data, check_rhat=True)
        assert isinstance(model.convergence_, ConvergenceReport)

    def test_check_rhat_false_leaves_convergence_none(
        self, toy_data: AnnotationData, confident_idata: az.InferenceData
    ) -> None:
        model = FakeModel(confident_idata)
        model.fit(toy_data, check_rhat=False)
        assert model.convergence_ is None

    def test_poor_convergence_emits_warning_but_still_returns_results(
        self,
        toy_data: AnnotationData,
        one_hot_posterior: PosteriorBuilder,
        rng: np.random.Generator,
    ) -> None:
        labels = np.array([0, 1, 0, 1])
        base = one_hot_posterior(labels, n_classes=2, n_draws=50, high=0.9)
        q_z = np.stack([base, base])
        # A junk scalar parameter that's badly mixed across chains, to
        # trigger the R-hat warning without touching label correctness.
        junk = np.stack(
            [
                rng.normal(loc=0.0, scale=0.1, size=(50,)),
                rng.normal(loc=50.0, scale=0.1, size=(50,)),
            ]
        )
        idata = az.from_dict(posterior={"q_z": q_z, "junk": junk})

        model = FakeModel(idata)
        with pytest.warns(ConvergenceWarning):
            model.fit(toy_data, check_rhat=True, rhat_threshold=1.01)

        assert model.convergence_ is not None
        assert not model.convergence_.ok
        # Results are still accessible, not blocked by the warning.
        assert model.labels.shape == (4,)

    def test_no_warning_when_check_rhat_false(
        self,
        toy_data: AnnotationData,
        one_hot_posterior: PosteriorBuilder,
        rng: np.random.Generator,
    ) -> None:
        labels = np.array([0, 1, 0, 1])
        base = one_hot_posterior(labels, n_classes=2, n_draws=50, high=0.9)
        q_z = np.stack([base, base])
        junk = np.stack(
            [
                rng.normal(loc=0.0, scale=0.1, size=(50,)),
                rng.normal(loc=50.0, scale=0.1, size=(50,)),
            ]
        )
        idata = az.from_dict(posterior={"q_z": q_z, "junk": junk})
        model = FakeModel(idata)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(toy_data, check_rhat=False)  # must not raise

import numpy as np
import pytest

from pyrater.data.annotation_data import AnnotationData
from pyrater.models.base import PyRaterModel

MODEL_NAMES = ["d&s", "mace", "multinomial", "hier_d&s", "item_difficulty", "lre"]


@pytest.mark.stan
@pytest.mark.parametrize("model_name", MODEL_NAMES)
class TestModelFitsOnTinyData:
    @pytest.fixture
    def fitted(self, model_name: str, tiny_data: AnnotationData) -> PyRaterModel:
        model = PyRaterModel.get(model_name)
        return model.fit(tiny_data, chains=4, warmup=1000, iters=1000, seed=1234)

    def test_labels_shape_and_validity(
        self, fitted: PyRaterModel, tiny_data: AnnotationData
    ) -> None:
        labels = fitted.labels
        assert labels.shape == (tiny_data.n_items,)
        assert set(labels) <= set(tiny_data.class_labels)

    def test_labels_proba_is_a_valid_distribution(
        self, fitted: PyRaterModel, tiny_data: AnnotationData
    ) -> None:
        proba = fitted.labels_proba
        assert proba.shape == (tiny_data.n_items, tiny_data.n_classes)
        assert not np.isnan(proba).any()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_difficulty_is_bounded(self, fitted: PyRaterModel, tiny_data: AnnotationData) -> None:
        difficulty = fitted.difficulty
        assert difficulty.shape == (tiny_data.n_items,)
        assert not np.isnan(difficulty).any()
        assert (difficulty >= -1e-8).all()
        assert (difficulty <= 1 + 1e-8).all()

    def test_credibilities_are_valid_for_every_model(
        self, fitted: PyRaterModel, tiny_data: AnnotationData
    ) -> None:
        # Every model supports credibilities now (post-inference, uniform
        # formula), including multinomial/item_difficulty, whose Stan files
        # have no annotator dimension at all.
        credibilities = fitted.credibilities
        assert credibilities.shape == (tiny_data.n_annotators,)
        assert not np.isnan(credibilities).any()
        assert (credibilities >= 0).all()
        assert (credibilities <= 1).all()

    # Deliberately no "recovers the correct gold labels" assertion here: on a
    # dataset this tiny (I=8), point-estimate label accuracy is a statistical
    # property of the fit, not a code-correctness property, and asserting an
    # exact accuracy threshold on real MCMC output is exactly the kind of
    # flaky check the test strategy calls for avoiding (shape/sanity only).
    # `align_to_gold`'s correctness is covered deterministically in
    # tests/unit/test_label_switching.py and test_base_model.py instead.

import numpy as np
import pandas as pd
import pytest

from pyrater.data.annotation_data import AnnotationData
from pyrater.models.base import PyRaterModel
from pyrater.models.baselines import KappaMajorityVotingModel, MajorityVotingModel


@pytest.fixture
def toy_data() -> AnnotationData:
    # i0: 2 "yes", 1 "no" -> majority "yes" (2/3 agree, difficulty 1/3)
    # i1: 3 "no" -> majority "no" (unanimous, difficulty 0)
    # i2: 1 "yes", 1 "no" -> tie, majority breaks to first-seen class ("yes")
    df = pd.DataFrame(
        {
            "item_id": ["i0", "i0", "i0", "i1", "i1", "i1", "i2", "i2"],
            "annotator_id": ["a0", "a1", "a2", "a0", "a1", "a2", "a0", "a1"],
            "label": ["yes", "yes", "no", "no", "no", "no", "yes", "no"],
        }
    )
    return AnnotationData.from_long_df(df)


class TestMajorityVotingModel:
    def test_registered_under_expected_name(self) -> None:
        assert "majority_voting" in PyRaterModel.list_available()
        assert isinstance(PyRaterModel.get("majority_voting"), MajorityVotingModel)

    def test_labels_are_the_per_item_mode(self, toy_data: AnnotationData) -> None:
        model = MajorityVotingModel()
        model.fit(toy_data, check_rhat=False)
        assert list(model.labels) == ["yes", "no", "yes"]

    def test_difficulty_is_disagreement_ratio(self, toy_data: AnnotationData) -> None:
        model = MajorityVotingModel()
        model.fit(toy_data, check_rhat=False)
        np.testing.assert_allclose(model.difficulty, [1 / 3, 0.0, 0.5])

    def test_credibilities_is_agreement_with_majority(self, toy_data: AnnotationData) -> None:
        model = MajorityVotingModel()
        model.fit(toy_data, check_rhat=False)
        assert toy_data.annotator_ids is not None
        # a0: yes(agree i0), no(agree i1), yes(agree i2) -> 3/3
        # a1: yes(agree i0), no(agree i1), no(disagree i2) -> 2/3
        # a2: no(disagree i0), no(agree i1) -> 1/2
        a0 = list(toy_data.annotator_ids).index("a0")
        a1 = list(toy_data.annotator_ids).index("a1")
        a2 = list(toy_data.annotator_ids).index("a2")
        assert model.credibilities[a0] == pytest.approx(1.0)
        assert model.credibilities[a1] == pytest.approx(2 / 3)
        assert model.credibilities[a2] == pytest.approx(0.5)

    def test_fit_ignores_sampling_kwargs(self, toy_data: AnnotationData) -> None:
        # Deterministic regardless of chains/seed: there's no MCMC here.
        model_a = MajorityVotingModel().fit(toy_data, chains=1, seed=1, check_rhat=False)
        model_b = MajorityVotingModel().fit(toy_data, chains=4, seed=99, check_rhat=False)
        assert list(model_a.labels) == list(model_b.labels)


class TestKappaMajorityVotingModel:
    def test_registered_under_expected_name(self) -> None:
        assert "kappa_majority_voting" in PyRaterModel.list_available()
        assert isinstance(PyRaterModel.get("kappa_majority_voting"), KappaMajorityVotingModel)

    def test_labels_and_difficulty_match_majority_voting(self, toy_data: AnnotationData) -> None:
        majority = MajorityVotingModel().fit(toy_data, check_rhat=False)
        kappa = KappaMajorityVotingModel().fit(toy_data, check_rhat=False)
        assert list(majority.labels) == list(kappa.labels)
        np.testing.assert_allclose(majority.difficulty, kappa.difficulty)

    def test_perfect_agreement_gives_kappa_one(self) -> None:
        df = pd.DataFrame(
            {
                "item_id": ["i0", "i0", "i1", "i1", "i2", "i2", "i3", "i3"],
                "annotator_id": ["a0", "a1", "a0", "a1", "a0", "a1", "a0", "a1"],
                "label": ["yes", "yes", "no", "no", "yes", "yes", "no", "no"],
            }
        )
        data = AnnotationData.from_long_df(df)
        model = KappaMajorityVotingModel().fit(data, check_rhat=False)
        np.testing.assert_allclose(model.credibilities, [1.0, 1.0], atol=1e-8)

    def test_insufficient_overlap_gives_nan_not_a_crash(self) -> None:
        # a0 and a1 never annotate the same item -> kappa undefined for either.
        df = pd.DataFrame(
            {
                "item_id": ["i0", "i1"],
                "annotator_id": ["a0", "a1"],
                "label": ["yes", "no"],
            }
        )
        data = AnnotationData.from_long_df(df)
        model = KappaMajorityVotingModel().fit(data, check_rhat=False)
        assert np.isnan(model.credibilities).all()

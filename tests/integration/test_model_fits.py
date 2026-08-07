import arviz as az
import numpy as np
import pytest
from scipy.special import logsumexp

from pyrater.data.annotation_data import AnnotationData
from pyrater.models.base import PyRaterModel

MODEL_NAMES = [
    "d&s",
    "mace",
    "multinomial",
    "hier_d&s",
    "item_difficulty",
    "lre",
    # Binary-only, so it belongs here only because `tiny_data` is K=2. A K>2
    # fixture must not be handed to it -- the model rejects that by design.
    "irt_2pl_ss",
]


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


@pytest.mark.stan
@pytest.mark.parametrize("model_name", MODEL_NAMES)
class TestPointwiseLogLikOnTinyData:
    """`log_lik` reaches the posterior in the shape ArviZ needs for LOO/WAIC.

    ArviZ routes a variable named `log_lik` into its own ``log_likelihood``
    group rather than leaving it in ``posterior``, which is why `az.loo` below
    needs no hand-built group.
    """

    @pytest.fixture
    def fitted(self, model_name: str, tiny_data: AnnotationData) -> PyRaterModel:
        model = PyRaterModel.get(model_name)
        return model.fit(tiny_data, chains=2, warmup=500, iters=500, seed=1234, check_rhat=False)

    def test_log_lik_has_one_entry_per_item(
        self, fitted: PyRaterModel, tiny_data: AnnotationData
    ) -> None:
        log_lik = fitted.idata["log_likelihood"]["log_lik"]
        # One per ITEM, not per annotation: the latent class is marginalized,
        # so the likelihood factorizes over items and tiny_data's 24 ratings
        # collapse to 8 pointwise terms.
        assert log_lik.shape[-1] == tiny_data.n_items
        assert log_lik.shape[-1] != tiny_data.n_annotations

    def test_log_lik_is_a_finite_log_probability(self, fitted: PyRaterModel) -> None:
        values = fitted.idata["log_likelihood"]["log_lik"].values
        assert np.isfinite(values).all()
        # A log-likelihood of a discrete outcome cannot exceed zero.
        assert (values <= 1e-8).all()

    def test_log_lik_equals_the_models_own_target_contribution(self, fitted: PyRaterModel) -> None:
        # The `model` block accumulates log_sum_exp(log_q_c[i]) per item, and
        # `log_q_c` is a transformed parameter, so the identity is checkable
        # draw by draw rather than merely asserted in a comment. If the two ever
        # part company, LOO would describe a different model from the one that
        # was fitted.
        expected = logsumexp(fitted.idata["posterior"]["log_q_c"].values, axis=-1)
        np.testing.assert_allclose(
            fitted.idata["log_likelihood"]["log_lik"].values, expected, atol=1e-6
        )

    def test_arviz_can_compute_loo(self, fitted: PyRaterModel) -> None:
        # The point of the whole addition: az.loo accepts the fit untouched.
        assert np.isfinite(az.loo(fitted.idata).elpd_loo)

    def test_posterior_samples_are_unchanged_by_the_addition(self, fitted: PyRaterModel) -> None:
        # log_lik is generated-quantities only, so it must not appear among the
        # sampled parameters. This is what makes the claim that cached fits stay
        # valid checkable rather than a comment in the .stan file.
        assert "log_lik" not in fitted.idata["posterior"].data_vars


@pytest.mark.stan
@pytest.mark.parametrize("model_name", MODEL_NAMES)
class TestPosteriorPredictiveOnTinyData:
    """The replicated vote histogram reaches the posterior and is coherent.

    Source-level tests cannot tell a correct emission from a plausible-looking
    wrong one -- only sampling can. These check the invariants that hold for
    every model regardless of how it describes raters, so a mis-transcribed
    emission shows up as a broken count rather than as a subtly wrong
    discrepancy nobody notices.
    """

    @pytest.fixture
    def fitted(self, model_name: str, tiny_data: AnnotationData) -> PyRaterModel:
        return PyRaterModel.get(model_name).fit(
            tiny_data, chains=2, warmup=500, iters=500, seed=1234, check_rhat=False
        )

    def test_histogram_is_shaped_by_classes_and_max_raters(
        self, fitted: PyRaterModel, tiny_data: AnnotationData
    ) -> None:
        hist = fitted.idata["posterior"]["rep_vote_hist"]
        # K x (R_max + 1): bins run from zero votes to every rater of the
        # busiest item voting for that class.
        assert hist.shape[-2] == tiny_data.n_classes
        assert hist.shape[-1] >= 2

    def test_every_item_lands_in_exactly_one_bin_per_class(
        self, fitted: PyRaterModel, tiny_data: AnnotationData
    ) -> None:
        # The invariant that catches an off-by-one in the +1 indexing or a
        # rep_counts array that is not reset between draws: for each class the
        # histogram partitions the items, so it must sum to n_items in every
        # draw of every chain.
        hist = fitted.idata["posterior"]["rep_vote_hist"].values
        np.testing.assert_array_equal(
            hist.sum(axis=-1), np.full(hist.shape[:-1], tiny_data.n_items)
        )

    def test_total_replicated_votes_match_the_observed_count(
        self, fitted: PyRaterModel, tiny_data: AnnotationData
    ) -> None:
        # Summing c * (number of items with c votes for class k) over all
        # classes and bins must recover the number of annotations: a
        # replication drops or invents no ratings.
        hist = fitted.idata["posterior"]["rep_vote_hist"].values
        counts = np.arange(hist.shape[-1])
        total = (hist * counts).sum(axis=(-1, -2))
        np.testing.assert_array_equal(total, np.full(total.shape, tiny_data.n_annotations))

    def test_latent_class_replication_is_in_range(
        self, fitted: PyRaterModel, tiny_data: AnnotationData
    ) -> None:
        z_rep = fitted.idata["posterior"]["z_rep"].values
        assert z_rep.shape[-1] == tiny_data.n_items
        assert z_rep.min() >= 1
        assert z_rep.max() <= tiny_data.n_classes

    def test_replication_is_not_a_copy_of_the_observed_data(
        self, fitted: PyRaterModel, tiny_data: AnnotationData
    ) -> None:
        # A check comparing the data against itself always passes and would
        # hide every misfit. The histogram must vary across draws.
        hist = fitted.idata["posterior"]["rep_vote_hist"].values
        assert hist.reshape(-1, hist.shape[-2] * hist.shape[-1]).std(axis=0).max() > 0.0

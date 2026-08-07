from importlib import resources

import arviz as az
import numpy as np
import pytest
from scipy.special import logsumexp

from pyrater.models.base import PyRaterModel
from pyrater.models.dawid_skene import DawidSkeneModel
from pyrater.models.hier_dawid_skene import HierDawidSkeneModel
from pyrater.models.item_difficulty import ItemDifficultyModel
from pyrater.models.logistic_random_effects import LogisticRandomEffectsModel
from pyrater.models.mace import MaceModel
from pyrater.models.multinomial import MultinomialModel


def _strip_comments(source: str) -> str:
    """Stan source with `//` comments removed.

    Source-level tests assert things about CODE. Matching against the raw file
    makes a model fail for explaining itself in a header comment, which
    punishes exactly the documentation these models most need.
    """
    return "\n".join(line.split("//")[0] for line in source.splitlines())


class TestRegistration:
    @pytest.mark.parametrize(
        ("name", "cls"),
        [
            ("d&s", DawidSkeneModel),
            ("mace", MaceModel),
            ("multinomial", MultinomialModel),
            ("hier_d&s", HierDawidSkeneModel),
            ("item_difficulty", ItemDifficultyModel),
            ("lre", LogisticRandomEffectsModel),
        ],
    )
    def test_registered_under_expected_name(self, name: str, cls: type[PyRaterModel]) -> None:
        assert name in PyRaterModel.list_available()
        assert isinstance(PyRaterModel.get(name), cls)

    def test_registration_order_matches_the_paper(self) -> None:
        # Reproduces Listing 3's order: 'd&s', 'hier_d&s', 'mace', ...
        available = PyRaterModel.list_available()
        assert available.index("d&s") < available.index("hier_d&s")
        assert available.index("hier_d&s") < available.index("mace")


class TestStanFileNames:
    def test_each_model_points_at_its_bundled_stan_file(self) -> None:
        assert DawidSkeneModel.stan_file == "DS.stan"
        assert MaceModel.stan_file == "MACE.stan"
        assert MultinomialModel.stan_file == "Multinomial.stan"
        assert HierDawidSkeneModel.stan_file == "HDS-NC.stan"
        assert ItemDifficultyModel.stan_file == "ID-NC.stan"
        assert LogisticRandomEffectsModel.stan_file == "LRE-NC.stan"


class TestRequiresAnnotatorDim:
    def test_multinomial_and_item_difficulty_have_no_annotator_dimension(self) -> None:
        assert MultinomialModel.requires_annotator_dim is False
        assert ItemDifficultyModel.requires_annotator_dim is False

    def test_annotator_modeling_models_require_it(self) -> None:
        assert DawidSkeneModel.requires_annotator_dim is True
        assert MaceModel.requires_annotator_dim is True
        assert HierDawidSkeneModel.requires_annotator_dim is True
        assert LogisticRandomEffectsModel.requires_annotator_dim is True


class TestClassIndexedParams:
    def test_dawid_skene(self) -> None:
        assert DawidSkeneModel.class_indexed_params == {"pi": 0, "beta": 1, "q_z": 1}

    def test_mace_has_no_label_switching_risk(self) -> None:
        assert MaceModel.class_indexed_params == {}

    def test_multinomial(self) -> None:
        assert MultinomialModel.class_indexed_params == {"pi": 0, "zeta": 0, "q_z": 1}

    def test_hier_dawid_skene(self) -> None:
        assert HierDawidSkeneModel.class_indexed_params == {
            "pi": 0,
            "zeta": 0,
            "Omega": 0,
            "beta_raw": 1,
            "beta_norm": 1,
            "q_z": 1,
        }

    def test_item_difficulty_excludes_theta_raw(self) -> None:
        params = ItemDifficultyModel.class_indexed_params
        assert params == {"pi": 0, "eta": 0, "Chi": 0, "theta_norm": 1, "q_z": 1}
        assert "theta_raw" not in params

    def test_logistic_random_effects_excludes_theta_raw(self) -> None:
        params = LogisticRandomEffectsModel.class_indexed_params
        assert params == {
            "pi": 0,
            "zeta": 0,
            "Omega": 0,
            "Chi": 0,
            "beta_raw": 1,
            "beta_norm": 1,
            "theta_norm": 1,
            "q_z": 1,
        }
        assert "theta_raw" not in params


class TestNativeDifficultyFromThetaNorm:
    """`item_difficulty` and `lre` both expose a `theta_norm[I,K,K]` (a
    log-softmax-normalized per-item confusion structure) and derive
    `.difficulty` from its diagonal, rather than the generic entropy
    fallback every other model uses."""

    @staticmethod
    def _fake_idata() -> az.InferenceData:
        # 2 items, K=3. Item 0 has a near-perfect diagonal (easy); item 1's
        # diagonal is weak (hard).
        rng = np.random.default_rng(0)
        pi = np.array([1 / 3, 1 / 3, 1 / 3])

        def log_softmax_row(strong: bool) -> np.ndarray:
            logits = np.array([5.0, -5.0, -5.0]) if strong else np.array([0.1, 0.0, -0.1])
            return logits - logsumexp(logits)

        theta_norm = np.zeros((2, 3, 3))
        theta_norm[0] = np.stack([np.roll(log_softmax_row(True), k) for k in range(3)])
        theta_norm[1] = np.stack([np.roll(log_softmax_row(False), k) for k in range(3)])

        q_z = rng.dirichlet(np.ones(3), size=2)[None, None, :, :]
        pi_arr = np.tile(pi, (1, 1, 1))
        theta_norm_arr = theta_norm[None, None, :, :, :]
        return az.from_dict(posterior={"q_z": q_z, "pi": pi_arr, "theta_norm": theta_norm_arr})

    def test_item_difficulty_ranks_weak_diagonal_as_harder(self) -> None:
        model = ItemDifficultyModel()
        model._idata = self._fake_idata()
        difficulty = model._compute_difficulty()
        assert difficulty[0] < difficulty[1]

    def test_lre_uses_the_same_formula(self) -> None:
        model = LogisticRandomEffectsModel()
        model._idata = self._fake_idata()
        difficulty = model._compute_difficulty()
        assert difficulty[0] < difficulty[1]


class TestPointwiseLogLik:
    """Every bundled model must expose `log_lik` so ArviZ can compute LOO/WAIC.

    Source-level checks: they need no CmdStan, so a model file that loses the
    declaration fails fast rather than at sampling time.
    """

    STAN_FILES = (
        "DS.stan",
        "MACE.stan",
        "Multinomial.stan",
        "HDS-NC.stan",
        "ID-NC.stan",
        "LRE-NC.stan",
        "IRT2PL-SS.stan",
    )

    @staticmethod
    def _source(stan_file: str) -> str:
        return resources.files("pyrater.stan").joinpath(stan_file).read_text()

    @pytest.mark.parametrize("stan_file", STAN_FILES)
    def test_declares_log_lik_over_items(self, stan_file: str) -> None:
        source = self._source(stan_file)
        # Indexed by item, not by annotation: these models marginalize the
        # latent class, so the likelihood factorizes over items.
        assert "vector[I] log_lik;" in source

    @pytest.mark.parametrize("stan_file", STAN_FILES)
    def test_log_lik_matches_the_models_own_target(self, stan_file: str) -> None:
        source = self._source(stan_file)
        # The `model` block accumulates exactly this quantity per item, so the
        # two must stay in step; a divergence between them would make LOO
        # describe a different model from the one that was fitted.
        assert "target += log_sum_exp(log_q_c[i]);" in source
        assert "log_lik[i] = log_sum_exp(log_q_c[i]);" in source

    @pytest.mark.parametrize("stan_file", STAN_FILES)
    def test_log_lik_is_additive_only(self, stan_file: str) -> None:
        # `log_lik` belongs to generated quantities alone. If it appeared in
        # `parameters` or `model` the addition would change the posterior.
        #
        # Counted over CODE, not over the file: a model whose header comment
        # explains what `log_lik` is would otherwise fail for documenting
        # itself, which is the opposite of what this is for.
        source = _strip_comments(self._source(stan_file))
        generated = source.split("generated quantities")[-1]
        assert source.count("log_lik") == generated.count("log_lik")


class TestPosteriorPredictiveCheck:
    """Every bundled model must replicate data it can be checked against.

    Convergence diagnostics report on the sampler, not on the model: R-hat and
    ESS can pass on a model that cannot produce data resembling the observed
    ratings. Han, Adiguzel & Carpenter (arXiv:2405.19521) show Dawid & Skene
    doing exactly that. These are source-level checks, so a model file that
    loses the block fails fast rather than after an hour of sampling.
    """

    STAN_FILES = TestPointwiseLogLik.STAN_FILES

    @staticmethod
    def _source(stan_file: str) -> str:
        return resources.files("pyrater.stan").joinpath(stan_file).read_text()

    @pytest.mark.parametrize("stan_file", STAN_FILES)
    def test_declares_the_replicated_vote_histogram(self, stan_file: str) -> None:
        source = self._source(stan_file)
        assert "array[K, R_max_ppc + 1] int rep_vote_hist" in source

    @pytest.mark.parametrize("stan_file", STAN_FILES)
    def test_samples_a_latent_class_per_item(self, stan_file: str) -> None:
        # The replication is conditional on a drawn true class, not on q_z
        # itself: averaging over the class posterior instead of sampling it
        # would understate the spread of the replicated histogram and make
        # every model look like a better fit than it is.
        source = self._source(stan_file)
        assert "z_rep[i] = categorical_rng(q_z[i]);" in source

    @pytest.mark.parametrize("stan_file", STAN_FILES)
    def test_replication_is_conditional_on_the_sampled_class(self, stan_file: str) -> None:
        # The emission differs per model -- that is what the check
        # discriminates between -- but every one of them must condition on
        # z_rep rather than on the observed y.
        # Anchored on the replication LOOP rather than on the `y_rep_n`
        # assignment. Models differ in where they mention the sampled class:
        # MACE puts it inside the draw, IRT2PL-SS computes a probability from
        # it on the line before. Both condition on it; only the loop bounds
        # them both.
        source = _strip_comments(self._source(stan_file))
        generated = source.split("generated quantities")[-1]
        loop = generated[generated.index("for (n in 1:N)") :]
        emission = loop[: loop.index("rep_counts[")]
        assert "z_rep[" in emission

    @pytest.mark.parametrize("stan_file", STAN_FILES)
    def test_per_item_counts_are_not_written_to_output(self, stan_file: str) -> None:
        # rep_counts is I x K per draw, so on a large corpus with a long chain
        # it is gigabytes of CSV nobody reads. It must stay inside a local
        # block so only the K x (R+1) histogram is saved.
        source = self._source(stan_file)
        generated = source[source.index("generated quantities") :]
        declaration = generated.index("array[I, K] int rep_counts")
        # The nearest enclosing brace before the declaration must be a bare
        # local block, not the generated quantities block itself.
        assert generated.rindex("{", 0, declaration) > generated.index("z_rep[i] =")

    @pytest.mark.parametrize("stan_file", STAN_FILES)
    def test_sizing_comes_from_the_data(self, stan_file: str) -> None:
        # R_max_ppc must be derived, not assumed: an item rated by more
        # annotators than the histogram has bins would write out of range.
        source = self._source(stan_file)
        assert "int R_max_ppc = max(n_ratings_ppc);" in source

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

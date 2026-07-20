from collections.abc import Callable

import arviz as az
import numpy as np
import numpy.typing as npt
import pytest

from pyrater.diagnostics.label_switching import (
    align_chains,
    align_to_gold,
    match_permutation,
)

PosteriorBuilder = Callable[..., npt.NDArray[np.float64]]

# A toy scenario: K=3 classes, I=6 items, 2 chains. Chain 0 uses the "true"
# labeling [0,0,1,1,2,2]. Chain 1 is internally consistent but uses classes 0
# and 1 swapped: [1,1,0,0,2,2]. `pi` is swapped identically, so we can check
# that alignment fixes *every* declared class-indexed parameter, not just q_z.
SWAP = np.array([1, 0, 2])  # class k in chain 1's numbering means class SWAP[k] in chain 0's.
CHAIN0_LABELS = np.array([0, 0, 1, 1, 2, 2])
CHAIN1_LABELS = SWAP[CHAIN0_LABELS]
PI_CHAIN0 = np.array([0.5, 0.3, 0.2])
PI_CHAIN1 = np.empty(3)
PI_CHAIN1[SWAP] = PI_CHAIN0
N_DRAWS = 4


@pytest.fixture
def toy_idata(one_hot_posterior: PosteriorBuilder) -> az.InferenceData:
    q_z = np.stack(
        [
            one_hot_posterior(CHAIN0_LABELS, n_classes=3, n_draws=N_DRAWS),
            one_hot_posterior(CHAIN1_LABELS, n_classes=3, n_draws=N_DRAWS),
        ]
    )  # (chain, draw, item, class)
    pi = np.stack(
        [np.tile(PI_CHAIN0, (N_DRAWS, 1)), np.tile(PI_CHAIN1, (N_DRAWS, 1))]
    )  # (chain, draw, class)
    return az.from_dict(posterior={"q_z": q_z, "pi": pi})


class TestMatchPermutation:
    def test_recovers_an_injected_swap(self) -> None:
        perm = match_permutation(CHAIN1_LABELS, CHAIN0_LABELS, n_classes=3)
        assert list(perm) == list(SWAP)

    def test_identity_when_already_aligned(self) -> None:
        perm = match_permutation(CHAIN0_LABELS, CHAIN0_LABELS, n_classes=3)
        assert list(perm) == [0, 1, 2]

    def test_is_a_valid_permutation(self) -> None:
        perm = match_permutation(CHAIN1_LABELS, CHAIN0_LABELS, n_classes=3)
        assert sorted(perm) == [0, 1, 2]


class TestAlignChains:
    def test_fixes_q_z_across_chains(self, toy_idata: az.InferenceData) -> None:
        aligned, _ = align_chains(
            toy_idata, class_indexed_params={"pi": 0, "q_z": 1}, reference_chain=0
        )
        q_z_mean = aligned["posterior"]["q_z"].mean(dim="draw").to_numpy()  # (chain, item, class)
        np.testing.assert_allclose(q_z_mean[0], q_z_mean[1], atol=1e-8)

    def test_fixes_every_declared_class_indexed_parameter(
        self, toy_idata: az.InferenceData
    ) -> None:
        aligned, _ = align_chains(
            toy_idata, class_indexed_params={"pi": 0, "q_z": 1}, reference_chain=0
        )
        pi_mean = aligned["posterior"]["pi"].mean(dim="draw").to_numpy()  # (chain, class)
        np.testing.assert_allclose(pi_mean[0], pi_mean[1], atol=1e-8)
        np.testing.assert_allclose(pi_mean[1], PI_CHAIN0, atol=1e-8)

    def test_reference_chain_permutation_is_identity(self, toy_idata: az.InferenceData) -> None:
        _, perm_by_chain = align_chains(
            toy_idata, class_indexed_params={"pi": 0, "q_z": 1}, reference_chain=0
        )
        assert list(perm_by_chain[0]) == [0, 1, 2]

    def test_other_chain_permutation_matches_injected_swap(
        self, toy_idata: az.InferenceData
    ) -> None:
        _, perm_by_chain = align_chains(
            toy_idata, class_indexed_params={"pi": 0, "q_z": 1}, reference_chain=0
        )
        assert list(perm_by_chain[1]) == list(SWAP)

    def test_no_op_when_class_indexed_params_is_empty(self, toy_idata: az.InferenceData) -> None:
        aligned, _ = align_chains(toy_idata, class_indexed_params={}, reference_chain=0)
        np.testing.assert_allclose(
            aligned["posterior"]["pi"].to_numpy(), toy_idata["posterior"]["pi"].to_numpy()
        )


class TestAlignToGold:
    @pytest.fixture
    def shifted_idata(self, one_hot_posterior: PosteriorBuilder) -> az.InferenceData:
        # Both chains already agree with each other (post align_chains), but their
        # *shared* numbering doesn't match the real class names: everything is
        # shifted by one relative to gold.
        shift = np.array([1, 2, 0])  # internal class k actually means gold class shift[k]
        shifted_labels = shift[CHAIN0_LABELS]
        q_z = np.stack(
            [
                one_hot_posterior(shifted_labels, n_classes=3, n_draws=N_DRAWS),
                one_hot_posterior(shifted_labels, n_classes=3, n_draws=N_DRAWS),
            ]
        )
        return az.from_dict(posterior={"q_z": q_z})

    def test_recovers_correct_global_numbering(
        self, shifted_idata: az.InferenceData, rng: np.random.Generator
    ) -> None:
        gold = {0: 0, 2: 1, 4: 2}  # item_idx -> true internal class id, one per class

        aligned, _ = align_to_gold(
            shifted_idata,
            class_indexed_params={"q_z": 1},
            gold=gold,
            n_classes=3,
            sample_size_per_class=5,
            rng=rng,
        )

        q_z_mean = aligned["posterior"]["q_z"].mean(dim=("chain", "draw")).to_numpy()
        predicted = q_z_mean.argmax(axis=1)
        for item_idx, true_class in gold.items():
            assert predicted[item_idx] == true_class

    def test_uses_at_most_sample_size_per_class(
        self, one_hot_posterior: PosteriorBuilder, rng: np.random.Generator
    ) -> None:
        q_z = np.stack(
            [
                one_hot_posterior(CHAIN0_LABELS, n_classes=3, n_draws=N_DRAWS),
                one_hot_posterior(CHAIN0_LABELS, n_classes=3, n_draws=N_DRAWS),
            ]
        )
        idata = az.from_dict(posterior={"q_z": q_z})
        gold = {i: int(CHAIN0_LABELS[i]) for i in range(6)}
        # Should not raise even though every item is gold-labeled and
        # sample_size_per_class is smaller than the number available per class.
        _, perm = align_to_gold(
            idata,
            class_indexed_params={"q_z": 1},
            gold=gold,
            n_classes=3,
            sample_size_per_class=1,
            rng=rng,
        )
        assert list(perm) == [0, 1, 2]

    def test_empty_gold_raises(
        self, one_hot_posterior: PosteriorBuilder, rng: np.random.Generator
    ) -> None:
        q_z = np.stack(
            [
                one_hot_posterior(CHAIN0_LABELS, n_classes=3, n_draws=N_DRAWS),
                one_hot_posterior(CHAIN0_LABELS, n_classes=3, n_draws=N_DRAWS),
            ]
        )
        idata = az.from_dict(posterior={"q_z": q_z})
        with pytest.raises(ValueError, match="gold"):
            align_to_gold(idata, class_indexed_params={"q_z": 1}, gold={}, n_classes=3, rng=rng)

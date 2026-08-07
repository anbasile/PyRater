"""IRT-2PL with sensitivity and specificity: Han, Adiguzel & Carpenter (2026).

The property under test is not fit quality but ORIENTATION. Every symmetric
annotation model in this registry has a latent class labelling that Stan cannot
pin down, and only gold resolves the global choice. This model's softplus
constraint forces every rater above chance, which excludes the flipped mode --
so a gold-free fit lands the right way up. That is the reason it is here, and
it is what these tests check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from pyrater.data.annotation_data import AnnotationData
from pyrater.exceptions import ModelCapabilityError
from pyrater.models.base import PyRaterModel
from pyrater.models.irt_2pl_ss import Irt2plSensSpecModel

pytestmark = pytest.mark.stan

N_ITEMS = 60
N_RATERS = 8


def _binary_data(seed: int = 0, flip_labels: bool = False) -> tuple[AnnotationData, np.ndarray]:
    """Raters who are all better than chance, with known true classes.

    `flip_labels` relabels the OBSERVATIONS, which is the situation that
    distinguishes an orientation-identified model from a symmetric one: the
    data means the same thing, and a symmetric model is free to come back
    upside down.
    """
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=N_ITEMS)
    # Accuracies from 0.65 to 0.95: all above chance, and spread enough that a
    # ranking exists to get right.
    accuracy = np.linspace(0.65, 0.95, N_RATERS)

    rows = []
    names = ["neg", "pos"] if not flip_labels else ["pos", "neg"]
    for i in range(N_ITEMS):
        for j in range(N_RATERS):
            correct = rng.random() < accuracy[j]
            label = truth[i] if correct else 1 - truth[i]
            rows.append({"item_id": f"i{i}", "annotator_id": f"r{j}", "label": names[label]})
    return AnnotationData.from_long_df(pd.DataFrame(rows)), truth


def _fit(data: AnnotationData) -> Irt2plSensSpecModel:
    # Constructed directly rather than through the registry so the
    # model's own sensitivity/specificity/discrimination are visible to a
    # type checker; `PyRaterModel.get` is exercised by the registry test.
    return Irt2plSensSpecModel().fit(
        data, chains=2, warmup=500, iters=500, seed=7, check_rhat=False
    )


class TestOrientation:
    def test_recovers_the_true_classes_without_gold(self) -> None:
        # No gold is passed to `fit`. A symmetric model could return the exact
        # complement of the truth and be equally happy; this one may not.
        data, truth = _binary_data()
        fitted = _fit(data)
        inferred = np.asarray(fitted.labels_proba).argmax(axis=1)
        # Map internal class index back through the label ordering the data
        # factorized, so this compares meanings rather than integers.
        as_label = np.asarray([list(data.class_labels).index("pos")] * 1)
        agreement = (inferred == (truth if as_label[0] == 1 else 1 - truth)).mean()
        assert agreement > 0.8, f"landed upside down or failed to fit: {agreement:.2f}"

    def test_every_rater_is_placed_above_chance(self) -> None:
        # The constraint, observed rather than asserted: sensitivity plus
        # specificity must exceed one for every rater, which is what rules the
        # flipped mode out of the support.
        data, _ = _binary_data()
        fitted = _fit(data)
        assert np.all(fitted.sensitivity + fitted.specificity > 1.0)

    def test_orientation_does_not_depend_on_row_order(self) -> None:
        # The failure this guards against is subtle and specific.
        # `from_long_df` factorizes labels with `sort=False`, so which real
        # class occupies internal slot 1 is decided by whichever rating happens
        # to come first in the frame. For a symmetric model that choice
        # propagates into the answer; here it must not.
        #
        # Same ratings, two orderings that factorize the other way round.
        data, truth = _binary_data()
        frame = data.to_long_df()
        first_label = frame["label"].iloc[0]
        reordered = pd.concat(
            [frame[frame["label"] != first_label], frame[frame["label"] == first_label]]
        ).reset_index(drop=True)
        other = AnnotationData.from_long_df(reordered)
        assert list(other.class_labels) != list(data.class_labels), "orderings did not differ"

        def recovered_labels(d: AnnotationData) -> np.ndarray:
            fitted = _fit(d)
            internal = np.asarray(fitted.labels_proba).argmax(axis=1)
            by_item = dict(zip(d.item_ids, [d.class_labels[c] for c in internal], strict=True))
            # Keyed by item id, so the two fits are comparable despite
            # factorizing items in different orders too.
            return np.asarray([by_item[f"i{i}"] for i in range(N_ITEMS)])

        assert (recovered_labels(data) == recovered_labels(other)).mean() > 0.8


class TestModelStructure:
    def test_separates_difficulty_from_discrimination(self) -> None:
        # The distinction the model exists to draw, and the one neither
        # item_difficulty nor lre makes: an item everyone misses and an item
        # only good raters get right are different objects.
        data, _ = _binary_data()
        fitted = _fit(data)
        assert fitted.difficulty.shape == (data.n_items,)
        assert fitted.discrimination.shape == (data.n_items,)
        assert np.all(fitted.discrimination > 0)

    def test_ranks_raters_by_ability(self) -> None:
        # Rater accuracy rises monotonically by construction, so a model that
        # estimates ability at all must correlate strongly with it.
        data, _ = _binary_data()
        fitted = _fit(data)
        assert data.annotator_ids is not None
        ids = data.annotator_ids.tolist()
        order = [ids.index(f"r{j}") for j in range(N_RATERS)]
        ability = (fitted.sensitivity + fitted.specificity)[order]
        rho = spearmanr(ability, np.linspace(0.65, 0.95, N_RATERS)).statistic
        assert rho > 0.6, f"ability ordering not recovered: rho = {rho:.2f}"

    def test_declares_no_class_indexed_parameters(self) -> None:
        # Empty on purpose: permuting the class axis would move the fit out of
        # the region the constraint admits rather than to an equivalent point.
        assert PyRaterModel.get("irt_2pl_ss").class_indexed_params == {}


class TestBinaryOnly:
    def test_rejects_more_than_two_classes(self) -> None:
        # Sensitivity and specificity are binary notions. A K > 2 dataset must
        # fail loudly rather than fit something meaningless: this model sits in
        # the same registry as six that accept any K, so a caller sweeping over
        # `list_available()` would otherwise reach it silently.
        rng = np.random.default_rng(1)
        rows = [
            {"item_id": f"i{i}", "annotator_id": f"r{j}", "label": f"c{rng.integers(0, 4)}"}
            for i in range(20)
            for j in range(4)
        ]
        data = AnnotationData.from_long_df(pd.DataFrame(rows))
        assert data.n_classes > 2
        with pytest.raises((RuntimeError, ValueError, ModelCapabilityError)):
            PyRaterModel.get("irt_2pl_ss").fit(
                data, chains=1, warmup=100, iters=100, seed=1, check_rhat=False
            )


class TestDifficultyContract:
    """`difficulty` means the same thing here as everywhere else in pyrater.

    The library's contract is one minus expected own-class accuracy: a
    probability, higher meaning harder, comparable across models. This model's
    natural difficulty parameter is an unbounded log-odds `beta`, so the two
    are deliberately different quantities under different names -- and the
    shared contract test in test_model_fits.py only checks the bounds.
    """

    def test_difficulty_is_a_probability(self) -> None:
        data, _ = _binary_data()
        fitted = _fit(data)
        assert fitted.difficulty.shape == (data.n_items,)
        assert np.all(fitted.difficulty >= 0.0)
        assert np.all(fitted.difficulty <= 1.0)

    def test_difficulty_logit_is_the_raw_parameter(self) -> None:
        # Exposed separately because it is what the model estimates, and it is
        # NOT bounded -- which is exactly why it cannot be `difficulty`.
        data, _ = _binary_data()
        fitted = _fit(data)
        assert fitted.difficulty_logit.shape == (data.n_items,)
        assert np.isfinite(fitted.difficulty_logit).all()

    def test_the_two_orderings_agree(self) -> None:
        # They are different scales but must rank items the same way: a larger
        # beta means a harder item means a lower expected accuracy. If these
        # ever disagree, one of the two is computed with a sign error.
        data, _ = _binary_data()
        fitted = _fit(data)
        rho = spearmanr(fitted.difficulty, fitted.difficulty_logit).statistic
        assert rho > 0.9, f"difficulty and its logit disagree on ordering: rho = {rho:.2f}"

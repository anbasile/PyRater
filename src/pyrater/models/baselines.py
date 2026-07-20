"""Non-Bayesian baselines: majority voting, and majority voting ranked by
Cohen's kappa instead of raw agreement — included so "is the Bayesian
machinery worth it" is always a one-line comparison against something free,
and to reproduce the paper's own Kappa-vs-majority-voting comparison."""

from __future__ import annotations

from typing import Any, ClassVar

import arviz as az
import numpy as np
import numpy.typing as npt

from pyrater.data.annotation_data import AnnotationData
from pyrater.exceptions import ModelCapabilityError
from pyrater.models.base import PyRaterModel


def _item_class_counts(data: AnnotationData) -> npt.NDArray[np.float64]:
    counts = np.zeros((data.n_items, data.n_classes))
    np.add.at(counts, (data.ii, data.y), 1)
    return counts


def _cohen_kappa(y1: npt.ArrayLike, y2: npt.ArrayLike, n_classes: int) -> float:
    """Cohen's kappa between two equal-length label sequences over the same
    items, hand-rolled to avoid a scikit-learn dependency for one metric.

    Returns NaN if chance agreement is total (e.g. only one class appears),
    since kappa is undefined in that degenerate case.
    """
    a = np.asarray(y1)
    b = np.asarray(y2)
    n = len(a)
    observed = float(np.mean(a == b))
    p1 = np.bincount(a, minlength=n_classes) / n
    p2 = np.bincount(b, minlength=n_classes) / n
    expected = float(np.sum(p1 * p2))
    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


@PyRaterModel.register("majority_voting")
class MajorityVotingModel(PyRaterModel):
    """Predicts each item's label as the mode of its raw annotations.

    Not a Stan model: `.fit()` ignores every MCMC-sampling keyword (chains,
    warmup, iters, seed) and computes an instant, deterministic result.
    `.credibilities` is inherited unchanged from the base class's uniform
    post-inference agreement formula, which for this model reduces exactly
    to "how often did this annotator agree with the majority."
    """

    name: ClassVar[str] = "majority_voting"
    class_indexed_params: ClassVar[dict[str, int]] = {}

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
        counts = _item_class_counts(data)
        majority = counts.argmax(axis=1)
        q_z = np.zeros((1, 1, data.n_items, data.n_classes))
        q_z[0, 0, np.arange(data.n_items), majority] = 1.0
        return az.from_dict(posterior={"q_z": q_z})

    def _compute_difficulty(self) -> npt.NDArray[np.float64]:
        assert self._data is not None
        counts = _item_class_counts(self._data)
        total = counts.sum(axis=1)
        majority_count = counts.max(axis=1)
        result: npt.NDArray[np.float64] = 1.0 - majority_count / total
        return result


@PyRaterModel.register("kappa_majority_voting")
class KappaMajorityVotingModel(MajorityVotingModel):
    """Same majority-vote labels/difficulty as `majority_voting`, but ranks
    annotators by their mean pairwise Cohen's kappa against every other
    annotator on commonly-annotated items, rather than raw agreement —
    reproducing the paper's own Kappa baseline for annotator ranking."""

    name: ClassVar[str] = "kappa_majority_voting"

    @property
    def credibilities(self) -> npt.NDArray[np.float64]:
        _ = self.idata  # raises NotFittedError if not fitted
        data = self._data
        assert data is not None
        if data.jj is None:
            raise ModelCapabilityError(
                "Credibilities are unavailable: the input data has no annotator identifiers."
            )

        by_annotator: list[dict[int, int]] = [{} for _ in range(data.n_annotators)]
        for item, annotator, label in zip(data.ii, data.jj, data.y, strict=True):
            by_annotator[annotator][int(item)] = int(label)

        kappas = np.full(data.n_annotators, np.nan)
        for j in range(data.n_annotators):
            pairwise: list[float] = []
            for k in range(data.n_annotators):
                if k == j:
                    continue
                shared = set(by_annotator[j]) & set(by_annotator[k])
                if len(shared) < 2:
                    continue
                y1 = [by_annotator[j][i] for i in shared]
                y2 = [by_annotator[k][i] for i in shared]
                kappa = _cohen_kappa(y1, y2, data.n_classes)
                if not np.isnan(kappa):
                    pairwise.append(kappa)
            if pairwise:
                kappas[j] = float(np.mean(pairwise))
        return kappas

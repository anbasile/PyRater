"""IRT-2PL with per-annotator sensitivity and specificity.

Han, Adiguzel & Carpenter (2026), "Hierarchical Bayesian Crowdsourcing with
Item Difficulty" (arXiv:2405.19521), reduction "a" of their Table 1 -- the
model that won on both of their datasets. Stan code adapted from
https://github.com/seongwoohan/bayes-crowdsourcing-irt (MIT).

Two properties distinguish it from the rest of the registry.

It separates an item's DIFFICULTY from its DISCRIMINATION, so an item everyone
gets wrong and an item that only good annotators get right are different objects.
Dawid & Skene has neither; `item_difficulty` and `lre` have the first but not
the second.

More consequentially, it is ORIENTATION-IDENTIFIED WITHOUT GOLD. See the
header of IRT2PL-SS.stan: the softplus constraint on sensitivity forces every
annotator above chance, which excludes the globally flipped mode that symmetric
models cannot rule out. `class_indexed_params` is therefore empty -- there is
no permutation to correct, exactly as for MACE.

Two restrictions follow, and both are checked rather than documented away.

BINARY ONLY. Sensitivity and specificity are two-class notions, so the model
rejects K > 2 in its `transformed data` block. Every other model in the
registry accepts any K, so code that sweeps `PyRaterModel.list_available()`
has to expect this one to refuse some datasets.

ANNOTATORS MUST BEAT CHANCE. That is what buys the orientation, and it is a
real restriction: a pool containing genuinely adversarial annotators violates
it, and the model will place them at the boundary rather than below it. Where
that is a live possibility, prefer a symmetric model and resolve the
orientation another way.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from pyrater.models._stan_backed import StanBackedModel
from pyrater.models.base import PyRaterModel


@PyRaterModel.register("irt_2pl_ss")
class Irt2plSensSpecModel(StanBackedModel):
    stan_file: ClassVar[str] = "IRT2PL-SS.stan"

    # Empty, and not an oversight. The identifiability constraint pins the
    # labelling, so permuting the class axis would move the fit OUT of the
    # region the model admits rather than into an equivalent one.
    class_indexed_params: ClassVar[dict[str, int]] = {}

    def _compute_difficulty(self) -> npt.NDArray[np.float64]:
        """One minus the item's expected own-class accuracy, in [0, 1].

        The library's contract: `difficulty` is a probability, comparable
        across models, higher meaning harder -- the same quantity
        `_difficulty.theta_norm_difficulty` computes for `item_difficulty` and
        `lre`. Returning `beta` here instead would be more natural to this
        model and would break every caller that compares difficulty between
        models, so the raw latent parameter is exposed as `difficulty_logit`
        below rather than through this name.

        Averaged over annotators and weighted by prevalence, since an item's
        difficulty in this model is only defined relative to who annotates it.
        """
        posterior = self.idata["posterior"]
        beta = posterior["beta"].mean(dim=("chain", "draw")).to_numpy()  # (I,)
        delta = posterior["delta"].mean(dim=("chain", "draw")).to_numpy()  # (I,)
        sens = posterior["logit_alpha_sens"].mean(dim=("chain", "draw")).to_numpy()  # (J,)
        spec = posterior["logit_alpha_spec"].mean(dim=("chain", "draw")).to_numpy()  # (J,)
        pi = float(posterior["pi"].mean(dim=("chain", "draw")).to_numpy())

        def expit(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return 1.0 / (1.0 + np.exp(-x))

        gap = delta[:, None] * (sens[None, :] - beta[:, None])  # (I, J)
        # P(correct | positive) and P(correct | negative); the second is
        # 1 - P(report positive | negative), which simplifies to the same form
        # with specificity in place of sensitivity.
        accuracy = pi * expit(gap).mean(axis=1) + (1.0 - pi) * expit(
            delta[:, None] * (spec[None, :] - beta[:, None])
        ).mean(axis=1)
        result: npt.NDArray[np.float64] = 1.0 - accuracy
        return result

    @property
    def difficulty_logit(self) -> npt.NDArray[np.float64]:
        """Item difficulty on the model's own log-odds scale (`beta`).

        Unlike `difficulty`, this is not comparable across models -- but it is
        the parameter the model actually estimates, and the one that pairs with
        `discrimination` to distinguish an item everyone misses from an item
        only good annotators get right.
        """
        result: npt.NDArray[np.float64] = (
            self.idata["posterior"]["beta"].mean(dim=("chain", "draw")).to_numpy()
        )
        return result

    @property
    def sensitivity(self) -> npt.NDArray[np.float64]:
        """Per-annotator sensitivity, on the probability scale.

        The rate of reporting the positive class on an item that is positive.
        Which real label "positive" names depends on the order
        `AnnotationData` factorized the labels in -- read it off
        `data.class_labels[1]`, never assume it.
        """
        logits = self.idata["posterior"]["logit_alpha_sens"].mean(dim=("chain", "draw")).to_numpy()
        result: npt.NDArray[np.float64] = 1.0 / (1.0 + np.exp(-logits))
        return result

    @property
    def specificity(self) -> npt.NDArray[np.float64]:
        """Per-annotator specificity, on the probability scale."""
        logits = self.idata["posterior"]["logit_alpha_spec"].mean(dim=("chain", "draw")).to_numpy()
        result: npt.NDArray[np.float64] = 1.0 / (1.0 + np.exp(-logits))
        return result

    @property
    def discrimination(self) -> npt.NDArray[np.float64]:
        """Per-item discrimination.

        How sharply an item separates annotators by ability. An item with delta
        near zero is answered the same way by everyone and carries no
        information about who is good.
        """
        result: npt.NDArray[np.float64] = (
            self.idata["posterior"]["delta"].mean(dim=("chain", "draw")).to_numpy()
        )
        return result

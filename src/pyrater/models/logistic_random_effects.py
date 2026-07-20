"""Logistic Random Effects: combines Hierarchical D&S's annotator-ability
structure and Item Difficulty's per-item difficulty structure into one
IRT-style model (probability of a correct annotation is a function of
annotator ability minus item difficulty)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from pyrater.models._difficulty import theta_norm_difficulty
from pyrater.models._stan_backed import StanBackedModel
from pyrater.models.base import PyRaterModel


@PyRaterModel.register("lre")
class LogisticRandomEffectsModel(StanBackedModel):
    stan_file: ClassVar[str] = "LRE-NC.stan"
    class_indexed_params: ClassVar[dict[str, int]] = {
        "pi": 0,
        "zeta": 0,
        "Omega": 0,
        "Chi": 0,
        "beta_raw": 1,
        "beta_norm": 1,
        "theta_norm": 1,
        "q_z": 1,
        # `theta_raw` is deliberately excluded, same reasoning as ItemDifficultyModel.
    }

    def _compute_difficulty(self) -> npt.NDArray[np.float64]:
        posterior = self.idata["posterior"]
        theta_norm_mean = posterior["theta_norm"].mean(dim=("chain", "draw")).to_numpy()
        pi_mean = posterior["pi"].mean(dim=("chain", "draw")).to_numpy()
        return theta_norm_difficulty(theta_norm_mean, pi_mean)

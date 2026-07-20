"""The Item Difficulty model: no annotator structure at all -- instead every
item has its own latent per-class confusion profile, drawn from a
class-level hierarchical prior."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from pyrater.models._difficulty import theta_norm_difficulty
from pyrater.models._stan_backed import StanBackedModel
from pyrater.models.base import PyRaterModel


@PyRaterModel.register("item_difficulty")
class ItemDifficultyModel(StanBackedModel):
    stan_file: ClassVar[str] = "ID-NC.stan"
    requires_annotator_dim: ClassVar[bool] = False
    class_indexed_params: ClassVar[dict[str, int]] = {
        "pi": 0,
        "eta": 0,
        "Chi": 0,
        "theta_norm": 1,
        "q_z": 1,
        # `theta_raw` is deliberately excluded: it's a shared, non-class-
        # indexed noise vector (class-dependence only appears after scaling
        # by `Chi`), so it must not be permuted.
    }

    def _compute_difficulty(self) -> npt.NDArray[np.float64]:
        posterior = self.idata["posterior"]
        theta_norm_mean = posterior["theta_norm"].mean(dim=("chain", "draw")).to_numpy()
        pi_mean = posterior["pi"].mean(dim=("chain", "draw")).to_numpy()
        return theta_norm_difficulty(theta_norm_mean, pi_mean)

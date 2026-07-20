"""The simplest pooled model: a single, shared confusion matrix for the
whole crowd (no per-annotator structure at all)."""

from __future__ import annotations

from typing import ClassVar

from pyrater.models._stan_backed import StanBackedModel
from pyrater.models.base import PyRaterModel


@PyRaterModel.register("multinomial")
class MultinomialModel(StanBackedModel):
    stan_file: ClassVar[str] = "Multinomial.stan"
    requires_annotator_dim: ClassVar[bool] = False
    class_indexed_params: ClassVar[dict[str, int]] = {"pi": 0, "zeta": 0, "q_z": 1}

"""Multi-Annotator Competence Estimation (Hovy et al., 2013): every
annotation is either "trustworthy" (reports the true label) or "spamming"
(reports noise), governed by a per-annotator competence scalar."""

from __future__ import annotations

from typing import ClassVar

from pyrater.models._stan_backed import StanBackedModel
from pyrater.models.base import PyRaterModel


@PyRaterModel.register("mace")
class MaceModel(StanBackedModel):
    stan_file: ClassVar[str] = "MACE.stan"
    # MACE has no label-switching problem: its `y[n] == h` indicator anchors
    # the latent class axis directly to the observed label codes, so there's
    # no symmetric relabeling that leaves the likelihood invariant.
    class_indexed_params: ClassVar[dict[str, int]] = {}

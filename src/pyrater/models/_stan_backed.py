"""Shared machinery for every model that samples via a bundled Stan file."""

from __future__ import annotations

from typing import Any, ClassVar

import arviz as az

from pyrater.data.annotation_data import AnnotationData
from pyrater.models.base import PyRaterModel
from pyrater.stan import get_compiled_model


class StanBackedModel(PyRaterModel):
    """A `PyRaterModel` whose posterior comes from compiling and sampling a
    bundled `.stan` file (as opposed to :class:`~pyrater.models.baselines.BaselineModel`,
    which computes everything in pure NumPy)."""

    stan_file: ClassVar[str]
    requires_annotator_dim: ClassVar[bool] = True

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
        stan_data = data.to_stan_data(include_annotators=self.requires_annotator_dim)
        model = get_compiled_model(self.stan_file)
        fit = model.sample(
            data=dict(stan_data),
            chains=chains,
            iter_warmup=warmup,
            iter_sampling=iters,
            seed=seed,
            **kwargs,
        )
        result: az.InferenceData = az.from_cmdstanpy(fit)
        return result

"""Shared machinery for models written in NumPyro rather than Stan.

Every model here sums the latent class out by hand, so the log density is a
smooth function of the rater parameters and one matmul per observed label. That
is what makes a dense items x raters matrix cheap: the cost per gradient is
items x raters x classes, with no per-annotation loop.

JAX and NumPyro are optional (`pip install pyrater[jax]`), so nothing in this
module imports them at import time.
"""

from __future__ import annotations

import abc
from collections.abc import Hashable, Mapping
from typing import Any, ClassVar

import arviz as az
import numpy as np
import numpy.typing as npt

from pyrater.data.annotation_data import AnnotationData
from pyrater.models.base import PyRaterModel

# Stands in for log(0) when a known label rules a class out. Finite, so the
# gradient of the logsumexp stays finite.
_RULED_OUT = -1e6


def dense_votes(data: AnnotationData) -> npt.NDArray[np.int64]:
    """(items, annotators) label codes, -1 where an annotator skipped an item."""
    if data.jj is None:
        raise ValueError("NumPyro-backed models need annotator identifiers.")
    votes = np.full((data.n_items, data.n_annotators), -1, dtype=np.int64)
    votes[data.ii, data.jj] = data.y
    return votes


def vote_loglik(indicators: list[Any], log_confusion: Any) -> Any:
    """Sum over annotators of log P(vote | class), for every item and class.

    `indicators[l]` is the (items, annotators) 0/1 matrix of "voted l";
    `log_confusion` is (annotators, class, vote). A skipped item is all-zero in
    every indicator and so contributes nothing.
    """
    return sum(ind @ log_confusion[:, :, label] for label, ind in enumerate(indicators))


class NumpyroBackedModel(PyRaterModel):
    """A `PyRaterModel` whose posterior comes from NumPyro.

    Subclasses implement `_priors`, which samples the parameters, and
    `_item_loglik`, which returns the (items, classes) joint log density of each
    item's votes and each class. Fitting, the per-item class posterior and the
    pointwise log likelihood are shared.

    `fit(..., method="map")` returns the posterior mode as a single draw, in
    seconds; `method="nuts"` samples.
    """

    requires_annotator_dim: ClassVar[bool] = True
    # Identified by a prior that favours the diagonal, so no relabelling pass.
    class_indexed_params: ClassVar[Mapping[str, int]] = {}

    @abc.abstractmethod
    def _priors(self, n_classes: int, tensors: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def _item_loglik(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _tensors(
        self, data: AnnotationData, groups: Mapping[Hashable, Hashable] | None
    ) -> dict[str, Any]:
        import jax.numpy as jnp

        votes = dense_votes(data)
        return {
            "indicators": [
                jnp.asarray((votes == label).astype(np.float64)) for label in range(data.n_classes)
            ],
            "n_annotators": data.n_annotators,
        }

    def _sample(
        self,
        data: AnnotationData,
        *,
        chains: int,
        warmup: int,
        iters: int,
        seed: int | None,
        method: str = "nuts",
        groups: Mapping[Hashable, Hashable] | None = None,
        anchors: Mapping[Hashable, Hashable] | None = None,
        map_steps: int = 1500,
        learning_rate: float = 0.05,
        **kwargs: Any,
    ) -> az.InferenceData:
        import jax
        import jax.numpy as jnp
        import numpyro
        from jax.scipy.special import logsumexp
        from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_median
        from numpyro.infer.autoguide import AutoDelta

        # The summed log density of 10^7 votes loses digits in float32.
        jax.config.update("jax_enable_x64", True)

        n_classes = data.n_classes
        tensors = self._tensors(data, groups)

        # A known label enters as an observation of the class itself.
        mask = np.zeros((data.n_items, n_classes))
        if anchors:
            for item, cls in AnnotationData._encode_gold(
                anchors, data.item_ids, data.class_labels
            ).items():
                mask[item] = _RULED_OUT
                mask[item, cls] = 0.0
        tensors["anchor_mask"] = jnp.asarray(mask)

        def joint(params: dict[str, Any]) -> Any:
            return self._item_loglik(params, tensors) + tensors["anchor_mask"]

        def model() -> None:
            params = self._priors(n_classes, tensors)
            numpyro.factor("votes", logsumexp(joint(params), axis=-1).sum())

        key = jax.random.PRNGKey(0 if seed is None else seed)
        if method == "map":
            guide = AutoDelta(model, init_loc_fn=init_to_median)
            svi = SVI(model, guide, numpyro.optim.Adam(learning_rate), Trace_ELBO())
            result = svi.run(key, map_steps, progress_bar=False)
            point = guide.median(result.params)
            draws = {name: np.asarray(value)[None, None] for name, value in point.items()}
        elif method == "nuts":
            mcmc = MCMC(
                NUTS(model, init_strategy=init_to_median),
                num_warmup=warmup,
                num_samples=iters,
                num_chains=chains,
                chain_method="sequential",
                progress_bar=False,
                **kwargs,
            )
            mcmc.run(key)
            draws = {
                name: np.asarray(value)
                for name, value in mcmc.get_samples(group_by_chain=True).items()
            }
        else:
            raise ValueError(f"method must be 'map' or 'nuts', got {method!r}")

        n_chains, n_draws = next(iter(draws.values())).shape[:2]
        # NumPy, not JAX: slicing a JAX array at a constant index outside `jit`
        # compiles a new program for every index, which made reading 4,000
        # draws back take longer than drawing them.
        flat = {name: v.reshape(-1, *v.shape[2:]) for name, v in draws.items()}

        @jax.jit
        def per_draw(params: dict[str, Any]) -> tuple[Any, Any, Any]:
            scores = joint(params)
            total = logsumexp(scores, axis=-1)
            return jnp.exp(scores - total[:, None]), total, self._competence(params, tensors)

        q_z, log_lik, theta = [], [], []
        for d in range(n_chains * n_draws):
            q, ll, th = per_draw({name: v[d] for name, v in flat.items()})
            q_z.append(np.asarray(q, dtype=np.float32))
            log_lik.append(np.asarray(ll))
            theta.append(np.asarray(th))

        def shaped(values: list[npt.NDArray[Any]]) -> npt.NDArray[Any]:
            stacked = np.stack(values)
            return stacked.reshape(n_chains, n_draws, *stacked.shape[1:])

        posterior = dict(draws)
        posterior["q_z"] = shaped(q_z)
        posterior["theta"] = shaped(theta)
        return az.from_dict(posterior=posterior, log_likelihood={"log_lik": shaped(log_lik)})

    def _competence(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        """One number per annotator: the probability of reporting the true
        class, averaged over classes. Named `theta` in the posterior."""
        import jax.numpy as jnp

        confusion = self._confusion(params, tensors)
        return jnp.diagonal(confusion, axis1=1, axis2=2).mean(axis=-1)

    def _confusion(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        """(annotators, class, vote) probabilities."""
        raise NotImplementedError

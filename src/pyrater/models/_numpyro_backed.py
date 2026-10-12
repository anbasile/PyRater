"""Shared machinery for models written in NumPyro rather than Stan.

Every model here sums the latent class out by hand, so the log density is a
smooth function of the rater parameters and one matmul per observed label. That
is what makes a dense items x raters matrix cheap: the cost per gradient is
items x raters x classes, with no per-annotation loop.

JAX and NumPyro are optional (`pip install "pyrater[jax]"`), so nothing in this
module imports them at import time: importing pyrater registers these models
either way, and a missing extra only surfaces, as an `ImportError` that names
the extra, when one of them is fitted.
"""

from __future__ import annotations

import abc
import logging
import warnings
from collections.abc import Hashable, Mapping
from typing import Any, ClassVar, Self

import arviz as az
import numpy as np
import numpy.typing as npt

from pyrater.data.annotation_data import AnnotationData
from pyrater.exceptions import ConvergenceWarning
from pyrater.models.base import PyRaterModel

logger = logging.getLogger(__name__)

# Stands in for log(0) when a known label rules a class out. Finite, so the
# gradient of the logsumexp stays finite.
_RULED_OUT = -1e6

_METHODS = ("nuts", "map")

# The mode search counts as converged when the loss, averaged over the last
# `_MAP_WINDOW` steps, differs from its average over the `_MAP_WINDOW` steps
# before those by less than `_MAP_RTOL`, relative to its size. Averages, not
# two single steps: Adam at a fixed learning rate keeps jittering around the
# mode, and one step's jitter says nothing about whether it is still moving.
# The tolerance is loose on purpose. A mode on the boundary of a simplex is
# approached but never reached in the unconstrained space the optimiser works
# in, so the loss keeps creeping down at the fifth or sixth digit for as long
# as it runs; this is meant to catch a run that was stopped early, not that.
_MAP_WINDOW = 100
_MAP_RTOL = 1e-4

# `MCMC` arguments that `fit` sets from its own parameters.
_MCMC_RESERVED = {"num_warmup": "warmup", "num_samples": "iters", "num_chains": "chains"}

# NumPyro's name for each per-draw sampler statistic -> ArviZ's name for it.
_SAMPLE_STATS = {
    "diverging": "diverging",
    "energy": "energy",
    "num_steps": "n_steps",
    "accept_prob": "acceptance_rate",
    "adapt_state.step_size": "step_size",
}


def _require_backend() -> None:
    """Raise an `ImportError` that names the extra if JAX or NumPyro is missing."""
    try:
        import jax  # noqa: F401
        import numpyro  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "The NumPyro-backed models (registry names starting with 'np_') need JAX and "
            'NumPyro, which are an optional extra. Install them with: pip install "pyrater[jax]"'
        ) from error


def dense_votes(data: AnnotationData) -> npt.NDArray[np.int64]:
    """(items, annotators) label codes, -1 where an annotator skipped an item.

    Raises:
        ValueError: if the data has no annotator identifiers, or if an
            annotator labelled the same item more than once. A dense matrix
            holds one vote per cell, so a repeat cannot be represented; it is
            refused rather than resolved by keeping whichever row came last.
    """
    if data.jj is None or data.annotator_ids is None:
        raise ValueError("NumPyro-backed models need annotator identifiers.")
    cells = data.ii * data.n_annotators + data.jj
    repeated, counts = np.unique(cells, return_counts=True)
    repeated = repeated[counts > 1]
    if repeated.size:
        pairs = [
            (data.item_ids[cell // data.n_annotators], data.annotator_ids[cell % data.n_annotators])
            for cell in repeated[:5].tolist()
        ]
        raise ValueError(
            f"{repeated.size} (item, annotator) pair(s) carry more than one label, e.g. {pairs}. "
            "NumPyro-backed models take at most one label per annotator and item; "
            "deduplicate the annotations first."
        )
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


def map_loss_converged(
    losses: npt.ArrayLike, *, window: int = _MAP_WINDOW, rtol: float = _MAP_RTOL
) -> bool:
    """Whether a loss trace has flattened out by its end.

    Compares the mean loss over the last `window` steps with the mean over the
    `window` steps before those; converged means they differ by at most `rtol`
    relative to the size of the later one. A trace shorter than two windows is
    split into halves instead. A trace of fewer than two steps, or one that is
    not finite at its end, is never converged.

    This is a check that the optimiser had stopped making visible progress, not
    a proof that it sits at the mode.
    """
    trace = np.asarray(losses, dtype=np.float64)
    window = min(window, trace.size // 2)
    if window < 1:
        return False
    last = trace[-window:].mean()
    before = trace[-2 * window : -window].mean()
    if not (np.isfinite(last) and np.isfinite(before)):
        return False
    return bool(abs(last - before) <= rtol * max(abs(last), np.finfo(np.float64).tiny))


class NumpyroBackedModel(PyRaterModel):
    """A `PyRaterModel` whose posterior comes from NumPyro.

    Subclasses implement `_priors`, which samples the parameters,
    `_item_loglik`, which returns the (items, classes) joint log density of each
    item's votes and each class, and `_confusion`, the annotators' confusion
    matrices as a function of the parameters. Fitting, the per-item class
    posterior and the pointwise log likelihood are shared.

    `fit(..., method="map")` returns the posterior mode as a single draw, in
    seconds; `method="nuts"` (the default) samples. See :meth:`fit`.

    Whatever the method, the posterior holds the model's own sampled sites
    plus two derived quantities: `q_z`, the (items, classes) posterior of each
    item's class given the parameters, and `theta`, one number per annotator
    (see `_competence`). The `log_likelihood` group holds `log_lik`, one value
    per item (the latent class is summed out, so the pointwise unit is the
    item, as in the Stan models).

    Attributes:
        seed_: The integer seed the last fit used; the one drawn at random if
            `seed=None` was passed, so that fit can be repeated.
        map_losses_: The loss (negative log joint density, up to a constant)
            at each optimisation step of the last `method="map"` fit; `None`
            before fitting and after a NUTS fit.
        map_converged_: Whether that loss had flattened out by the last step
            (see :func:`map_loss_converged`); `None` whenever `map_losses_` is.
    """

    requires_annotator_dim: ClassVar[bool] = True
    # Identified by a prior that favours the diagonal, so no relabelling pass.
    class_indexed_params: ClassVar[Mapping[str, int]] = {}
    # Whether `fit(groups=...)` means anything to this model.
    uses_groups: ClassVar[bool] = False

    # Class-level defaults, so a model unpickled from before these existed
    # still answers `None` rather than raising `AttributeError`.
    seed_: int | None = None
    map_losses_: npt.NDArray[np.float64] | None = None
    map_converged_: bool | None = None

    # Deliberately narrower than `PyRaterModel.fit`, which takes `**cmdstanpy_kwargs`:
    # every argument is named, so a misspelt one is a `TypeError`, not a no-op.
    def fit(  # ty: ignore[invalid-method-override]
        self,
        data: AnnotationData,
        *,
        gold: Mapping[Hashable, Hashable] | None = None,
        chains: int = 4,
        warmup: int = 1000,
        iters: int = 1000,
        seed: int | None = None,
        align: bool = True,
        check_rhat: bool = True,
        rhat_threshold: float = 1.01,
        sample_size_per_class: int = 5,
        rng: np.random.Generator | None = None,
        method: str = "nuts",
        groups: Mapping[Hashable, Hashable] | None = None,
        anchors: Mapping[Hashable, Hashable] | None = None,
        map_steps: int = 1500,
        learning_rate: float = 0.05,
        mcmc_kwargs: Mapping[str, Any] | None = None,
    ) -> Self:
        """Fit the model, by sampling or by finding the posterior mode.

        Unlike the Stan-backed models, this takes no free-form keyword
        arguments: a name not listed below raises `TypeError`.

        Args:
            data: The annotations. Needs annotator identifiers, and at most one
                label per (item, annotator) pair.
            gold: Has no effect on these models, and passing it emits a
                `UserWarning`. Elsewhere in pyrater gold labels pick the
                permutation of the class axis; these models fix that axis with
                a prior that favours agreement with the true class, declare no
                `class_indexed_params`, and so run no alignment. Gold labels
                attached to `data` are ignored the same way, silently. To let
                known labels inform the fit, pass them as `anchors`.
            chains: Number of NUTS chains, run one after another. Ignored
                under `method="map"`.
            warmup: NUTS warm-up iterations per chain. Ignored under
                `method="map"`.
            iters: NUTS draws kept per chain. Ignored under `method="map"`.
            seed: Seed of the JAX PRNG key. A given integer always reproduces
                the same fit. `None` draws a fresh seed every call; the one
                used is kept as `seed_`.
            align: Accepted for a uniform interface; there is nothing to align
                (see `gold`).
            check_rhat: Under NUTS, compute split-R-hat into `convergence_`
                and warn if it exceeds `rhat_threshold`. Under `method="map"`
                there is one draw and no R-hat, so the check is skipped and
                `convergence_` is `None` whatever is passed here; read
                `map_converged_` instead.
            rhat_threshold: The R-hat above which a parameter counts as not
                converged.
            sample_size_per_class: Accepted for a uniform interface; only gold
                alignment uses it.
            rng: Accepted for a uniform interface; only gold alignment uses
                it. It does not seed the fit: `seed` does.
            method: `"nuts"` to sample with the No-U-Turn sampler, or `"map"`
                for the posterior mode, found with Adam and returned as a
                posterior of one chain and one draw.
            groups: `{annotator_id: group_id}`, for models whose annotators
                are grouped (`np_nested`, which requires it and needs every
                annotator in `data` to have an entry). Ignored, with a
                debug-level log message, by models that do not group
                annotators, so one call can serve several models.
            anchors: `{item_id: class_label}` for items whose class is known.
                Each enters the likelihood as an observation of the class
                itself, so the annotators' votes on that item inform their
                parameters as votes on a known class, and the item's `q_z` is
                a point mass on it. An item or label that is not in `data`
                raises `ValueError`.
            map_steps: Number of Adam steps under `method="map"`. Ignored
                under NUTS.
            learning_rate: Adam's learning rate under `method="map"`. Ignored
                under NUTS.
            mcmc_kwargs: Further keyword arguments for `numpyro.infer.MCMC`
                under NUTS, e.g. `{"chain_method": "vectorized"}` or
                `{"thinning": 2}`. `num_warmup`, `num_samples` and
                `num_chains` are refused here: set them through `warmup`,
                `iters` and `chains`. Ignored under `method="map"`.

        Returns:
            The fitted model itself.

        Raises:
            ValueError: for an unknown `method`, a non-positive `map_steps`,
                data without annotator identifiers or with a repeated (item,
                annotator) pair, a grouped model fitted without `groups` or
                with annotators missing from it, or an unknown anchor.
            TypeError: if `mcmc_kwargs` names an argument that `fit` sets.
            ImportError: if JAX and NumPyro (the `jax` extra) are missing.

        Warns:
            UserWarning: if `gold` is passed.
            ConvergenceWarning: under NUTS, if `check_rhat` and some R-hat
                exceeds `rhat_threshold`; under `method="map"`, if the loss
                was still moving at the last step (`map_converged_` is
                `False`), whatever `check_rhat` says. Raise `map_steps` then.

        After a NUTS fit, `idata` also has a `sample_stats` group with, for
        every draw, `diverging`, `energy`, `n_steps`, `tree_depth`,
        `acceptance_rate`, `step_size` and `lp`. After a MAP fit, the loss
        trace is in `map_losses_`.

        Fitting needs 64-bit JAX: the summed log density of millions of votes
        loses digits in float32. The global `jax_enable_x64` flag is switched
        on for the duration of the fit and put back to what it was afterwards.
        The flag is process-wide, so other JAX code running in another thread
        during a fit sees it change.
        """
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        if map_steps < 1:
            raise ValueError(f"map_steps must be at least 1, got {map_steps!r}")
        reserved = sorted(set(mcmc_kwargs or {}) & set(_MCMC_RESERVED))
        if reserved:
            raise TypeError(
                "mcmc_kwargs may not set "
                + ", ".join(f"{name!r} (use `{_MCMC_RESERVED[name]}`)" for name in reserved)
            )
        _require_backend()

        if gold is not None and not self.class_indexed_params:
            warnings.warn(
                f"{self.name}: `gold` has no effect on NumPyro-backed models, which run no "
                "label alignment. Pass known labels as `anchors` to have them inform the fit.",
                UserWarning,
                stacklevel=2,
            )
        if groups is not None and not self.uses_groups:
            logger.debug("%s does not group annotators; ignoring `groups`.", self.name)

        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        self.seed_ = seed
        self.map_losses_ = None
        self.map_converged_ = None

        super().fit(
            data,
            gold=gold,
            chains=chains,
            warmup=warmup,
            iters=iters,
            seed=seed,
            align=align,
            # One draw has no R-hat; `map_converged_` is the MAP diagnostic.
            check_rhat=check_rhat and method == "nuts",
            rhat_threshold=rhat_threshold,
            sample_size_per_class=sample_size_per_class,
            rng=rng,
            method=method,
            groups=groups,
            anchors=anchors,
            map_steps=map_steps,
            learning_rate=learning_rate,
            mcmc_kwargs=mcmc_kwargs,
        )

        if self.map_converged_ is False:
            warnings.warn(
                f"{self.name}: the MAP loss was still changing after {map_steps} steps; the "
                "result may not be the posterior mode. Raise `map_steps` (or adjust "
                "`learning_rate`) and inspect `.map_losses_`.",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    @abc.abstractmethod
    def _priors(self, n_classes: int, tensors: dict[str, Any]) -> dict[str, Any]:
        """Sample every parameter with `numpyro.sample` and return them by name."""
        raise NotImplementedError

    @abc.abstractmethod
    def _item_loglik(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        """(items, classes): log P(class) + log P(the item's votes | class)."""
        raise NotImplementedError

    @abc.abstractmethod
    def _confusion(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        """(annotators, class, vote): P(annotator reports `vote` | true `class`)."""
        raise NotImplementedError

    def _competence(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        """One number per annotator, stored as `theta` in the posterior.

        By default the probability of reporting the true class, averaged over
        classes: the mean of the diagonal of the annotator's confusion matrix.
        """
        import jax.numpy as jnp

        confusion = self._confusion(params, tensors)
        return jnp.diagonal(confusion, axis1=1, axis2=2).mean(axis=-1)

    def _tensors(
        self, data: AnnotationData, groups: Mapping[Hashable, Hashable] | None
    ) -> dict[str, Any]:
        """Everything about the data that `_priors` and `_item_loglik` need."""
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
        mcmc_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        if kwargs:
            raise TypeError(f"Unexpected keyword argument(s): {sorted(kwargs)}")
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        _require_backend()
        import jax

        # The summed log density of 10^7 votes loses digits in float32. The
        # flag is global, so it is put back once the draws are NumPy arrays.
        x64_before = bool(jax.config.read("jax_enable_x64"))
        jax.config.update("jax_enable_x64", True)
        try:
            return self._sample_x64(
                data,
                chains=chains,
                warmup=warmup,
                iters=iters,
                seed=seed,
                method=method,
                groups=groups,
                anchors=anchors,
                map_steps=map_steps,
                learning_rate=learning_rate,
                mcmc_kwargs=dict(mcmc_kwargs or {}),
            )
        finally:
            jax.config.update("jax_enable_x64", x64_before)

    def _sample_x64(
        self,
        data: AnnotationData,
        *,
        chains: int,
        warmup: int,
        iters: int,
        seed: int | None,
        method: str,
        groups: Mapping[Hashable, Hashable] | None,
        anchors: Mapping[Hashable, Hashable] | None,
        map_steps: int,
        learning_rate: float,
        mcmc_kwargs: dict[str, Any],
    ) -> az.InferenceData:
        """`_sample`, with 64-bit JAX already switched on."""
        import jax
        import jax.numpy as jnp
        import numpyro
        from jax.scipy.special import logsumexp

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

        if seed is None:  # `fit` always resolves it; this is for direct callers.
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        key = jax.random.PRNGKey(seed)
        sample_stats: dict[str, npt.NDArray[Any]] | None = None
        if method == "map":
            draws = self._run_map(model, key, map_steps=map_steps, learning_rate=learning_rate)
        else:
            draws, sample_stats = self._run_nuts(
                model, key, chains=chains, warmup=warmup, iters=iters, mcmc_kwargs=mcmc_kwargs
            )

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
        groups_out: dict[str, Any] = {
            "posterior": posterior,
            "log_likelihood": {"log_lik": shaped(log_lik)},
        }
        if sample_stats is not None:
            groups_out["sample_stats"] = sample_stats
        return az.from_dict(**groups_out)

    def _run_map(
        self, model: Any, key: Any, *, map_steps: int, learning_rate: float
    ) -> dict[str, npt.NDArray[Any]]:
        """The posterior mode as a (1 chain, 1 draw) posterior; records the loss trace."""
        import numpyro
        from numpyro.infer import SVI, Trace_ELBO, init_to_median
        from numpyro.infer.autoguide import AutoDelta

        guide = AutoDelta(model, init_loc_fn=init_to_median)
        svi = SVI(model, guide, numpyro.optim.Adam(learning_rate), Trace_ELBO())
        result = svi.run(key, map_steps, progress_bar=False)
        point = guide.median(result.params)
        self.map_losses_ = np.asarray(result.losses, dtype=np.float64)
        self.map_converged_ = map_loss_converged(self.map_losses_)
        return {name: np.asarray(value)[None, None] for name, value in point.items()}

    def _run_nuts(
        self,
        model: Any,
        key: Any,
        *,
        chains: int,
        warmup: int,
        iters: int,
        mcmc_kwargs: dict[str, Any],
    ) -> tuple[dict[str, npt.NDArray[Any]], dict[str, npt.NDArray[Any]]]:
        """NUTS draws and per-draw sampler statistics, both (chains, draws, ...)."""
        from numpyro.infer import MCMC, NUTS, init_to_median

        mcmc = MCMC(
            NUTS(model, init_strategy=init_to_median),
            num_warmup=warmup,
            num_samples=iters,
            num_chains=chains,
            **{"chain_method": "sequential", "progress_bar": False, **mcmc_kwargs},
        )
        # Recording these does not touch the sampler's state or its draws.
        mcmc.run(key, extra_fields=(*_SAMPLE_STATS, "potential_energy"))
        draws = {
            name: np.asarray(value) for name, value in mcmc.get_samples(group_by_chain=True).items()
        }
        extra = {
            name: np.asarray(value)
            for name, value in mcmc.get_extra_fields(group_by_chain=True).items()
        }
        sample_stats = {_SAMPLE_STATS[name]: extra[name] for name in _SAMPLE_STATS}
        # NumPyro reports leapfrog steps, not depth. A tree of depth d takes
        # 2^(d-1) to 2^d - 1 of them (the last doubling may stop part-way);
        # this is the conversion ArviZ's own NumPyro reader uses.
        sample_stats["tree_depth"] = np.log2(sample_stats["n_steps"]).astype(np.int64) + 1
        sample_stats["lp"] = -extra["potential_energy"]
        return draws, sample_stats

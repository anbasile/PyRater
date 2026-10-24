"""`np_stan_d&s`, `np_stan_mace` and `np_stan_hier_d&s` are the Stan models.

Two references: the bundled Stan files, whose log density must differ from
ours by a constant (Stan's `~` drops normalising constants), and NumPyro's own
annotation example, whose log density must equal ours. Then what alignment
does to them: their priors are symmetric in the classes, as in Stan.
"""

from __future__ import annotations

from math import lgamma
from typing import Any

import cmdstanpy
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp as np_logsumexp

jax = pytest.importorskip("jax")
pytest.importorskip("numpyro")

import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
import numpyro_annotation_example as example  # noqa: E402
from jax.scipy.special import logsumexp  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402

from pyrater.data.annotation_data import AnnotationData  # noqa: E402
from pyrater.models._numpyro_backed import NumpyroBackedModel, _x64  # noqa: E402
from pyrater.models.base import PyRaterModel  # noqa: E402
from pyrater.stan import get_compiled_model  # noqa: E402

TWINS = {
    "np_stan_d&s": ("d&s", "DS.stan", "dawid_skene"),
    "np_stan_mace": ("mace", "MACE.stan", "mace"),
    "np_stan_hier_d&s": ("hier_d&s", "HDS-NC.stan", "hierarchical_dawid_skene"),
}


def _complete(n_items: int = 40, n_annotators: int = 4, n_classes: int = 3, seed: int = 0):
    """Every annotator labels every item once: the layout the example takes."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(n_classes, size=n_items)
    noise = rng.integers(n_classes, size=(n_items, n_annotators))
    votes = np.where(rng.random((n_items, n_annotators)) < 0.7, truth[:, None], noise)
    frame = pd.DataFrame(
        [
            (f"i{i}", f"a{j}", f"c{votes[i, j]}")
            for i in range(n_items)
            for j in range(n_annotators)
        ],
        columns=["item_id", "annotator_id", "label"],
    )
    data = AnnotationData.from_long_df(frame)
    codes = np.full((n_items, n_annotators), -1)
    codes[data.ii, data.jj] = data.y
    gold = {f"i{i}": f"c{c}" for i, c in enumerate(truth)}
    return data, codes, gold


def _numpyro_model(model: NumpyroBackedModel, data: AnnotationData) -> Any:
    """The model `fit` samples from, as a NumPyro program."""
    tensors = model._tensors(data, None)

    def program() -> None:
        params = model._priors(data.n_classes, tensors)
        loglik = model._item_loglik(params, tensors)
        numpyro.factor("votes", logsumexp(loglik, axis=-1).sum())

    return program


def _prior_points(program: Any, n: int) -> list[dict[str, np.ndarray]]:
    """`n` draws from the prior, constrained, as the parameter dicts of a fit."""
    points = []
    for s in range(n):
        seeded = numpyro.handlers.seed(program, jax.random.PRNGKey(s))
        trace = numpyro.handlers.trace(seeded).get_trace()
        points.append(
            {
                k: np.asarray(v["value"])
                for k, v in trace.items()
                if v["type"] == "sample" and not v["is_observed"]
            }
        )
    return points


def _ours(name: str, data: AnnotationData, n: int = 5) -> tuple[list[dict[str, Any]], list[float]]:
    with _x64():
        program = _numpyro_model(PyRaterModel.get(name), data)  # ty: ignore[invalid-argument-type]
        points = _prior_points(program, n)
        return points, [float(log_density(program, (), {}, p)[0]) for p in points]


@pytest.mark.stan
@pytest.mark.parametrize("name", TWINS)
def test_log_density_is_stans_up_to_a_constant(name: str) -> None:
    try:
        cmdstanpy.cmdstan_path()
    except Exception:
        pytest.skip("CmdStan is not installed")
    data, _, _ = _complete()
    points, ours = _ours(name, data)
    stan = get_compiled_model(TWINS[name][1])
    stan_data = dict(data.to_stan_data(include_annotators=True))
    theirs = [
        float(stan.log_prob(p, data=stan_data, jacobian=False)["lp__"].iloc[0]) for p in points
    ]
    offsets = np.subtract(ours, theirs)
    # Stan prints lp__ to six significant digits.
    assert np.ptp(offsets) < 1e-4, offsets


def test_the_dropped_constant_of_dawid_skene_is_the_dirichlet_normaliser() -> None:
    """What Stan drops is known for `d&s`: log Gamma(K) for every Dirichlet(1)
    row, of which there are J * K + 1. Ours keeps it, so subtracting it must
    leave the log density Stan reports, which this pins without CmdStan."""
    data, codes, _ = _complete(n_items=12, n_annotators=3, n_classes=3)
    points, ours = _ours("np_stan_d&s", data, n=2)
    k = data.n_classes
    for point, lp in zip(points, ours, strict=True):
        beta, pi = point["beta"], point["pi"]
        joint = np.log(pi)[None] + sum(
            np.log(beta[j][:, codes[:, j]]).T for j in range(codes.shape[1])
        )
        stan = float(np_logsumexp(joint, axis=1).sum())
        assert lp - stan == pytest.approx((codes.shape[1] * k + 1) * lgamma(k), abs=1e-9)


@pytest.mark.parametrize("name", TWINS)
def test_log_density_equals_numpyros_annotation_example(name: str) -> None:
    pytest.importorskip("funsor")
    from numpyro.contrib.funsor import config_enumerate, enum  # noqa: PLC0415 -- needs funsor
    from numpyro.contrib.funsor import log_density as enum_log_density  # noqa: PLC0415

    data, codes, _ = _complete()
    points, ours = _ours(name, data)
    program = enum(config_enumerate(getattr(example, TWINS[name][2])), first_available_dim=-3)
    args = (jnp.arange(codes.shape[1]), jnp.asarray(codes))
    for point, lp in zip(points, ours, strict=True):
        # The example's LocScaleReparam names the offsets `beta_decentered`.
        renamed = {("beta_decentered" if k == "beta_raw" else k): v for k, v in point.items()}
        with _x64():
            theirs = float(enum_log_density(program, args, {}, renamed)[0])
        assert lp == pytest.approx(theirs, abs=1e-9)


class TestRegistry:
    @pytest.mark.parametrize("name", TWINS)
    def test_registered_backed_and_documented(self, name: str) -> None:
        model = PyRaterModel.get(name)
        assert isinstance(model, NumpyroBackedModel)
        assert "Generative model" in (type(model).__doc__ or "")

    @pytest.mark.parametrize("name", TWINS)
    def test_aligned_exactly_where_the_stan_model_is(self, name: str) -> None:
        stan = PyRaterModel.get(TWINS[name][0])
        assert bool(PyRaterModel.get(name).class_indexed_params) == bool(stan.class_indexed_params)


class TestAlignment:
    def test_theta_is_recomputed_from_the_aligned_draws(self) -> None:
        """Swap the class axis of a fit by hand; `_after_alignment` must make
        `theta` the mean diagonal of the swapped matrices, not keep the old one."""
        data, _, gold = _complete(n_items=60, n_classes=2, seed=1)
        model = PyRaterModel.get("np_stan_d&s").fit(
            data, chains=1, warmup=100, iters=50, seed=0, gold=gold, check_rhat=False
        )
        assert isinstance(model, NumpyroBackedModel)
        idata = model.idata
        posterior = idata["posterior"]
        before = posterior["theta"].values.copy()
        # The values, not `isel`: assigning a reindexed array back realigns it.
        posterior["beta"].values[:] = posterior["beta"].values[..., ::-1, :]
        model._after_alignment(idata, data)
        after = posterior["theta"].values
        beta = posterior["beta"].values
        assert np.allclose(after, np.diagonal(beta, axis1=-2, axis2=-1).mean(axis=-1))
        assert not np.allclose(after, before)

    @pytest.mark.parametrize("name", TWINS)
    def test_a_fit_with_gold_finds_the_classes(self, name: str) -> None:
        data, _, gold = _complete(n_items=150, n_annotators=6, seed=2)
        model = PyRaterModel.get(name)
        anchor = {"gold": gold} if model.class_indexed_params else {}
        model.fit(data, chains=2, warmup=200, iters=200, seed=0, check_rhat=False, **anchor)
        truth = np.array([gold[i] for i in data.item_ids])
        assert (model.labels == truth).mean() > 0.9
        theta = model.idata["posterior"]["theta"].values
        assert theta.shape == (2, 200, data.n_annotators)
        assert (theta.mean(axis=(0, 1)) > 0.4).all()

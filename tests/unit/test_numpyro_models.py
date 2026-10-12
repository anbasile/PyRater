"""The NumPyro backend: what it recovers from simulated votes, what `fit`
accepts and refuses, and which diagnostics it keeps."""

from __future__ import annotations

import logging
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("numpyro")

from pyrater.data.annotation_data import AnnotationData  # noqa: E402
from pyrater.diagnostics.convergence import check_convergence  # noqa: E402
from pyrater.exceptions import ConvergenceWarning  # noqa: E402
from pyrater.models._numpyro_backed import (  # noqa: E402
    NumpyroBackedModel,
    dense_votes,
    map_loss_converged,
)
from pyrater.models.base import PyRaterModel  # noqa: E402

NAMES = ["np_d&s", "np_mace", "np_hier_d&s", "np_nested"]
# What the NUTS tests hold R-hat to. Two chains of 400 draws on the small
# problem stayed below 1.03 for every model on each of six seeds; the slack is
# for platforms whose floating point sends the chains elsewhere.
RHAT = 1.05
SAMPLE_STATS = {
    "diverging",
    "energy",
    "n_steps",
    "tree_depth",
    "acceptance_rate",
    "step_size",
    "lp",
}


def _simulate(n_items=600, n_annotators=12, n_classes=3, seed=0, missing=0.0):
    rng = np.random.default_rng(seed)
    truth = rng.integers(n_classes, size=n_items)
    accuracy = np.linspace(0.45, 0.9, n_annotators)
    rows = []
    for j in range(n_annotators):
        correct = rng.random(n_items) < accuracy[j]
        noise = rng.integers(n_classes, size=n_items)
        seen = rng.random(n_items) >= missing
        for i, label in enumerate(np.where(correct, truth, noise)):
            if seen[i]:
                rows.append((f"i{i}", f"a{j}", f"c{label}"))
    frame = pd.DataFrame(rows, columns=["item_id", "annotator_id", "label"])
    gold = {f"i{i}": f"c{c}" for i, c in enumerate(truth)}
    return AnnotationData.from_long_df(frame), gold, accuracy


def _accuracy(model, data, gold):
    return float(
        np.mean([gold[i] == label for i, label in zip(data.item_ids, model.labels, strict=True)])
    )


def _own_groups(data: AnnotationData) -> dict[Any, Any]:
    """Every annotator a group of its own: what `np_nested` needs at the least."""
    assert data.annotator_ids is not None
    return {a: a for a in data.annotator_ids}


def _get(name: str) -> NumpyroBackedModel:
    model = PyRaterModel.get(name)
    assert isinstance(model, NumpyroBackedModel)
    return model


def _posterior(model: PyRaterModel, name: str) -> np.ndarray:
    return np.asarray(model.idata["posterior"][name].values)


@pytest.fixture(scope="module")
def small() -> tuple[AnnotationData, dict[Any, Any]]:
    """Small enough that a fit is dominated by compilation, not by the data."""
    data, gold, _ = _simulate(n_items=60, n_annotators=6, seed=4)
    return data, gold


@pytest.fixture(scope="module")
def sparse() -> tuple[AnnotationData, dict[Any, Any]]:
    """Four cells in ten are empty."""
    data, gold, _ = _simulate(n_items=150, n_annotators=8, seed=5, missing=0.4)
    return data, gold


@pytest.fixture(scope="module")
def plain_map(small: tuple[AnnotationData, dict[Any, Any]]) -> NumpyroBackedModel:
    """`np_d&s` at its mode with every argument at its default: the fit that
    the "this argument changes nothing" tests compare against."""
    data, _ = small
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # in particular, no R-hat check on one draw
        return _get("np_d&s").fit(data, method="map", seed=0)


@pytest.mark.parametrize("name", ["np_d&s", "np_mace", "np_hier_d&s"])
def test_map_recovers_labels_and_ranks_annotators(name):
    data, gold, accuracy = _simulate()
    model = PyRaterModel.get(name).fit(data, method="map", check_rhat=False, seed=1)
    assert _accuracy(model, data, gold) > 0.93
    theta = model.idata["posterior"]["theta"].to_numpy()[0, 0]
    assert np.corrcoef(theta, accuracy)[0, 1] > 0.9
    assert model.labels_proba.shape == (data.n_items, data.n_classes)
    assert model.idata["log_likelihood"]["log_lik"].shape == (1, 1, data.n_items)


def test_nuts_agrees_with_map():
    data, gold, _ = _simulate(n_items=200, n_annotators=8)
    point = PyRaterModel.get("np_d&s").fit(data, method="map", check_rhat=False, seed=1)
    full = PyRaterModel.get("np_d&s").fit(data, chains=2, warmup=200, iters=200, seed=1)
    assert full.convergence_ is not None
    assert np.abs(point.labels_proba - full.labels_proba).mean() < 0.05
    assert (point.labels == full.labels).mean() > 0.97


def test_nested_treats_a_group_as_one_opinion():
    """Twenty copies of one mediocre annotator must not outvote three good ones."""
    rng = np.random.default_rng(3)
    n_items, n_classes = 500, 2
    truth = rng.integers(n_classes, size=n_items)

    def noisy(source, accuracy):
        flip = rng.random(n_items) > accuracy
        return np.where(flip, 1 - source, source)

    opinion = noisy(truth, 0.6)
    votes = {f"copy{j}": noisy(opinion, 0.97) for j in range(20)}
    votes |= {f"good{j}": noisy(truth, 0.85) for j in range(3)}
    rows = [(f"i{i}", a, f"c{v[i]}") for a, v in votes.items() for i in range(n_items)]
    data = AnnotationData.from_long_df(
        pd.DataFrame(rows, columns=["item_id", "annotator_id", "label"])
    )
    gold = {f"i{i}": f"c{c}" for i, c in enumerate(truth)}
    groups = {a: ("clone" if a.startswith("copy") else a) for a in votes}

    vote = PyRaterModel.get("majority_voting").fit(data)
    nested = PyRaterModel.get("np_nested").fit(
        data, method="map", groups=groups, check_rhat=False, seed=0
    )
    assert _accuracy(vote, data, gold) < 0.7
    assert _accuracy(nested, data, gold) > 0.85


def test_anchors_fix_the_known_items():
    data, gold, _ = _simulate(n_items=150, n_annotators=6)
    wrong = {item: "c0" for item in list(gold)[:10]}
    model = PyRaterModel.get("np_d&s").fit(
        data, method="map", anchors=wrong, check_rhat=False, seed=0
    )
    index = {item: n for n, item in enumerate(data.item_ids)}
    assert all(model.labels[index[item]] == "c0" for item in wrong)


def test_nested_groups_of_one_kind_share_a_confusion_matrix():
    data, gold, _ = _simulate(n_items=200, n_annotators=6)
    assert data.annotator_ids is not None
    groups = {a: ("crowd", a) for a in data.annotator_ids}
    model = PyRaterModel.get("np_nested").fit(
        data, method="map", groups=groups, check_rhat=False, seed=0
    )
    # Six groups, one kind: one shared matrix at the group level.
    assert model.idata["posterior"]["kind_confusion"].shape[2] == 1
    assert _accuracy(model, data, gold) > 0.8


class TestRegistry:
    def test_the_four_models_are_registered(self) -> None:
        assert set(NAMES) <= set(PyRaterModel.list_available())

    @pytest.mark.parametrize("name", NAMES)
    def test_each_is_numpyro_backed_and_documented(self, name: str) -> None:
        model = PyRaterModel.get(name)
        assert isinstance(model, NumpyroBackedModel)
        assert model.name == name
        assert "Generative model" in (type(model).__doc__ or "")

    def test_only_the_nested_model_uses_groups(self) -> None:
        assert {name for name in NAMES if _get(name).uses_groups} == {"np_nested"}


class TestSparseVotes:
    def test_dense_votes_marks_missing_cells(self, sparse: tuple[AnnotationData, Any]) -> None:
        data, _ = sparse
        votes = dense_votes(data)
        assert votes.shape == (data.n_items, data.n_annotators)
        assert (votes == -1).sum() == data.n_items * data.n_annotators - data.n_annotations
        assert data.jj is not None
        assert np.array_equal(votes[data.ii, data.jj], data.y)

    @pytest.mark.parametrize("name", NAMES)
    def test_map_fits_a_matrix_with_holes(
        self, name: str, sparse: tuple[AnnotationData, dict[Any, Any]]
    ) -> None:
        data, gold = sparse
        model = _get(name).fit(data, method="map", seed=0, groups=_own_groups(data))
        assert _accuracy(model, data, gold) > 0.85
        assert np.isfinite(np.asarray(model.idata["log_likelihood"]["log_lik"].values)).all()
        assert np.allclose(model.labels_proba.sum(axis=1), 1.0, atol=1e-5)
        assert _posterior(model, "theta").shape == (1, 1, data.n_annotators)

    def test_an_item_nobody_in_a_group_labelled_is_left_to_the_others(self) -> None:
        # "a0" skips the second half of the items; it is a group of its own, so
        # on those items its group has no reading at all.
        data, gold, _ = _simulate(n_items=80, n_annotators=5, seed=6)
        frame = data.to_long_df()
        late = {f"i{i}" for i in range(40, 80)}
        frame = frame[~((frame["annotator_id"] == "a0") & frame["item_id"].isin(late))]
        holed = AnnotationData.from_long_df(frame.reset_index(drop=True))
        model = _get("np_nested").fit(holed, method="map", seed=0, groups=_own_groups(holed))
        assert np.isfinite(model.labels_proba).all()
        assert _accuracy(model, holed, gold) > 0.85


class TestFitArguments:
    def test_an_unknown_keyword_raises(self, small: tuple[AnnotationData, Any]) -> None:
        data, _ = small
        misspelt: dict[str, Any] = {"metod": "map"}
        with pytest.raises(TypeError, match="metod"):
            _get("np_d&s").fit(data, **misspelt)

    def test_mcmc_arguments_are_not_accepted_as_bare_keywords(
        self, small: tuple[AnnotationData, Any]
    ) -> None:
        data, _ = small
        bare: dict[str, Any] = {"thinning": 2}
        with pytest.raises(TypeError, match="thinning"):
            _get("np_d&s").fit(data, **bare)

    def test_an_unknown_method_raises(self, small: tuple[AnnotationData, Any]) -> None:
        data, _ = small
        with pytest.raises(ValueError, match="'nuts', 'map'.*'vi'"):
            _get("np_d&s").fit(data, method="vi")

    def test_map_steps_must_be_positive(self, small: tuple[AnnotationData, Any]) -> None:
        data, _ = small
        with pytest.raises(ValueError, match="map_steps"):
            _get("np_d&s").fit(data, method="map", map_steps=0)

    def test_mcmc_kwargs_may_not_set_what_fit_sets(self, small: tuple[AnnotationData, Any]) -> None:
        data, _ = small
        with pytest.raises(TypeError, match="num_chains.*use `chains`"):
            _get("np_d&s").fit(data, mcmc_kwargs={"num_chains": 2})

    def test_mcmc_kwargs_reach_the_sampler(self, small: tuple[AnnotationData, Any]) -> None:
        data, _ = small
        model = _get("np_mace").fit(
            data,
            chains=1,
            warmup=100,
            iters=20,
            seed=0,
            check_rhat=False,
            mcmc_kwargs={"thinning": 2},
        )
        assert _posterior(model, "competence").shape[:2] == (1, 10)

    @pytest.mark.parametrize("name", ["np_d&s", "np_mace", "np_hier_d&s"])
    def test_groups_are_ignored_by_models_without_groups(
        self,
        name: str,
        small: tuple[AnnotationData, Any],
        plain_map: NumpyroBackedModel,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        data, _ = small
        with caplog.at_level(logging.DEBUG, logger="pyrater.models._numpyro_backed"):
            grouped = _get(name).fit(data, method="map", seed=0, groups=_own_groups(data))
        assert "ignoring `groups`" in caplog.text
        if name == plain_map.name:
            assert np.array_equal(plain_map.labels_proba, grouped.labels_proba)

    def test_sampler_sizes_are_ignored_under_map(
        self, small: tuple[AnnotationData, Any], plain_map: NumpyroBackedModel
    ) -> None:
        data, _ = small
        sized = _get("np_d&s").fit(
            data, method="map", seed=0, chains=1, warmup=1, iters=1, mcmc_kwargs={"thinning": 3}
        )
        assert _posterior(sized, "confusion").shape[:2] == (1, 1)
        assert np.array_equal(plain_map.labels_proba, sized.labels_proba)

    def test_gold_is_ignored_with_a_warning(
        self, small: tuple[AnnotationData, dict[Any, Any]], plain_map: NumpyroBackedModel
    ) -> None:
        data, gold = small
        with pytest.warns(UserWarning, match="`gold` has no effect"):
            with_gold = _get("np_d&s").fit(data, method="map", seed=0, gold=gold)
        assert np.array_equal(plain_map.labels_proba, with_gold.labels_proba)

    def test_a_missing_backend_names_the_extra(
        self, small: tuple[AnnotationData, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data, _ = small
        # `None` in `sys.modules` makes the import statement raise ImportError.
        monkeypatch.setitem(sys.modules, "numpyro", None)
        with pytest.raises(ImportError, match=r"pyrater\[jax\]"):
            _get("np_d&s").fit(data, method="map")


class TestDataErrors:
    def test_a_repeated_item_annotator_pair_raises(self) -> None:
        frame = pd.DataFrame(
            {
                "item_id": ["i0", "i0", "i0", "i1", "i1"],
                "annotator_id": ["a0", "a1", "a0", "a0", "a1"],
                "label": ["yes", "no", "no", "no", "no"],
            }
        )
        data = AnnotationData.from_long_df(frame)
        with pytest.raises(ValueError, match=r"more than one label.*'i0', 'a0'"):
            dense_votes(data)
        with pytest.raises(ValueError, match="more than one label"):
            _get("np_d&s").fit(data, method="map")

    def test_data_without_annotators_raises(self) -> None:
        data = AnnotationData(
            ii=np.array([0, 0, 1], dtype=np.int64),
            jj=None,
            y=np.array([0, 1, 1], dtype=np.int64),
            item_ids=np.array(["i0", "i1"], dtype=object),
            annotator_ids=None,
            class_labels=np.array(["no", "yes"], dtype=object),
        )
        for name in ("np_d&s", "np_nested"):
            with pytest.raises(ValueError, match="annotator identifiers"):
                _get(name).fit(data, method="map", groups={})

    def test_nested_without_groups_raises(self, small: tuple[AnnotationData, Any]) -> None:
        data, _ = small
        with pytest.raises(ValueError, match="needs `groups`"):
            _get("np_nested").fit(data, method="map")

    def test_nested_names_the_annotators_missing_from_groups(
        self, small: tuple[AnnotationData, Any]
    ) -> None:
        data, _ = small
        groups = _own_groups(data)
        del groups["a1"], groups["a4"]
        with pytest.raises(ValueError, match=r"no entry for 2 of the 6 annotator\(s\).*'a1', 'a4'"):
            _get("np_nested").fit(data, method="map", groups=groups)

    def test_groups_may_mention_annotators_that_are_not_in_the_data(
        self, small: tuple[AnnotationData, Any]
    ) -> None:
        data, _ = small
        groups = _own_groups(data) | {"somebody else": "elsewhere"}
        _get("np_nested").fit(data, method="map", seed=0, map_steps=5, groups=groups)

    @pytest.mark.parametrize(
        ("anchors", "message"),
        [({"no such item": "c0"}, "item 'no such item'"), ({"i0": "c9"}, "class 'c9'")],
    )
    def test_an_unknown_anchor_raises(
        self, anchors: dict[Any, Any], message: str, small: tuple[AnnotationData, Any]
    ) -> None:
        data, _ = small
        with pytest.raises(ValueError, match=message):
            _get("np_d&s").fit(data, method="map", anchors=anchors)


class TestMapDiagnostics:
    def test_the_rhat_check_is_skipped_and_the_loss_trace_kept(
        self, plain_map: NumpyroBackedModel
    ) -> None:
        # `plain_map` left `check_rhat` on and was fitted with warnings as errors.
        model = plain_map
        assert model.convergence_ is None
        assert model.map_converged_ is True
        assert model.map_losses_ is not None
        assert model.map_losses_.shape == (1500,)
        assert np.isfinite(model.map_losses_).all()
        assert model.map_losses_[-1] < model.map_losses_[0]
        assert "sample_stats" not in model.idata.groups()

    def test_a_run_stopped_early_warns(self, small: tuple[AnnotationData, Any]) -> None:
        data, _ = small
        with pytest.warns(ConvergenceWarning, match="still changing after 30 steps"):
            model = _get("np_hier_d&s").fit(
                data, method="map", seed=0, map_steps=30, check_rhat=False
            )
        assert model.map_converged_ is False
        assert model.map_losses_ is not None
        assert model.map_losses_.shape == (30,)
        # The result is still there: convergence is reported, not enforced.
        assert model.labels_proba.shape == (data.n_items, data.n_classes)

    def test_a_nuts_fit_clears_the_map_diagnostics(self, small: tuple[AnnotationData, Any]) -> None:
        data, _ = small
        model = _get("np_mace")
        model.fit(data, method="map", seed=0)
        assert model.map_losses_ is not None
        model.fit(data, chains=1, warmup=10, iters=10, seed=0, check_rhat=False)
        assert model.map_losses_ is None
        assert model.map_converged_ is None

    def test_flat_and_moving_traces(self) -> None:
        steps = np.arange(400.0)
        assert map_loss_converged(np.full(400, 123.0))
        assert map_loss_converged(100.0 + np.exp(-steps / 10))
        assert not map_loss_converged(1000.0 - steps)
        # Jitter around a level is not movement.
        assert map_loss_converged(100.0 + 1e-3 * np.cos(steps))

    def test_short_and_broken_traces(self) -> None:
        assert not map_loss_converged([])
        assert not map_loss_converged([5.0])
        assert map_loss_converged([5.0, 5.0])
        assert not map_loss_converged([9.0, 5.0])
        assert not map_loss_converged(np.r_[np.full(300, 7.0), np.nan])


@pytest.fixture(scope="module", params=NAMES)
def sampled(
    request: pytest.FixtureRequest, small: tuple[AnnotationData, dict[Any, Any]]
) -> tuple[NumpyroBackedModel, AnnotationData, dict[Any, Any]]:
    """Each model sampled once, for every test in `TestNuts`."""
    data, gold = small
    kwargs: dict[str, Any] = {}
    if request.param == "np_nested":
        ids = [] if data.annotator_ids is None else list(data.annotator_ids)
        kwargs["groups"] = {a: ("first" if a in ids[:3] else a) for a in ids}
    model = _get(request.param).fit(
        data, chains=2, warmup=300, iters=400, seed=0, rhat_threshold=RHAT, **kwargs
    )
    return model, data, gold


class TestNuts:
    def test_rhat_of_every_sampled_site(self, sampled: tuple[NumpyroBackedModel, Any, Any]) -> None:
        model, _, _ = sampled
        sites = [
            str(name) for name in model.idata["posterior"].data_vars if name not in ("q_z", "theta")
        ]
        report = check_convergence(model.idata, var_names=sites, threshold=RHAT)
        assert report.ok, report.summary()
        assert max(float(np.nanmax(values)) for values in report.rhat.values()) < RHAT
        # `fit` ran the same check, over the derived quantities as well.
        assert model.convergence_ is not None
        assert model.convergence_.threshold == RHAT
        assert model.convergence_.ok, model.convergence_.summary()

    def test_sample_stats_are_kept(self, sampled: tuple[NumpyroBackedModel, Any, Any]) -> None:
        model, _, _ = sampled
        assert "sample_stats" in model.idata.groups()
        stats = model.idata["sample_stats"]
        assert set(stats.data_vars) >= SAMPLE_STATS
        for name in SAMPLE_STATS:
            assert stats[name].shape == (2, 400), name
            assert np.isfinite(np.asarray(stats[name].values, dtype=np.float64)).all(), name
        assert stats["diverging"].dtype == bool
        assert int(stats["diverging"].sum()) <= 0.01 * stats["diverging"].size
        n_steps = np.asarray(stats["n_steps"].values)
        depth = np.asarray(stats["tree_depth"].values)
        assert ((2 ** (depth - 1) <= n_steps) & (n_steps < 2**depth)).all()
        acceptance = np.asarray(stats["acceptance_rate"].values)
        assert ((acceptance >= 0) & (acceptance <= 1)).all()

    def test_posterior_shapes_and_labels(
        self, sampled: tuple[NumpyroBackedModel, AnnotationData, dict[Any, Any]]
    ) -> None:
        model, data, gold = sampled
        assert _posterior(model, "q_z").shape == (2, 400, data.n_items, data.n_classes)
        assert _posterior(model, "theta").shape == (2, 400, data.n_annotators)
        assert model.idata["log_likelihood"]["log_lik"].shape == (2, 400, data.n_items)
        assert _accuracy(model, data, gold) > 0.85
        assert model.map_losses_ is None
        assert model.map_converged_ is None


class TestSeeds:
    def test_an_integer_seed_reproduces_a_map_fit(
        self, small: tuple[AnnotationData, Any], plain_map: NumpyroBackedModel
    ) -> None:
        data, _ = small
        again = _get("np_d&s").fit(data, method="map", seed=0)
        assert plain_map.seed_ == again.seed_ == 0
        assert np.array_equal(_posterior(plain_map, "confusion"), _posterior(again, "confusion"))
        assert plain_map.map_losses_ is not None
        assert again.map_losses_ is not None
        assert np.array_equal(plain_map.map_losses_, again.map_losses_)

    def test_no_seed_draws_one_that_reproduces_the_fit(
        self, small: tuple[AnnotationData, Any]
    ) -> None:
        data, _ = small
        sizes: dict[str, Any] = {"chains": 1, "warmup": 100, "iters": 20, "check_rhat": False}
        first = _get("np_mace").fit(data, **sizes)
        second = _get("np_mace").fit(data, **sizes)
        assert isinstance(first.seed_, int)
        assert isinstance(second.seed_, int)
        assert first.seed_ != second.seed_
        assert not np.array_equal(_posterior(first, "competence"), _posterior(second, "competence"))
        # The recorded seed, passed back as an integer, gives the same draws.
        again = _get("np_mace").fit(data, seed=first.seed_, **sizes)
        assert np.array_equal(_posterior(first, "competence"), _posterior(again, "competence"))


class TestAnchorsWithTheNestedModel:
    def test_anchors_fix_the_known_items(self, small: tuple[AnnotationData, Any]) -> None:
        data, gold = small
        groups = _own_groups(data) | {"a0": "pair", "a1": "pair"}
        free = _get("np_nested").fit(data, method="map", seed=0, groups=groups)
        index = {item: n for n, item in enumerate(data.item_ids)}
        # Anchor five items to a class the unanchored fit does not give them.
        classes = [str(c) for c in data.class_labels]
        wrong = {
            item: next(c for c in classes if c != free.labels[index[item]])
            for item in list(data.item_ids)[:5]
        }
        anchored = _get("np_nested").fit(
            data, method="map", seed=0, groups=groups, anchors=wrong, check_rhat=False
        )
        for item, label in wrong.items():
            assert anchored.labels[index[item]] == label
            assert anchored.labels_proba[index[item]].max() > 0.999
        # The anchors moved the fit; they did not just overwrite five rows.
        assert not np.array_equal(
            _posterior(free, "kind_confusion"), _posterior(anchored, "kind_confusion")
        )


class TestSixtyFourBitFlag:
    @pytest.mark.parametrize("before", [False, True])
    def test_the_global_flag_is_put_back(
        self, before: bool, small: tuple[AnnotationData, Any]
    ) -> None:
        data, _ = small
        original = bool(jax.config.read("jax_enable_x64"))
        jax.config.update("jax_enable_x64", before)
        try:
            model = _get("np_d&s").fit(data, method="map", seed=0, map_steps=5, check_rhat=False)
            assert bool(jax.config.read("jax_enable_x64")) is before
            # The fit itself ran in 64 bits whatever the flag said.
            assert _posterior(model, "confusion").dtype == np.float64
        finally:
            jax.config.update("jax_enable_x64", original)

    def test_the_flag_is_put_back_when_the_fit_fails(
        self, small: tuple[AnnotationData, Any]
    ) -> None:
        data, _ = small
        before = bool(jax.config.read("jax_enable_x64"))
        with pytest.raises(ValueError, match="needs `groups`"):
            _get("np_nested").fit(data, method="map")
        assert bool(jax.config.read("jax_enable_x64")) is before

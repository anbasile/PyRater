"""The NumPyro backend recovers what it should from simulated votes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numpyro")

from pyrater.data.annotation_data import AnnotationData  # noqa: E402
from pyrater.models.base import PyRaterModel  # noqa: E402


def _simulate(n_items=600, n_annotators=12, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.integers(n_classes, size=n_items)
    accuracy = np.linspace(0.45, 0.9, n_annotators)
    rows = []
    for j in range(n_annotators):
        correct = rng.random(n_items) < accuracy[j]
        noise = rng.integers(n_classes, size=n_items)
        for i, label in enumerate(np.where(correct, truth, noise)):
            rows.append((f"i{i}", f"a{j}", f"c{label}"))
    frame = pd.DataFrame(rows, columns=["item_id", "annotator_id", "label"])
    gold = {f"i{i}": f"c{c}" for i, c in enumerate(truth)}
    return AnnotationData.from_long_df(frame), gold, accuracy


def _accuracy(model, data, gold):
    return float(
        np.mean([gold[i] == label for i, label in zip(data.item_ids, model.labels, strict=True)])
    )


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

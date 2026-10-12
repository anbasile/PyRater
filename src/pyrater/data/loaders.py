"""Bundled and downloadable example datasets."""

from __future__ import annotations

import urllib.request
from collections.abc import Callable, Hashable, Mapping
from importlib import resources
from pathlib import Path

import pandas as pd
import platformdirs

from pyrater.data.annotation_data import AnnotationData

# The file from the original release that holds every collected label. The
# same repository's `rte/rte.anno.csv` does not: it has 2,040 of the 8,000
# labels, and its `value` column is the gold label rather than the worker's.
_RTE_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/edwinrobots/bayesian_combination_tutorial2/main/"
    "all_collected_data/rte.standardized.tsv"
)


def _read_gold_csv(path: Path) -> Mapping[Hashable, Hashable]:
    df = pd.read_csv(path)
    return dict(zip(df["item_id"], df["label"], strict=True))


def _load_synthetic_small() -> AnnotationData:
    package_dir = resources.files("pyrater.data.datasets")
    with (
        resources.as_file(package_dir / "synthetic_small.csv") as annotations_path,
        resources.as_file(package_dir / "synthetic_small_gold.csv") as gold_path,
    ):
        gold = _read_gold_csv(gold_path)
        return AnnotationData.from_csv(annotations_path, gold=gold)


def _rte_cache_dir() -> Path:
    return Path(platformdirs.user_cache_dir("pyrater")) / "datasets" / "rte"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (fixed, hardcoded, HTTPS URLs)


def _load_rte() -> AnnotationData:
    annotations_path = _rte_cache_dir() / "rte.standardized.tsv"
    if not annotations_path.exists():
        _download(_RTE_ANNOTATIONS_URL, annotations_path)
    return AnnotationData.from_snow2008(annotations_path)


_LOADERS: dict[str, Callable[[], AnnotationData]] = {
    "synthetic_small": _load_synthetic_small,
    "rte": _load_rte,
}


def load_dataset(name: str) -> AnnotationData:
    """Load a built-in example dataset by name.

    ``"synthetic_small"`` is bundled with the package (fast, no network,
    used in tests and tutorials).

    ``"rte"`` is the recognising-textual-entailment task of the Snow et al.
    (2008) Mechanical Turk release: 800 items with 10 labels each, 8,000
    labels from 164 workers, and a gold label for every item. Labels are the
    release's own codes, ``1`` for entailment and ``0`` for none. The file is
    downloaded on first use and cached under a per-user cache directory (not
    bundled in the wheel — see :func:`_rte_cache_dir`).
    """
    try:
        loader = _LOADERS[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list(_LOADERS)}") from None
    return loader()

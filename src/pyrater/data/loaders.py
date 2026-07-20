"""Bundled and downloadable example datasets."""

from __future__ import annotations

import urllib.request
from collections.abc import Callable, Hashable, Mapping
from importlib import resources
from pathlib import Path

import pandas as pd
import platformdirs

from pyrater.data.annotation_data import AnnotationData

_RTE_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/edwinrobots/bayesian_combination_tutorial2/main/"
    "snow2008_mturk_data_with_orig_files_assembled_201904/rte/rte.anno.csv"
)
_RTE_GOLD_URL = (
    "https://raw.githubusercontent.com/edwinrobots/bayesian_combination_tutorial2/main/"
    "snow2008_mturk_data_with_orig_files_assembled_201904/rte/rte1.tsv"
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
    cache = _rte_cache_dir()
    annotations_path = cache / "rte.anno.csv"
    gold_path = cache / "rte1.tsv"
    if not annotations_path.exists():
        _download(_RTE_ANNOTATIONS_URL, annotations_path)
    if not gold_path.exists():
        _download(_RTE_GOLD_URL, gold_path)
    return AnnotationData.from_snow2008(annotations_path, gold_path)


_LOADERS: dict[str, Callable[[], AnnotationData]] = {
    "synthetic_small": _load_synthetic_small,
    "rte": _load_rte,
}


def load_dataset(name: str) -> AnnotationData:
    """Load a built-in example dataset by name.

    ``"synthetic_small"`` is bundled with the package (fast, no network,
    used in tests and tutorials). ``"rte"`` is the real Snow et al. (2008)
    Mechanical Turk RTE release; it's downloaded on first use and cached
    under a per-user cache directory (not bundled in the wheel — see
    :func:`_rte_cache_dir`).
    """
    try:
        loader = _LOADERS[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list(_LOADERS)}") from None
    return loader()

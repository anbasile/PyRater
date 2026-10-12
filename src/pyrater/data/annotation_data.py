"""The canonical in-memory representation of an annotation matrix."""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

IntArray = npt.NDArray[np.int64]
ObjectArray = npt.NDArray[np.object_]


class StanData(TypedDict):
    """The shape produced by :meth:`AnnotationData.to_stan_data`."""

    K: int
    N: int
    I: int
    ii: list[int]
    y: list[int]
    J: NotRequired[int]
    jj: NotRequired[list[int]]


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class AnnotationData:
    """A canonical, immutable annotation matrix.

    All identifiers (items, annotators, classes) may be arbitrary hashable
    values on construction; internally they are remapped to dense
    ``0..n-1`` integer ranges, recoverable via ``item_ids``/``annotator_ids``/
    ``class_labels``.
    """

    ii: IntArray
    jj: IntArray | None
    y: IntArray
    item_ids: ObjectArray
    annotator_ids: ObjectArray | None
    class_labels: ObjectArray
    gold: dict[int, int] | None = None

    def __post_init__(self) -> None:
        if self.n_classes < 2:
            raise ValueError(f"Need at least 2 classes, got {self.n_classes}.")

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_annotators(self) -> int:
        return 0 if self.annotator_ids is None else len(self.annotator_ids)

    @property
    def n_classes(self) -> int:
        return len(self.class_labels)

    @property
    def n_annotations(self) -> int:
        return len(self.y)

    def __repr__(self) -> str:
        gold_note = "no gold labels" if not self.gold else f"{len(self.gold)} gold labels"
        return (
            f"AnnotationData: {self.n_items} items, {self.n_annotators} annotators, "
            f"{self.n_classes} classes, {self.n_annotations} annotations, {gold_note}"
        )

    # -- construction -----------------------------------------------------

    @classmethod
    def from_long_df(
        cls,
        df: pd.DataFrame,
        *,
        item_col: str = "item_id",
        annotator_col: str = "annotator_id",
        label_col: str = "label",
        gold: Mapping[Hashable, Hashable] | None = None,
    ) -> AnnotationData:
        ii_codes, item_uniques = pd.factorize(df[item_col].to_numpy(), sort=False)
        jj_codes, annotator_uniques = pd.factorize(df[annotator_col].to_numpy(), sort=False)
        y_codes, class_uniques = pd.factorize(df[label_col].to_numpy(), sort=False)

        item_ids: ObjectArray = np.asarray(item_uniques, dtype=object)
        annotator_ids: ObjectArray = np.asarray(annotator_uniques, dtype=object)
        class_labels: ObjectArray = np.asarray(class_uniques, dtype=object)

        encoded_gold = cls._encode_gold(gold, item_ids, class_labels) if gold is not None else None

        return cls(
            ii=ii_codes.astype(np.int64),
            jj=jj_codes.astype(np.int64),
            y=y_codes.astype(np.int64),
            item_ids=item_ids,
            annotator_ids=annotator_ids,
            class_labels=class_labels,
            gold=encoded_gold,
        )

    @classmethod
    def from_wide_df(
        cls,
        df: pd.DataFrame,
        *,
        item_col: str | None,
        annotator_cols: Sequence[str] | None = None,
        gold_col: str | None = None,
        gold: Mapping[Hashable, Hashable] | None = None,
    ) -> AnnotationData:
        working = df.set_index(item_col) if item_col is not None else df.copy()

        gold_series = working.pop(gold_col) if gold_col is not None else None
        if annotator_cols is not None:
            working = working[list(annotator_cols)]

        empty_items = working.index[working.isna().all(axis=1)]
        if len(empty_items) > 0:
            raise ValueError(f"Item(s) with zero annotations: {list(empty_items)}")
        empty_annotators = working.columns[working.isna().all(axis=0)]
        if len(empty_annotators) > 0:
            raise ValueError(f"Annotator(s) with zero annotations: {list(empty_annotators)}")

        long = (
            working.reset_index(names="__item__")
            .melt(id_vars="__item__", var_name="__annotator__", value_name="__label__")
            .dropna(subset=["__label__"])
        )

        gold_dict: dict[Hashable, Hashable] | None = None
        if gold_series is not None:
            gold_dict = dict(gold_series.dropna().items())
        elif gold is not None:
            gold_dict = dict(gold)

        return cls.from_long_df(
            long,
            item_col="__item__",
            annotator_col="__annotator__",
            label_col="__label__",
            gold=gold_dict,
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        item_col: str = "item_id",
        annotator_col: str = "annotator_id",
        label_col: str = "label",
        gold: Mapping[Hashable, Hashable] | None = None,
        **read_csv_kwargs: Any,
    ) -> AnnotationData:
        df = pd.read_csv(path, **read_csv_kwargs)
        return cls.from_long_df(
            df, item_col=item_col, annotator_col=annotator_col, label_col=label_col, gold=gold
        )

    @classmethod
    def from_snow2008(
        cls,
        annotations_path: str | Path,
        *,
        gold: Mapping[Hashable, Hashable] | None = None,
    ) -> AnnotationData:
        """Load one task of the Snow et al. (2008) Mechanical Turk release.

        ``annotations_path`` is a ``<task>.standardized.tsv`` file from the
        release's ``all_collected_data`` folder (e.g. ``rte.standardized.tsv``):
        tab-separated, one row per collected label, with the columns

        - ``!amt_worker_ids``: the worker;
        - ``orig_id``: the item;
        - ``response``: the label that worker gave that item;
        - ``gold``: the item's gold label, repeated on each of its rows, in
          the same coding as ``response``.

        Other columns (``!amt_annotation_ids``) are ignored. Labels keep the
        file's own coding; for RTE that is ``1`` for entailment and ``0`` for
        none. Gold labels are taken from the ``gold`` column unless ``gold``
        is passed, which replaces them.

        The release also has per-task ``<task>.anno.csv`` files. Those are not
        this layout and are not accepted: for RTE that file covers a quarter
        of the labels, and its ``value`` column is the gold label, not the
        worker's.
        """
        annotations = pd.read_csv(annotations_path, sep="\t")
        missing = [
            column
            for column in ("!amt_worker_ids", "orig_id", "response")
            if column not in annotations.columns
        ]
        if missing:
            raise ValueError(
                f"{annotations_path} is not a Snow et al. (2008) `<task>.standardized.tsv` "
                f"file: it has no column(s) {missing}. Found: {list(annotations.columns)}."
            )

        long_df = pd.DataFrame(
            {
                "item_id": annotations["orig_id"],
                "annotator_id": annotations["!amt_worker_ids"],
                "label": annotations["response"],
            }
        )
        gold_dict: Mapping[Hashable, Hashable] | None = gold
        if gold_dict is None and "gold" in annotations.columns:
            per_item = annotations.groupby("orig_id", sort=False)["gold"]
            ambiguous = per_item.nunique()
            ambiguous = ambiguous[ambiguous > 1]
            if len(ambiguous) > 0:
                raise ValueError(
                    f"Item(s) with more than one gold label: {ambiguous.index.tolist()[:10]}"
                )
            first = per_item.first()
            gold_dict = dict(zip(first.index.tolist(), first.tolist(), strict=True))
        return cls.from_long_df(long_df, gold=gold_dict)

    @staticmethod
    def _encode_gold(
        gold: Mapping[Hashable, Hashable], item_ids: ObjectArray, class_labels: ObjectArray
    ) -> dict[int, int]:
        item_index = {item: idx for idx, item in enumerate(item_ids)}
        class_index = {label: idx for idx, label in enumerate(class_labels)}
        encoded: dict[int, int] = {}
        for item, label in gold.items():
            if item not in item_index:
                raise ValueError(f"Gold item {item!r} was not found among the annotated items.")
            if label not in class_index:
                raise ValueError(f"Gold class {label!r} was not found among the annotated classes.")
            encoded[item_index[item]] = class_index[label]
        return encoded

    # -- conversion ---------------------------------------------------------

    def to_stan_data(self, *, include_annotators: bool = True) -> StanData:
        data: StanData = {
            "K": self.n_classes,
            "N": self.n_annotations,
            "I": self.n_items,
            "ii": (self.ii + 1).tolist(),
            "y": (self.y + 1).tolist(),
        }
        if include_annotators:
            if self.jj is None:
                raise ValueError(
                    "This model requires annotator identifiers, but the data has none "
                    "(`jj` is None)."
                )
            data["J"] = self.n_annotators
            data["jj"] = (self.jj + 1).tolist()
        return data

    def decode_labels(self, internal_class_ids: npt.ArrayLike) -> ObjectArray:
        return self.class_labels[np.asarray(internal_class_ids)]

    def to_long_df(self) -> pd.DataFrame:
        columns: dict[str, ObjectArray] = {
            "item_id": self.item_ids[self.ii],
            "label": self.class_labels[self.y],
        }
        if self.jj is not None and self.annotator_ids is not None:
            columns["annotator_id"] = self.annotator_ids[self.jj]
        return pd.DataFrame(columns)

    def _gold_as_external(self) -> dict[Hashable, Hashable] | None:
        if self.gold is None:
            return None
        return {
            self.item_ids[item_idx]: self.class_labels[class_idx]
            for item_idx, class_idx in self.gold.items()
        }

    # -- filtering ------------------------------------------------------------

    def filter_items(self, item_ids: Sequence[Hashable]) -> AnnotationData:
        keep = set(item_ids)
        df = self.to_long_df()
        filtered = df[df["item_id"].isin(keep)]
        return AnnotationData.from_long_df(filtered, gold=self._gold_as_external())

    def filter_annotators(self, *, min_annotations: int) -> AnnotationData:
        if self.jj is None or self.annotator_ids is None:
            raise ValueError("This data has no annotator identifiers to filter on.")
        counts = np.bincount(self.jj, minlength=self.n_annotators)
        keep = {
            self.annotator_ids[j] for j in range(self.n_annotators) if counts[j] >= min_annotations
        }
        df = self.to_long_df()
        filtered = df[df["annotator_id"].isin(keep)]
        return AnnotationData.from_long_df(filtered, gold=self._gold_as_external())

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pyrater.data.annotation_data import AnnotationData


@pytest.fixture
def long_df(long_annotations_df: pd.DataFrame) -> pd.DataFrame:
    return long_annotations_df


@pytest.fixture
def data(long_df: pd.DataFrame) -> AnnotationData:
    return AnnotationData.from_long_df(long_df)


class TestFromLongDf:
    def test_shapes(self, data: AnnotationData) -> None:
        assert data.n_items == 3
        assert data.n_annotators == 2
        assert data.n_classes == 2
        assert data.n_annotations == 6
        assert data.ii.shape == (6,)
        assert data.jj is not None
        assert data.jj.shape == (6,)
        assert data.y.shape == (6,)

    def test_class_ordering_is_first_appearance_not_alphabetical(
        self, data: AnnotationData
    ) -> None:
        # "yes" appears before "no" in the input.
        assert list(data.class_labels) == ["yes", "no"]

    def test_item_ordering_is_first_appearance(self, data: AnnotationData) -> None:
        assert list(data.item_ids) == ["b", "a", "c"]

    def test_annotator_ordering_is_first_appearance(self, data: AnnotationData) -> None:
        assert data.annotator_ids is not None
        assert list(data.annotator_ids) == ["w2", "w1"]

    def test_indices_are_internally_consistent(
        self, long_df: pd.DataFrame, data: AnnotationData
    ) -> None:
        assert data.jj is not None
        assert data.annotator_ids is not None
        for n in range(len(long_df)):
            assert data.item_ids[data.ii[n]] == long_df["item_id"].iloc[n]
            assert data.annotator_ids[data.jj[n]] == long_df["annotator_id"].iloc[n]
            assert data.class_labels[data.y[n]] == long_df["label"].iloc[n]

    def test_custom_column_names(self, long_df: pd.DataFrame) -> None:
        renamed = long_df.rename(
            columns={"item_id": "item", "annotator_id": "worker", "label": "response"}
        )
        data = AnnotationData.from_long_df(
            renamed, item_col="item", annotator_col="worker", label_col="response"
        )
        assert data.n_items == 3

    def test_repr_is_a_readable_summary(self, data: AnnotationData) -> None:
        text = str(data)
        assert "3 items" in text
        assert "2 annotators" in text
        assert "2 classes" in text
        assert "6 annotations" in text

    def test_is_immutable(self, data: AnnotationData) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            data.ii = np.zeros(1, dtype=np.int64)  # ty: ignore[invalid-assignment]


class TestGold:
    def test_gold_encoded_to_internal_ids(self, long_df: pd.DataFrame) -> None:
        data = AnnotationData.from_long_df(long_df, gold={"a": "yes", "c": "no"})
        assert data.gold is not None
        a_idx = list(data.item_ids).index("a")
        c_idx = list(data.item_ids).index("c")
        yes_idx = list(data.class_labels).index("yes")
        no_idx = list(data.class_labels).index("no")
        assert data.gold[a_idx] == yes_idx
        assert data.gold[c_idx] == no_idx

    def test_gold_with_unknown_class_raises(self, long_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="class"):
            AnnotationData.from_long_df(long_df, gold={"a": "maybe"})

    def test_gold_with_unknown_item_raises(self, long_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="item"):
            AnnotationData.from_long_df(long_df, gold={"nonexistent-item": "yes"})

    def test_no_gold_by_default(self, data: AnnotationData) -> None:
        assert data.gold is None


class TestValidation:
    def test_single_class_raises(self) -> None:
        df = pd.DataFrame(
            {"item_id": ["a", "b"], "annotator_id": ["w1", "w1"], "label": ["yes", "yes"]}
        )
        with pytest.raises(ValueError, match="class"):
            AnnotationData.from_long_df(df)

    def test_annotator_with_zero_annotations_raises(self) -> None:
        wide = pd.DataFrame({"w1": ["yes", "no"], "w2": [np.nan, np.nan]}, index=["a", "b"])
        with pytest.raises(ValueError, match="w2"):
            AnnotationData.from_wide_df(wide, item_col=None)

    def test_item_with_zero_annotations_raises(self) -> None:
        wide = pd.DataFrame({"w1": ["yes", np.nan], "w2": ["no", np.nan]}, index=["a", "b"])
        with pytest.raises(ValueError, match="b"):
            AnnotationData.from_wide_df(wide, item_col=None)


class TestFromWideDf:
    def test_matches_long_equivalent(self) -> None:
        wide = pd.DataFrame(
            {"w1": ["yes", "no", "yes"], "w2": ["yes", np.nan, "no"]}, index=["a", "b", "c"]
        )
        data = AnnotationData.from_wide_df(wide, item_col=None)
        assert data.n_items == 3
        assert data.n_annotators == 2
        assert data.n_annotations == 5  # one NaN cell skipped

    def test_item_col_as_a_regular_column(self) -> None:
        wide = pd.DataFrame({"item": ["a", "b"], "w1": ["yes", "no"], "w2": ["yes", "no"]})
        data = AnnotationData.from_wide_df(wide, item_col="item")
        assert data.n_items == 2
        assert list(data.item_ids) == ["a", "b"]


class TestToStanData:
    def test_one_indexed_and_shapes(self, data: AnnotationData) -> None:
        stan_data = data.to_stan_data()
        assert stan_data["I"] == 3
        assert stan_data["J"] == 2
        assert stan_data["K"] == 2
        assert stan_data["N"] == 6
        assert min(stan_data["ii"]) == 1
        assert max(stan_data["ii"]) == 3
        assert min(stan_data["jj"]) == 1
        assert max(stan_data["jj"]) == 2
        assert min(stan_data["y"]) == 1
        assert max(stan_data["y"]) == 2

    def test_exclude_annotators(self, data: AnnotationData) -> None:
        stan_data = data.to_stan_data(include_annotators=False)
        assert "J" not in stan_data
        assert "jj" not in stan_data
        assert stan_data["I"] == 3

    def test_include_annotators_without_jj_raises(self) -> None:
        wide = pd.DataFrame({"w1": ["yes", "no"]}, index=["a", "b"])
        data = AnnotationData.from_wide_df(wide, item_col=None)
        object.__setattr__(data, "jj", None)
        object.__setattr__(data, "annotator_ids", None)
        with pytest.raises(ValueError, match="annotator"):
            data.to_stan_data(include_annotators=True)


class TestDecodeLabels:
    def test_round_trip(self, data: AnnotationData) -> None:
        internal = np.array([0, 1, 0])
        decoded = data.decode_labels(internal)
        assert list(decoded) == [data.class_labels[0], data.class_labels[1], data.class_labels[0]]


class TestFilters:
    def test_filter_items_keeps_only_requested(self, data: AnnotationData) -> None:
        filtered = data.filter_items(["a", "b"])
        assert filtered.n_items == 2
        assert set(filtered.item_ids) == {"a", "b"}

    def test_filter_annotators_by_min_annotations(self) -> None:
        df = pd.DataFrame(
            {
                "item_id": ["a", "b", "c", "d"],
                "annotator_id": ["w1", "w1", "w1", "w2"],
                "label": ["yes", "no", "yes", "no"],
            }
        )
        data = AnnotationData.from_long_df(df)
        filtered = data.filter_annotators(min_annotations=2)
        assert filtered.n_annotators == 1
        assert filtered.annotator_ids is not None
        assert list(filtered.annotator_ids) == ["w1"]


class TestFromCsv:
    def test_round_trips_through_disk(self, long_df: pd.DataFrame, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "annotations.csv"
        long_df.to_csv(path, index=False)
        data = AnnotationData.from_csv(path)
        assert data.n_items == 3
        assert data.n_annotators == 2
        assert data.n_annotations == 6


class TestFromSnow2008:
    def test_joins_annotations_and_gold_on_id(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        anno_path = tmp_path / "rte.anno.csv"
        gold_path = tmp_path / "rte1.tsv"
        pd.DataFrame(
            {
                "!amt_assignment_ids": ["asg1", "asg2", "asg3"],
                "!amt_worker_ids": ["worker1", "worker1", "worker2"],
                "id": [101, 102, 101],
                "value": ["TRUE", "FALSE", "FALSE"],
                "task": ["MT", "MT", "MT"],
            }
        ).to_csv(anno_path, index=False)
        pd.DataFrame({"id": [101, 102], "value": ["TRUE", "FALSE"]}).to_csv(
            gold_path, sep="\t", index=False
        )

        data = AnnotationData.from_snow2008(anno_path, gold_path)

        assert data.n_items == 2
        assert data.n_annotators == 2
        assert data.n_annotations == 3
        assert data.gold is not None
        assert len(data.gold) == 2

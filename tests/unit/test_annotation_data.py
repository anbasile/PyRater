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
    """Fixtures follow the release's `<task>.standardized.tsv` layout."""

    @staticmethod
    def _write(path, **overrides) -> None:  # type: ignore[no-untyped-def]
        columns = {
            "!amt_annotation_ids": ["an1", "an2", "an3", "an4"],
            "!amt_worker_ids": ["worker1", "worker2", "worker1", "worker3"],
            "orig_id": [101, 101, 102, 102],
            "response": [1, 0, 0, 0],
            "gold": [1, 1, 0, 0],
        }
        columns.update(overrides)
        frame = pd.DataFrame({k: v for k, v in columns.items() if v is not None})
        frame.to_csv(path, sep="\t", index=False)

    def test_reads_responses_as_labels_and_gold_from_its_column(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "rte.standardized.tsv"
        self._write(path)

        data = AnnotationData.from_snow2008(path)

        assert data.n_items == 2
        assert data.n_annotators == 3
        assert data.n_annotations == 4
        frame = data.to_long_df()
        assert frame["label"].tolist() == [1, 0, 0, 0]
        assert frame["annotator_id"].tolist() == ["worker1", "worker2", "worker1", "worker3"]
        assert data.gold is not None
        decoded = {data.item_ids[i]: data.class_labels[c] for i, c in data.gold.items()}
        assert decoded == {101: 1, 102: 0}

    def test_explicit_gold_replaces_the_gold_column(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "rte.standardized.tsv"
        self._write(path)

        data = AnnotationData.from_snow2008(path, gold={101: 0})

        assert data.gold is not None
        decoded = {data.item_ids[i]: data.class_labels[c] for i, c in data.gold.items()}
        assert decoded == {101: 0}

    def test_a_file_without_a_gold_column_has_no_gold(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "task.standardized.tsv"
        self._write(path, gold=None)

        assert AnnotationData.from_snow2008(path).gold is None

    def test_rejects_the_anno_csv_layout(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        # The layout of the release's `rte.anno.csv`, where `value` is the gold label.
        path = tmp_path / "rte.anno.csv"
        pd.DataFrame(
            {
                "!amt_assignment_ids": ["asg1", "asg1"],
                "!amt_worker_ids": ["worker1", "worker1"],
                "!hypothesis": ["No", "Yes"],
                "id": [101, 102],
                "value": [False, True],
            }
        ).to_csv(path, index=False)

        with pytest.raises(ValueError, match="standardized.tsv"):
            AnnotationData.from_snow2008(path)

    def test_rejects_an_item_with_two_gold_labels(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "rte.standardized.tsv"
        self._write(path, gold=[1, 0, 0, 0])

        with pytest.raises(ValueError, match="more than one gold label.*101"):
            AnnotationData.from_snow2008(path)

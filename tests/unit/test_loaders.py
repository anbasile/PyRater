from pathlib import Path

import pandas as pd
import pytest

from pyrater.data.annotation_data import AnnotationData
from pyrater.data.loaders import load_dataset


class TestSyntheticSmall:
    def test_loads_a_non_trivial_dataset(self) -> None:
        data = load_dataset("synthetic_small")
        assert data.n_items == 12
        assert data.n_annotators == 5
        assert data.n_classes == 3
        assert data.n_annotations == 60

    def test_has_full_gold(self) -> None:
        data = load_dataset("synthetic_small")
        assert data.gold is not None
        assert len(data.gold) == data.n_items

    def test_is_usable_as_a_fixture_across_calls(self) -> None:
        # Loading twice should give equivalent (not necessarily identical-object) data.
        first = load_dataset("synthetic_small")
        second = load_dataset("synthetic_small")
        assert first.n_items == second.n_items
        assert list(first.item_ids) == list(second.item_ids)


class TestUnknownDataset:
    def test_raises_with_available_list(self) -> None:
        with pytest.raises(KeyError, match="synthetic_small"):
            load_dataset("does-not-exist")


def _standardized_rows() -> pd.DataFrame:
    """Six rows in the layout of the release's `rte.standardized.tsv`: two
    items with three labels each, one gold label per item repeated on its rows."""
    return pd.DataFrame(
        {
            "!amt_annotation_ids": ["an1", "an2", "an3", "an4", "an5", "an6"],
            "!amt_worker_ids": ["A19IB", "AEX5N", "A17RP", "A19IB", "AEX5N", "A15L6"],
            "orig_id": [266, 266, 266, 804, 804, 804],
            "response": [1, 1, 0, 0, 1, 0],
            "gold": [1, 1, 1, 0, 0, 0],
        }
    )


class TestRte:
    @pytest.fixture(autouse=True)
    def _fake_download(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        calls: list[str] = []

        def fake_download(url: str, dest: Path) -> None:
            calls.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            _standardized_rows().to_csv(dest, sep="\t", index=False)

        monkeypatch.setattr("pyrater.data.loaders._download", fake_download)
        monkeypatch.setattr("pyrater.data.loaders._rte_cache_dir", lambda: tmp_path / "rte-cache")
        return calls

    def test_downloads_and_parses(self, _fake_download: list[str]) -> None:
        data = load_dataset("rte")
        assert isinstance(data, AnnotationData)
        assert data.n_items == 2
        assert data.n_annotators == 4
        assert data.n_annotations == 6
        assert len(_fake_download) == 1
        assert _fake_download[0].endswith("all_collected_data/rte.standardized.tsv")

    def test_labels_are_the_workers_responses_not_the_gold_column(self) -> None:
        data = load_dataset("rte")
        frame = data.to_long_df()
        first_item = frame[frame["item_id"] == 266]
        # The workers disagreed on item 266; its gold label is 1 on every row.
        assert first_item["label"].tolist() == [1, 1, 0]
        assert data.gold is not None
        gold = {data.item_ids[i]: data.class_labels[c] for i, c in data.gold.items()}
        assert gold == {266: 1, 804: 0}

    def test_second_call_uses_the_cache_not_the_network(self, _fake_download: list[str]) -> None:
        load_dataset("rte")
        load_dataset("rte")
        assert len(_fake_download) == 1  # not 2: the second call hit the cache

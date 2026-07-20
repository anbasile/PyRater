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


class TestRte:
    @pytest.fixture(autouse=True)
    def _fake_download(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        calls: list[str] = []

        def fake_download(url: str, dest: Path) -> None:
            calls.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.name == "rte1.tsv":
                pd.DataFrame({"id": [101, 102], "value": ["TRUE", "FALSE"]}).to_csv(
                    dest, sep="\t", index=False
                )
            else:
                pd.DataFrame(
                    {
                        "!amt_worker_ids": ["worker1", "worker1", "worker2"],
                        "id": [101, 102, 101],
                        "value": ["TRUE", "FALSE", "FALSE"],
                    }
                ).to_csv(dest, index=False)

        monkeypatch.setattr("pyrater.data.loaders._download", fake_download)
        monkeypatch.setattr("pyrater.data.loaders._rte_cache_dir", lambda: tmp_path / "rte-cache")
        return calls

    def test_downloads_and_parses(self, _fake_download: list[str]) -> None:
        data = load_dataset("rte")
        assert isinstance(data, AnnotationData)
        assert data.n_items == 2
        assert data.n_annotators == 2
        assert len(_fake_download) == 2  # annotations + gold

    def test_second_call_uses_the_cache_not_the_network(self, _fake_download: list[str]) -> None:
        load_dataset("rte")
        load_dataset("rte")
        assert len(_fake_download) == 2  # not 4: the second call hit the cache

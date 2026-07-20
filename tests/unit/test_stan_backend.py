from pathlib import Path

import pytest

from pyrater.stan import _cache_dir, _cached_path, _cached_stan_path


class TestCachedPath:
    def test_writes_content_on_first_call(self, tmp_path: Path) -> None:
        dest = _cached_path(b"model contents", "foo.stan", tmp_path)
        assert dest.read_bytes() == b"model contents"
        assert dest.suffix == ".stan"

    def test_is_idempotent_and_does_not_rewrite(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = _cached_path(b"model contents", "foo.stan", tmp_path)

        write_calls = []
        original_write_bytes = Path.write_bytes

        def tracking_write_bytes(self: Path, data: bytes) -> int:
            write_calls.append(self)
            return original_write_bytes(self, data)

        monkeypatch.setattr(Path, "write_bytes", tracking_write_bytes)
        second = _cached_path(b"model contents", "foo.stan", tmp_path)

        assert second == first
        assert write_calls == []

    def test_different_content_gives_a_different_path(self, tmp_path: Path) -> None:
        first = _cached_path(b"version one", "foo.stan", tmp_path)
        second = _cached_path(b"version two", "foo.stan", tmp_path)
        assert first != second
        assert first.exists()
        assert second.exists()

    def test_creates_cache_dir_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "does" / "not" / "exist" / "yet"
        dest = _cached_path(b"x", "foo.stan", nested)
        assert dest.exists()


class TestCachedStanPath:
    def test_bundled_model_is_cached_with_matching_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("pyrater.stan._cache_dir", lambda: tmp_path)
        dest = _cached_stan_path("DS.stan")
        assert dest.exists()
        assert b"data {" in dest.read_bytes()

    def test_default_cache_dir_is_under_platformdirs_cache(self) -> None:
        cache_dir = _cache_dir()
        assert "pyrater" in str(cache_dir)

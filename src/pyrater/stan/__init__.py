"""Compiling and caching the bundled Stan models.

Bundled ``.stan`` source files live inside the installed package, which may
be a read-only location (e.g. system site-packages) -- CmdStan needs to
write a compiled binary next to the ``.stan`` file, so we copy the source
into a writable, per-user cache directory before compiling there. The cached
copy is keyed by a hash of the source, so a new pyrater release that changes
a bundled model automatically invalidates any stale compiled binary instead
of silently reusing one compiled from different source.
"""

from __future__ import annotations

import hashlib
from functools import cache
from importlib import resources
from pathlib import Path

import platformdirs
from cmdstanpy import CmdStanModel


def _cache_dir() -> Path:
    return Path(platformdirs.user_cache_dir("pyrater")) / "stan"


def _cached_path(content: bytes, filename: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(content).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem, suffix = Path(filename).stem, Path(filename).suffix
    dest = cache_dir / f"{stem}-{digest}{suffix}"
    if not dest.exists():
        dest.write_bytes(content)
    return dest


def _cached_stan_path(stan_file: str) -> Path:
    content = resources.files("pyrater.stan").joinpath(stan_file).read_bytes()
    return _cached_path(content, stan_file, _cache_dir())


@cache
def get_compiled_model(stan_file: str) -> CmdStanModel:
    """Compile (or fetch an already-compiled) bundled Stan model.

    Compilation happens in a writable per-user cache dir, keyed by content
    hash (see module docstring), and is memoized in-process so repeated
    `.fit()` calls within the same run don't repeatedly hit the filesystem.
    """
    stan_path = _cached_stan_path(stan_file)
    return CmdStanModel(stan_file=str(stan_path))

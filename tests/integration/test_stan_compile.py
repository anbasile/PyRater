import pytest
from cmdstanpy import CmdStanModel

from pyrater.stan import get_compiled_model

STAN_FILES = [
    "DS.stan",
    "MACE.stan",
    "Multinomial.stan",
    "HDS-NC.stan",
    "ID-NC.stan",
    "LRE-NC.stan",
]


@pytest.mark.stan
@pytest.mark.parametrize("stan_file", STAN_FILES)
def test_bundled_model_compiles(stan_file: str) -> None:
    model = get_compiled_model(stan_file)
    assert isinstance(model, CmdStanModel)
    assert model.exe_file is not None

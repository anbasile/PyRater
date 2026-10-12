"""The `PyRaterModel` registry. Importing this package registers every
built-in model (Bayesian and baseline) as a side effect. Import order below
matches the paper's own Listing 3 (`'d&s', 'hier_d&s', 'mace', ...`) --
deliberately NOT alphabetized, since registration order is registry order.
"""

# ruff: noqa: I001

from pyrater.models import dawid_skene as _dawid_skene  # noqa: F401
from pyrater.models import hier_dawid_skene as _hier_dawid_skene  # noqa: F401
from pyrater.models import mace as _mace  # noqa: F401
from pyrater.models import multinomial as _multinomial  # noqa: F401
from pyrater.models import item_difficulty as _item_difficulty  # noqa: F401
from pyrater.models import logistic_random_effects as _logistic_random_effects  # noqa: F401

# Not from the paper's listing: Han, Adiguzel & Carpenter (arXiv:2405.19521).
# Registered after the six it extends, so registry order still reads as
# "the 2018 comparison, then what came after it".
from pyrater.models import irt_2pl_ss as _irt_2pl_ss  # noqa: F401
from pyrater.models import baselines as _baselines  # noqa: F401

# The NumPyro backend. Registered last; JAX is imported only when one is fitted.
from pyrater.models import numpyro_models as _numpyro_models  # noqa: F401
from pyrater.models.base import PyRaterModel

__all__ = ["PyRaterModel"]

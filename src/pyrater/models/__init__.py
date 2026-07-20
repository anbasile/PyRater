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
from pyrater.models import baselines as _baselines  # noqa: F401
from pyrater.models.base import PyRaterModel

__all__ = ["PyRaterModel"]

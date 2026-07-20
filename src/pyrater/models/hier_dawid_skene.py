"""Hierarchical Dawid & Skene: annotator confusion matrices are drawn from a
shared population distribution, regularizing low-annotation-count annotators
toward the population mean rather than overfitting to their few labels."""

from __future__ import annotations

from typing import ClassVar

from pyrater.models._stan_backed import StanBackedModel
from pyrater.models.base import PyRaterModel


@PyRaterModel.register("hier_d&s")
class HierDawidSkeneModel(StanBackedModel):
    stan_file: ClassVar[str] = "HDS-NC.stan"
    class_indexed_params: ClassVar[dict[str, int]] = {
        "pi": 0,
        "zeta": 0,
        "Omega": 0,
        "beta_raw": 1,
        "beta_norm": 1,
        "q_z": 1,
    }

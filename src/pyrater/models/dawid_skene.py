"""Dawid & Skene (1979): every annotator has their own confusion matrix."""

from __future__ import annotations

from typing import ClassVar

from pyrater.models._stan_backed import StanBackedModel
from pyrater.models.base import PyRaterModel


@PyRaterModel.register("d&s")
class DawidSkeneModel(StanBackedModel):
    stan_file: ClassVar[str] = "DS.stan"
    class_indexed_params: ClassVar[dict[str, int]] = {"pi": 0, "beta": 1, "q_z": 1}

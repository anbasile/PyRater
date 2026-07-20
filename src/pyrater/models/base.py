"""The `PyRaterModel` contract: registry, fitting orchestration, and the
uniform post-inference computation of labels/difficulty/credibilities shared
by every model."""

from __future__ import annotations

import abc
import warnings
from collections.abc import Callable, Hashable, Mapping
from typing import Any, ClassVar, Self, TypeVar

import arviz as az
import numpy as np
import numpy.typing as npt

from pyrater.data.annotation_data import AnnotationData
from pyrater.diagnostics.convergence import ConvergenceReport, check_convergence
from pyrater.diagnostics.label_switching import align_chains, align_to_gold
from pyrater.exceptions import ConvergenceWarning, ModelCapabilityError, NotFittedError

M = TypeVar("M", bound="PyRaterModel")


class PyRaterModel(abc.ABC):
    """Base class and registry for all annotation models.

    Subclasses declare `name` (set automatically by :meth:`register`) and
    `class_indexed_params` (which posterior parameters are indexed by the
    latent "true class" axis, and along which axis — used for label-switching
    correction; empty for models with no such ambiguity, e.g. MACE).
    """

    name: ClassVar[str]
    class_indexed_params: ClassVar[Mapping[str, int]] = {}

    _registry: ClassVar[dict[str, type[PyRaterModel]]] = {}

    def __init__(self) -> None:
        self._data: AnnotationData | None = None
        self._idata: az.InferenceData | None = None
        self.convergence_: ConvergenceReport | None = None

    # -- registry -----------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> Callable[[type[M]], type[M]]:
        def decorator(model_cls: type[M]) -> type[M]:
            if name in PyRaterModel._registry:
                existing = PyRaterModel._registry[name].__name__
                raise ValueError(f"A model is already registered under {name!r}: {existing}")
            model_cls.name = name
            PyRaterModel._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> PyRaterModel:
        try:
            model_cls = PyRaterModel._registry[name]
        except KeyError:
            raise KeyError(
                f"Unknown model {name!r}. Available models: {cls.list_available()}"
            ) from None
        return model_cls()

    @classmethod
    def list_available(cls) -> list[str]:
        return list(PyRaterModel._registry)

    # -- fitting --------------------------------------------------------------

    def fit(
        self,
        data: AnnotationData,
        *,
        gold: Mapping[Hashable, Hashable] | None = None,
        chains: int = 4,
        warmup: int = 1000,
        iters: int = 1000,
        seed: int | None = None,
        align: bool = True,
        check_rhat: bool = True,
        rhat_threshold: float = 1.01,
        sample_size_per_class: int = 5,
        rng: np.random.Generator | None = None,
        **cmdstanpy_kwargs: Any,
    ) -> Self:
        self._data = data
        idata = self._sample(
            data, chains=chains, warmup=warmup, iters=iters, seed=seed, **cmdstanpy_kwargs
        )

        if align and self.class_indexed_params:
            idata, _ = align_chains(idata, self.class_indexed_params)

            if gold is not None:
                encoded_gold = AnnotationData._encode_gold(gold, data.item_ids, data.class_labels)
            else:
                encoded_gold = data.gold
            if encoded_gold:
                idata, _ = align_to_gold(
                    idata,
                    self.class_indexed_params,
                    encoded_gold,
                    data.n_classes,
                    sample_size_per_class=sample_size_per_class,
                    rng=rng,
                )

        self._idata = idata

        if check_rhat:
            self.convergence_ = check_convergence(idata, threshold=rhat_threshold)
            if not self.convergence_.ok:
                warnings.warn(
                    f"{self.name}: R-hat check found parameters above threshold "
                    f"{rhat_threshold}; results may not have converged. "
                    f"See `.convergence_.summary()` for details.",
                    ConvergenceWarning,
                    stacklevel=2,
                )
        else:
            self.convergence_ = None

        return self

    @abc.abstractmethod
    def _sample(
        self,
        data: AnnotationData,
        *,
        chains: int,
        warmup: int,
        iters: int,
        seed: int | None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Return the posterior as an `arviz.InferenceData`. Real Stan-backed
        models compile+sample; baselines build a synthetic point-mass
        posterior instead."""
        raise NotImplementedError

    # -- shared post-inference computations ------------------------------------

    def _q_z_mean(self) -> npt.NDArray[np.float64]:
        posterior = self.idata["posterior"]
        result: npt.NDArray[np.float64] = posterior["q_z"].mean(dim=("chain", "draw")).to_numpy()
        return result

    def _compute_difficulty(self) -> npt.NDArray[np.float64]:
        """Default: normalized posterior entropy of each item's class
        distribution. Models with a model-native difficulty parameter
        (item_difficulty, lre) override this."""
        proba = self._q_z_mean()
        n_classes = proba.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.nansum(proba * np.log(proba), axis=1)
        result: npt.NDArray[np.float64] = entropy / np.log(n_classes)
        return result

    # -- public properties ----------------------------------------------------

    @property
    def idata(self) -> az.InferenceData:
        if self._idata is None:
            raise NotFittedError(self.name)
        return self._idata

    @property
    def labels_proba(self) -> npt.NDArray[np.float64]:
        return self._q_z_mean()

    @property
    def labels(self) -> npt.NDArray[np.object_]:
        internal = self._q_z_mean().argmax(axis=1)  # raises NotFittedError if not fitted
        assert self._data is not None
        return self._data.decode_labels(internal)

    @property
    def difficulty(self) -> npt.NDArray[np.float64]:
        _ = self.idata  # raises NotFittedError if not fitted
        return self._compute_difficulty()

    @property
    def credibilities(self) -> npt.NDArray[np.float64]:
        _ = self.idata  # raises NotFittedError if not fitted
        assert self._data is not None
        if self._data.jj is None:
            raise ModelCapabilityError(
                "Credibilities are unavailable: the input data has no annotator identifiers."
            )
        internal_labels = self._q_z_mean().argmax(axis=1)
        matches = (internal_labels[self._data.ii] == self._data.y).astype(np.float64)

        n_annotators = self._data.n_annotators
        agree = np.zeros(n_annotators)
        counts = np.zeros(n_annotators)
        np.add.at(agree, self._data.jj, matches)
        np.add.at(counts, self._data.jj, 1)
        result: npt.NDArray[np.float64] = agree / counts
        return result

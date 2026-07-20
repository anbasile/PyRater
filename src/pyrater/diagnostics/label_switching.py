"""Label-switching correction for mixture-of-classes annotation models.

Stan has no idea which of its `K` latent class slots corresponds to which
real-world class name; different chains (and, without gold, the whole fit)
can settle on different, equally valid internal numberings. This module
fixes that: first by realigning every chain against a reference chain
(:func:`align_chains`), then, if gold labels are available, by anchoring the
now-mutually-aligned chains to the real class names (:func:`align_to_gold`).

The permutation search uses the Hungarian algorithm
(:func:`scipy.optimize.linear_sum_assignment`) rather than a brute-force
search over all `K!` permutations: same optimal answer, polynomial instead
of factorial.

ArviZ's ``InferenceData`` sets its groups (``posterior``, ``sample_stats``,
...) dynamically in ``__init__`` rather than declaring them as static
attributes, so a type checker can't see ``idata.posterior`` as valid; this
module reads groups via ``idata["posterior"]`` (which *is* properly typed,
returning ``xr.Dataset``) and writes the corrected group back via
``setattr(idata, "posterior", ...)`` rather than `idata.posterior = ...`,
for the same reason.
"""

from __future__ import annotations

from collections.abc import Mapping

import arviz as az
import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.optimize import linear_sum_assignment

IntArray = npt.NDArray[np.int64]


def infer_chain_labels(
    idata: az.InferenceData, *, var_name: str = "q_z", subset: IntArray | None = None
) -> IntArray:
    """Per-chain argmax of the posterior-mean class distribution.

    Returns an ``(n_chains, n_items)`` array (or ``(n_chains, len(subset))``
    if ``subset`` is given).
    """
    mean = idata["posterior"][var_name].mean(dim="draw")  # (chain, item, class)
    if subset is not None:
        item_dim = mean.dims[1]
        mean = mean.isel({item_dim: subset})
    labels: IntArray = mean.to_numpy().argmax(axis=-1)
    return labels


def match_permutation(inferred: IntArray, reference: IntArray, n_classes: int) -> IntArray:
    """The permutation of `inferred`'s labels that best matches `reference`.

    Returns ``perm`` such that ``perm[k]`` is the class in ``reference``'s
    numbering that class ``k`` in ``inferred``'s numbering should become.
    Solved optimally via the Hungarian algorithm on the ``K x K`` agreement
    (contingency) matrix.
    """
    contingency = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(contingency, (inferred, reference), 1)
    row_ind, col_ind = linear_sum_assignment(contingency, maximize=True)
    perm: IntArray = np.empty(n_classes, dtype=np.int64)
    perm[row_ind] = col_ind
    return perm


def _apply_permutation(
    idata: az.InferenceData, class_indexed_params: Mapping[str, int], perm: npt.NDArray[np.int64]
) -> az.InferenceData:
    """Apply `perm` (shape (K,) for a single global perm, or (n_chains, K) for
    a per-chain one) to every declared class-indexed parameter of the
    ``posterior`` group, along its declared axis, and return a new
    InferenceData (the input is left untouched)."""
    if not class_indexed_params:
        return idata

    posterior = idata["posterior"]
    new_vars: dict[str, xr.DataArray] = {
        str(name): posterior[str(name)] for name in posterior.data_vars
    }
    for name, axis in class_indexed_params.items():
        if name not in new_vars:
            continue
        da = new_vars[name]
        class_dim = da.dims[2 + axis]  # dims: (chain, draw, *param_dims)
        inv_perm = np.argsort(perm, axis=-1)
        if perm.ndim == 1:
            indexer = xr.DataArray(inv_perm, dims=[class_dim])
        else:
            indexer = xr.DataArray(inv_perm, dims=["chain", class_dim])
        # `.isel` also permutes the (otherwise-meaningless) integer coordinate
        # on `class_dim`; drop it so the reassignment below is purely
        # positional instead of xarray silently re-aligning the reordered
        # data back to its original coordinate labels.
        new_vars[name] = da.isel({class_dim: indexer}).drop_vars(class_dim, errors="ignore")

    result = idata.copy()
    setattr(result, "posterior", posterior.assign(new_vars))  # noqa: B010 (see module docstring)
    return result


def align_chains(
    idata: az.InferenceData,
    class_indexed_params: Mapping[str, int],
    *,
    var_name: str = "q_z",
    reference_chain: int = 0,
) -> tuple[az.InferenceData, IntArray]:
    """Realign every chain's class numbering against `reference_chain`.

    Returns the corrected InferenceData and a ``(n_chains, n_classes)``
    array of the permutation applied to each chain (identity for the
    reference chain).
    """
    if not class_indexed_params:
        n_chains = idata["posterior"].sizes["chain"]
        n_classes = idata["posterior"][var_name].sizes[idata["posterior"][var_name].dims[-1]]
        identity = np.tile(np.arange(n_classes, dtype=np.int64), (n_chains, 1))
        return idata, identity

    chain_labels = infer_chain_labels(idata, var_name=var_name)  # (n_chains, n_items)
    n_chains, _ = chain_labels.shape
    n_classes = idata["posterior"][var_name].sizes[idata["posterior"][var_name].dims[-1]]

    perm_by_chain = np.empty((n_chains, n_classes), dtype=np.int64)
    for chain in range(n_chains):
        if chain == reference_chain:
            perm_by_chain[chain] = np.arange(n_classes)
        else:
            perm_by_chain[chain] = match_permutation(
                chain_labels[chain], chain_labels[reference_chain], n_classes
            )

    aligned = _apply_permutation(idata, class_indexed_params, perm_by_chain)
    return aligned, perm_by_chain


def align_to_gold(
    idata: az.InferenceData,
    class_indexed_params: Mapping[str, int],
    gold: Mapping[int, int],
    n_classes: int,
    *,
    var_name: str = "q_z",
    sample_size_per_class: int = 5,
    rng: np.random.Generator | None = None,
) -> tuple[az.InferenceData, IntArray]:
    """Anchor a (chain-mutually-aligned) fit's numbering to real class names.

    Takes a class-stratified sample of up to `sample_size_per_class` gold
    items per class (or all of them if fewer are available), finds the
    single global permutation that best matches the model's predictions on
    that sample against the true gold labels, and applies it uniformly to
    every chain.
    """
    if not gold:
        raise ValueError("`gold` must be a non-empty mapping of item index to class index.")
    if rng is None:
        rng = np.random.default_rng()

    by_class: dict[int, list[int]] = {}
    for item_idx, class_idx in gold.items():
        by_class.setdefault(class_idx, []).append(item_idx)

    sample_items: list[int] = []
    sample_labels: list[int] = []
    for class_idx, items in by_class.items():
        chosen = (
            items
            if len(items) <= sample_size_per_class
            else list(rng.choice(items, size=sample_size_per_class, replace=False))
        )
        sample_items.extend(chosen)
        sample_labels.extend([class_idx] * len(chosen))

    mean = idata["posterior"][var_name].mean(dim=("chain", "draw"))  # (item, class)
    item_dim = mean.dims[0]
    predicted: IntArray = (
        mean.isel({item_dim: np.asarray(sample_items, dtype=np.int64)}).to_numpy().argmax(axis=-1)
    )

    perm = match_permutation(predicted, np.asarray(sample_labels, dtype=np.int64), n_classes)
    aligned = _apply_permutation(idata, class_indexed_params, perm)
    return aligned, perm

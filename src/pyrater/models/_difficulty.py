"""Shared model-native difficulty formula for `item_difficulty` and `lre`.

Both models expose `theta_norm[I, K, K]`: a log-softmax-normalized per-item
confusion structure (dim 1 = the hypothesis/true class, dim 2 = the assigned
class). An item's expected "own-class" accuracy is
`sum_h pi[h] * exp(theta_norm[i, h, h])`; difficulty is one minus that.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def theta_norm_difficulty(
    theta_norm_mean: npt.NDArray[np.float64], pi_mean: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    diagonal = np.diagonal(theta_norm_mean, axis1=1, axis2=2)  # (I, K), log-probabilities
    expected_accuracy = (pi_mean[np.newaxis, :] * np.exp(diagonal)).sum(axis=1)
    result: npt.NDArray[np.float64] = 1.0 - expected_accuracy
    return result

from collections.abc import Hashable, Mapping

import cmdstanpy
import pandas as pd
import pytest

from pyrater.data.annotation_data import AnnotationData


def _cmdstan_available() -> bool:
    try:
        cmdstanpy.cmdstan_path()
    except Exception:
        return False
    return True


@pytest.fixture(autouse=True)
def _skip_if_no_cmdstan() -> None:
    if not _cmdstan_available():
        pytest.skip("CmdStan is not installed")


@pytest.fixture(scope="session")
def tiny_data() -> AnnotationData:
    """I=8, J=3, K=2, N=24: small enough to fit in well under a second per
    chain, with enough annotator-disagreement structure (ann2 is noisy) to
    be a non-trivial exercise for every model, including the ones with
    annotator-ability parameters."""
    true_labels = ["A", "A", "A", "A", "B", "B", "B", "B"]
    responses = {
        "ann0": ["A", "A", "A", "B", "B", "B", "B", "B"],  # 1 error
        "ann1": ["A", "A", "B", "A", "B", "B", "A", "B"],  # 2 errors
        "ann2": ["A", "B", "A", "B", "A", "B", "A", "B"],  # noisy
    }
    rows = [
        {"item_id": f"item_{i}", "annotator_id": ann, "label": responses[ann][i]}
        for i in range(len(true_labels))
        for ann in responses
    ]
    gold: Mapping[Hashable, Hashable] = {f"item_{i}": label for i, label in enumerate(true_labels)}
    return AnnotationData.from_long_df(pd.DataFrame(rows), gold=gold)

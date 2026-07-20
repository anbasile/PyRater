"""Maintainer-only script: regenerates the bundled `synthetic_small` dataset
used for fast tests and tutorials. Not part of the installed package.

Run with:
    uv run python scripts/generate_synthetic_dataset.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "src" / "pyrater" / "data" / "datasets"

CLASSES = ["entailed", "not-entailed", "unknown"]
N_ITEMS = 12
ANNOTATORS = ["ann_0", "ann_1", "ann_2", "ann_3", "ann_spammer"]
SPAMMER_ID = "ann_spammer"


def main() -> None:
    rng = np.random.default_rng(42)
    true_labels = [CLASSES[i % 3] for i in range(N_ITEMS)]

    rows: list[tuple[str, str, str]] = []
    for item_idx, true_label in enumerate(true_labels):
        item_id = f"item_{item_idx:02d}"
        for annotator in ANNOTATORS:
            if annotator == SPAMMER_ID:
                # A spammer: mostly parrots one class regardless of the truth.
                label = "unknown" if rng.random() < 0.8 else str(rng.choice(CLASSES))
            # A reasonably reliable annotator: usually right, sometimes not.
            elif rng.random() < 0.85:
                label = true_label
            else:
                other_classes = [c for c in CLASSES if c != true_label]
                label = str(rng.choice(other_classes))
            rows.append((item_id, annotator, label))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUT_DIR / "synthetic_small.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "annotator_id", "label"])
        writer.writerows(rows)

    with (OUT_DIR / "synthetic_small_gold.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "label"])
        for item_idx, true_label in enumerate(true_labels):
            writer.writerow([f"item_{item_idx:02d}", true_label])

    print(f"Wrote {len(rows)} annotations and {N_ITEMS} gold labels to {OUT_DIR}")


if __name__ == "__main__":
    main()

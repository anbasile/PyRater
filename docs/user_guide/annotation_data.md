# Annotation data

`AnnotationData` is the canonical, immutable representation every model,
diagnostic, and the CLI agree on.

## Loading your data

```python
from pyrater import AnnotationData

# Long format: one row per (item, annotator, label).
data = AnnotationData.from_csv("annotations.csv")
# or, from an in-memory DataFrame:
data = AnnotationData.from_long_df(df, item_col="item_id", annotator_col="annotator_id", label_col="label")

# Wide format: one row per item, one column per annotator, NaN = no annotation.
data = AnnotationData.from_wide_df(df, item_col="item_id")

# The real Snow et al. (2008) Mechanical Turk RTE release's two-file layout.
data = AnnotationData.from_snow2008("rte.anno.csv", "rte1.tsv")
```

Item, annotator, and class identifiers can be any hashable value on the way
in (strings, ints, UUIDs) — they're remapped internally to dense integer
ranges and decoded back on the way out. Class ordering is deterministic:
sorted by first appearance in the input, never alphabetical and never
random, so re-running the same script twice gives the same internal-to-
external mapping.

## Gold labels

```python
data = AnnotationData.from_long_df(df, gold={"item_1": "positive", "item_5": "negative"})
data.gold  # dict[int, int]: internal item id -> internal class id
```

Gold need not cover every item — usually it's a small stratified subsample,
used to anchor a model's arbitrary internal class numbering to the real
class names (see [Diagnostics](diagnostics.md)).

## Validation

- Fewer than 2 classes, an item with zero annotations, or an annotator with
  zero annotations (in wide format) all raise `ValueError` at construction
  time, naming the offender.
- A gold label pointing at an item or class that was never actually
  annotated also raises `ValueError` — likely a data bug, not something to
  silently ignore.

## Filtering

```python
data.filter_items(["item_1", "item_2"])
data.filter_annotators(min_annotations=5)
```

Both return a new `AnnotationData`; the original is untouched.

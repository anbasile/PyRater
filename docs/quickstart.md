# Quickstart

```python
import pyrater as pyr

# A small bundled synthetic dataset: 12 items, 5 annotators, 3 classes.
data = pyr.load_dataset("synthetic_small")

model = pyr.PyRaterModel.get("d&s")
model.fit(data)

model.labels           # (12,) predicted true label per item
model.labels_proba     # (12, 3) full posterior distribution per item
model.difficulty        # (12,) how hard each item was to annotate
model.credibilities     # (5,) how trustworthy each annotator was

model.convergence_.ok        # did R-hat look fine?
print(model.convergence_.summary())
```

## Comparing against a baseline

```python
baseline = pyr.PyRaterModel.get("majority_voting")
baseline.fit(data)

agree = (model.labels == baseline.labels).mean()
print(f"D&S and majority voting agree on {agree:.1%} of items")
```

## From the command line

```bash
pyrater --list-models
pyrater -m d&s path/to/annotations.csv -o results/
```

See the [User Guide](user_guide/annotation_data.md) for loading your own
data, and [Models &rarr; Overview](models/overview.md) for picking the right
model.

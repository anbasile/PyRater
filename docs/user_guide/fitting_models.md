# Fitting models

## The registry

```python
from pyrater import PyRaterModel

PyRaterModel.list_available()
# ['d&s', 'hier_d&s', 'mace', 'multinomial', 'item_difficulty', 'lre',
#  'majority_voting', 'kappa_majority_voting']

model = PyRaterModel.get("d&s")
```

`get("typo")` raises `KeyError` naming the valid options.

## `fit()`

```python
model.fit(
    data,
    gold=None,             # optional dict of item -> true label, for alignment
    chains=4,
    warmup=1000,
    iters=1000,
    seed=None,              # None = a fresh random seed every call
    align=True,
    check_rhat=True,
    rhat_threshold=1.01,
)
```

`fit()` always returns `self`, so it composes naturally:
`model = PyRaterModel.get("d&s").fit(data)`.

Every property (`labels`, `labels_proba`, `difficulty`, `credibilities`,
`idata`) raises `pyrater.exceptions.NotFittedError` until `fit()` has been
called.

## What each property means

| Property | Shape | Meaning |
|---|---|---|
| `labels` | `(I,)` | Predicted true class per item, in your original label space |
| `labels_proba` | `(I, K)` | Full posterior class distribution per item |
| `difficulty` | `(I,)` | How hard each item was to classify (0 = easy, 1 = maximally uncertain) |
| `credibilities` | `(J,)` | Per-annotator agreement with the model's own inferred labels |

`credibilities` is computed the *same way* for every model, including
`multinomial` and `item_difficulty` (whose Stan files have no annotator
dimension at all): the proportion of an annotator's own raw annotations
that match the model's inferred label for that item. It only raises
`ModelCapabilityError` if the input data has no annotator identifiers at
all — a property of the *data*, not of whichever model you picked.

## Registering your own model

```python
@PyRaterModel.register("my-model")
class MyModel(PyRaterModel):
    class_indexed_params = {"pi": 0, "q_z": 1}

    def _sample(self, data, *, chains, warmup, iters, seed, **kwargs):
        ...  # return an arviz.InferenceData with a `q_z[I, K]` posterior variable
```

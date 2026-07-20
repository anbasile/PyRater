# PyRater

PyRater is a Python toolkit for annotation analysis with Bayesian models:
given a dataset labeled by multiple annotators, it answers three questions:

- **Q1** — what is the most likely true label for each item?
- **Q2** — which items were hardest to annotate?
- **Q3** — which annotators are most reliable?

It implements six probabilistic models of annotation (Dawid & Skene, MACE,
Hierarchical Dawid & Skene, Item Difficulty, Logistic Random Effects, and a
pooled Multinomial baseline), all fit via [Stan](https://mc-stan.org/)
through [CmdStanPy](https://mc-stan.org/cmdstanpy/), behind one consistent
API:

```python
import pyrater as pyr

data = pyr.load_dataset("synthetic_small")
model = pyr.PyRaterModel.get("d&s")
model.fit(data)

model.labels          # Q1: the predicted true label per item
model.difficulty       # Q2: how hard each item was to annotate
model.credibilities    # Q3: how trustworthy each annotator was
```

See [Quickstart](quickstart.md) to get running in a few minutes, or
[Models &rarr; Overview](models/overview.md) for a plain-English guide to
picking the right model.

# NumPyro backend (`np_d&s`, `np_mace`, `np_hier_d&s`, `np_nested`)

Four models written in [NumPyro](https://num.pyro.ai/) on
[JAX](https://docs.jax.dev/) rather than in Stan. They need no CmdStan and
no C++ toolchain, but they do need an optional extra:

```bash
pip install "pyrater[jax]"
```

Without it the four names are still registered and listed, and fitting one
raises an `ImportError` that says what to install.

| Registry name | Model | Parameters in the posterior |
|---|---|---|
| `np_d&s` | Dawid & Skene (1979) | `pi`, `confusion` |
| `np_mace` | MACE (Hovy et al., 2013) | `pi`, `competence`, `spam` |
| `np_hier_d&s` | Hierarchical Dawid & Skene (Paun et al., 2018) | `pi`, `mean`, `scale`, `raw` |
| `np_nested` | Grouped annotators that share an opinion | `pi`, `kind_confusion`, `confusion` |

Every posterior also holds `q_z`, the class distribution of each item, and
`theta`, one summary number per annotator.

Each model sums the latent class out by hand, so the log density is one
matrix product per observed label, on a dense items x annotators matrix.
That suits data where most annotators label most items, such as many
classifiers or prompts run over one test set. The cost of a gradient grows
with items x annotators x classes however few of the cells are filled.

## These are not the Stan models under another name

`np_d&s`, `np_mace` and `np_hier_d&s` have the same likelihood as `d&s`,
`mace` and `hier_d&s`. **Their priors differ, and so do some parameters and
their names.** A posterior from one backend is not a replication of the
other, and results should name the model that produced them.

| | Stan (`d&s`, `mace`, `hier_d&s`) | NumPyro (`np_*`) |
|---|---|---|
| Class prior `pi` | `d&s`, `hier_d&s`: Dirichlet(1). `mace`: not a parameter, fixed at uniform | Always learnt, Dirichlet(2) |
| D&S confusion rows | `beta[j, k]` ~ Dirichlet(1, ..., 1) | `confusion[j, k]` ~ Dirichlet(1, ..., 3, ..., 1), the 3 on the diagonal |
| MACE competence | `theta[j]` ~ Beta(0.5, 0.5) | `competence[j]` ~ Beta(2, 2) |
| MACE spamming distribution | `epsilon[j]` ~ Dirichlet(10) | `spam[j]` ~ Dirichlet(5) |
| Hierarchical D&S population mean | `zeta` ~ Normal(0, 1), on K - 1 free logits per row | `mean` ~ Normal(2 on the diagonal and 0 off it, 2), on K logits per row |
| Hierarchical D&S population scale | `Omega` ~ half-Normal(1) | `scale` ~ half-Normal(1) |
| Hierarchical D&S annotator offsets | `beta_raw` ~ Normal(0, 1) | `raw` ~ Normal(0, 1) |
| `theta` in the posterior | a parameter of `mace` only | always present; a derived summary, except in `np_mace` where it equals `competence` |
| Class axis | `d&s`, `hier_d&s`: fixed after sampling by [alignment](../user_guide/diagnostics.md) | fixed by the priors, which favour reporting the true class; no alignment, `gold=` has no effect |
| Posterior predictive draws (`z_rep`, `rep_vote_hist`) | yes | no |
| Inference | NUTS | NUTS, or the posterior mode with `method="map"` |

Two consequences are worth spelling out.

- The diagonal-favouring priors are informative. They say annotators are
  better than chance, which is what pins the class axis down without gold
  labels, and it works against an annotator who systematically inverts the
  labels.
- `np_hier_d&s` leaves all K logits of a row free, so they are identified
  only up to an additive constant, held in place by the priors. The Stan
  model pins the last logit to zero instead.

No comparison of the two backends' posteriors on the same data has been run.

## `fit()`

The NumPyro models have their own, explicit `fit` signature. Unlike the
Stan-backed models, which forward unknown keyword arguments to CmdStanPy, a
misspelt argument here raises `TypeError`.

```python
model = PyRaterModel.get("np_d&s").fit(
    data,
    method="nuts",        # or "map"
    chains=4, warmup=1000, iters=1000,   # NUTS only
    seed=None,            # None draws a seed; the one used is model.seed_
    anchors=None,         # {item_id: class_label} for items whose class is known
    groups=None,          # {annotator_id: group_id}; np_nested only
    map_steps=1500, learning_rate=0.05,  # MAP only
    mcmc_kwargs=None,     # further numpyro.infer.MCMC arguments, NUTS only
    check_rhat=True, rhat_threshold=1.01,
)
```

Arguments that are valid but do not apply are accepted and ignored, so one
call can serve several models and both methods: `groups` on a model without
groups (a debug-level log message says so), `chains`/`warmup`/`iters` and
`mcmc_kwargs` under `method="map"`, `map_steps`/`learning_rate` under NUTS.

`gold=` is the exception that warns. These models run no label alignment, so
gold labels would be silently unused; passing them emits a `UserWarning`.
To let known labels inform the fit, pass them as `anchors`: each anchored
item enters the likelihood as an observation of its class.

The data must have at most one label per annotator and item. A repeated
pair raises `ValueError` rather than keeping one of the labels arbitrarily.

### `method="nuts"`

Samples with the No-U-Turn sampler; chains run one after another. The
returned `idata` has a `sample_stats` group with `diverging`, `energy`,
`n_steps`, `tree_depth`, `acceptance_rate`, `step_size` and `lp` for every
draw:

```python
int(model.idata["sample_stats"]["diverging"].sum())   # divergent transitions
model.convergence_.ok                                   # R-hat, as for Stan
```

### `method="map"`

Finds the posterior mode with Adam and returns it as a posterior of one
chain and one draw, in seconds. There is no posterior uncertainty in it:
`labels_proba` is the class distribution at the mode.

One draw has no R-hat, so the R-hat check is skipped and `convergence_` is
`None` whatever `check_rhat` says. The diagnostic to read instead is the
loss trace:

```python
model.map_losses_      # the loss at each of the map_steps steps
model.map_converged_   # False if it was still moving at the end
```

`map_converged_` compares the mean loss over the last 100 steps with the
mean over the 100 before; it is `True` when they differ by at most 1e-4 of
the loss. A `ConvergenceWarning` is emitted when it is `False`; raise
`map_steps` then. The flag is deliberately coarse: it catches a run that was
stopped early, and it is not a proof that the optimiser sits at the mode.
When a mode lies on the boundary of a simplex, the loss keeps creeping down
in its fifth or sixth digit for as long as the optimiser runs.

### Reproducibility and precision

An integer `seed` always reproduces the same fit on the same machine and
library versions. `seed=None` draws a fresh seed and records it in
`model.seed_`.

Fits run in 64-bit floating point: the summed log density of millions of
votes loses digits in float32. JAX's `jax_enable_x64` flag is process-wide,
so `fit` switches it on and puts it back to its previous value when it
returns. Other JAX code running in another thread during a fit sees the
flag change.

## `np_nested`: annotators that are not independent

Dawid & Skene and its relatives assume annotators err independently given
the true class. That fails when several "annotators" are one model under
different prompts, or checkpoints of one model: their errors are shared, and
a model that counts them as independent votes becomes overconfident in
whatever they agree on.

`np_nested` puts a latent *opinion* between the true class and the votes.
Each group forms one opinion per item, through the group's confusion matrix;
each annotator in the group reports that opinion through its own.

```
pi                   ~ Dirichlet(2, ..., 2)
kind_confusion[t, k] ~ Dirichlet(1, ..., 3, ..., 1)
confusion[j, k]      ~ Dirichlet(1, ..., 3, ..., 1)
z[i]                 ~ Categorical(pi)
o[i, g]              ~ Categorical(kind_confusion[kind(g), z[i]])
y[i, j]              ~ Categorical(confusion[j, o[i, group(j)]])
```

Both `z` and the opinions `o` are summed out. Note that `confusion[j]` is
conditional on the group's opinion, not on the true class; `theta[j]` still
refers to the true class.

```python
groups = {
    "gpt:prompt-a": "gpt", "gpt:prompt-b": "gpt", "gpt:prompt-c": "gpt",
    "worker-17": ("human", "worker-17"),
    "worker-52": ("human", "worker-52"),
}
model = PyRaterModel.get("np_nested").fit(data, groups=groups, method="map")
```

`groups` is required and must cover every annotator in the data. A group id
may be a `(kind, name)` tuple: groups of the same kind share one confusion
matrix. That is how anonymous annotators from one population are written.
In the example each worker is a group of their own, so the workers stay
independent of each other, and there is one `kind_confusion` matrix for all
humans.

The model was written for this library. Its dependence structure is that of
Jaffe et al. (2016), who detect the groups from the data; here the groups
are given and every parameter has a prior.

## References

- A. P. Dawid and A. M. Skene (1979). Maximum Likelihood Estimation of
  Observer Error-Rates Using the EM Algorithm. *Applied Statistics* 28(1),
  20–28.
- D. Hovy, T. Berg-Kirkpatrick, A. Vaswani and E. Hovy (2013). Learning Whom
  to Trust with MACE. *NAACL-HLT 2013*, 1120–1130.
- S. Paun, B. Carpenter, J. Chamberlain, D. Hovy, U. Kruschwitz and M. Poesio
  (2018). Comparing Bayesian Models of Annotation. *Transactions of the ACL*
  6, 571–585.
- A. Jaffe, E. Fetaya, B. Nadler, T. Jiang and Y. Kluger (2016).
  Unsupervised Ensemble Learning with Dependent Classifiers. *AISTATS 2016*,
  PMLR 51, 351–360.

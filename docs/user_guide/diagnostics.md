# Diagnostics

## Label switching

Stan has no idea which of its `K` latent class slots corresponds to which
real-world class name — different MCMC chains can settle on different,
equally valid internal numberings. `fit()` fixes this automatically, in two
steps:

1. **Chain alignment** (`pyrater.diagnostics.label_switching.align_chains`):
   every chain is realigned against a reference chain (chain 0), using the
   Hungarian algorithm on each chain's agreement with the reference — an
   optimal, polynomial-time replacement for a brute-force search over all
   `K!` permutations. This runs whenever a model has `chains > 1` and
   declares any `class_indexed_params`; it's a correctness requirement, not
   an optional diagnostic — averaging unaligned chains is statistically
   meaningless.
2. **Gold alignment** (`align_to_gold`), if gold labels are available (via
   `fit(gold=...)` or `data.gold`): one further global permutation, found
   from a small class-stratified sample (5 items per class by default),
   anchoring the now-mutually-aligned chains to the real class names.

`mace` is immune to this entirely — its likelihood anchors the class axis
directly to the observed label codes, so there's nothing to realign. The
NumPyro-backed models (`np_*`) are not realigned either: their priors favour
reporting the true class, and that is what fixes the class axis. Gold labels
have no effect on them, and passing `gold=` warns.

## Convergence

```python
model.convergence_.ok
print(model.convergence_.summary())
```

`check_convergence` wraps `arviz.rhat` with `method="rank"` (rank-normalized
split-R-hat, the current best-practice diagnostic), comparing every
individual parameter entry against `rhat_threshold` (default `1.01`). A
failed check emits a `pyrater.exceptions.ConvergenceWarning` — a real,
filterable Python warning — but **never blocks access to results**: a bad
R-hat means "trust these numbers less," not "these numbers don't exist."

Pass `check_rhat=False` to `fit()` to skip the check entirely (`convergence_`
stays `None`).

The NumPyro-backed models add two things. After NUTS, `model.idata` has a
`sample_stats` group (divergent transitions, tree depth, energy, acceptance
rate). After `fit(method="map")` there is a single draw and so no R-hat:
`convergence_` is `None`, and `model.map_converged_` and `model.map_losses_`
report on the optimiser instead. See
[NumPyro backend](../models/numpyro.md#methodmap).

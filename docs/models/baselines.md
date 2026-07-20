# Baselines

Two non-Bayesian baselines, registered under the same API, so "is the
Bayesian machinery worth it" is always a one-line comparison against
something free:

## `majority_voting`

Not a Stan model — pure, instant NumPy. `.fit()` ignores every MCMC-sampling
keyword. `labels` is the per-item mode; `difficulty` is the disagreement
ratio (`1 - majority_count / total`); `credibilities` is inherited
unchanged from the base class's uniform formula, which for this model
reduces exactly to "how often did this annotator agree with the majority."

## `kappa_majority_voting`

Same `labels`/`difficulty` as `majority_voting`, but `credibilities` is
overridden with each annotator's mean pairwise Cohen's kappa against every
other annotator on commonly-annotated items (hand-rolled, no scikit-learn
dependency for one metric) — reproducing the paper's own Kappa-vs-majority-
voting comparison (Table 2). Returns `NaN` for an annotator who shares fewer
than 2 commonly-annotated items with any other annotator.

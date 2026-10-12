# Changelog

## 0.2.0

### Added

- **A NumPyro backend**, behind the optional `jax` extra
  (`pip install "pyrater[jax]"`): `np_d&s`, `np_mace`, `np_hier_d&s`, and
  `np_nested`, a new model for annotators that are not independent given the
  class (one model under several prompts, say). They fit by NUTS or, with
  `fit(method="map")`, by finding the posterior mode; `fit(anchors=...)`
  takes items whose class is known. See
  [NumPyro backend](models/numpyro.md). The first three share a likelihood
  with `d&s`, `mace` and `hier_d&s` but have different priors and parameter
  names: they are not the same models, and that page lists the differences.
- The NumPyro models have an explicit `fit` signature (`method`, `groups`,
  `anchors`, `map_steps`, `learning_rate`, `mcmc_kwargs`); an unknown keyword
  argument raises `TypeError`.
- Diagnostics for those models: a `sample_stats` group (divergences, energy,
  tree depth, acceptance rate, step size, log density) after NUTS;
  `map_losses_` and `map_converged_` after MAP, with a `ConvergenceWarning`
  when the loss was still moving; `seed_`, the seed a fit used.
- Posterior predictive checks in all six Stan models (`z_rep`,
  `rep_vote_hist` in the posterior), and a pointwise `log_lik` per item.
- `irt_2pl_ss`, the IRT model with sensitivity and specificity of Han,
  Adiguzel & Carpenter (arXiv:2405.19521), for binary labels.

### Fixed

- **`load_dataset("rte")` returned the wrong labels.** It read
  `rte.anno.csv`, which holds 2,040 of the 8,000 labels and whose `value`
  column is the gold label, not the worker's. It now reads
  `all_collected_data/rte.standardized.tsv` from the same release: 800
  items, 164 workers, 10 labels per item, gold for every item. Anything
  computed from the old loader's output should be recomputed.

### Changed

- **`AnnotationData.from_snow2008` reads the release's
  `<task>.standardized.tsv` layout** (`!amt_worker_ids`, `orig_id`,
  `response`, `gold`) and takes one path. The old two-file form,
  `from_snow2008(annotations, gold_path)`, is gone, because the layout it
  read does not hold the workers' labels. Labels keep the file's own coding
  (for RTE, the integers `1` and `0`, where the old loader produced the
  strings `"True"` and `"False"`). Gold labels come from the file's `gold`
  column; `gold=` replaces them.
- The command line's `--format snow2008` follows: it no longer requires
  `--gold`.
- The posterior predictive additions changed the realised draws of the Stan
  models for a given seed. The posterior distribution is the same.

## 0.1.0

Initial implementation: `AnnotationData`, the `PyRaterModel` registry and
fitting orchestration, label-switching correction and convergence
diagnostics, all six paper models (`d&s`, `hier_d&s`, `mace`, `multinomial`,
`item_difficulty`, `lre`), two baselines (`majority_voting`,
`kappa_majority_voting`), a bundled synthetic dataset plus a downloadable
`rte` loader, plotting, a CLI, and a Docker image with CmdStan preinstalled.

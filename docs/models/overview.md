# Choosing a model

| Model | Registry name | Models annotators? | Models item difficulty natively? | Good for |
|---|---|---|---|---|
| Multinomial | `multinomial` | No | No | A fast, pooled baseline |
| Dawid & Skene | `d&s` | Yes (unpooled) | No | The default choice, with enough data per annotator |
| MACE | `mace` | Yes (trustworthy/spamming) | No | Noisy crowds with adversarial or careless annotators |
| Hierarchical D&S | `hier_d&s` | Yes (pooled toward a population mean) | No | A long tail of low-count annotators |
| Item Difficulty | `item_difficulty` | No | Yes | When Q2 (item difficulty) is the main interest |
| Logistic Random Effects | `lre` | Yes | Yes | When you need all three answers well-estimated together |

Every model exposes `.labels`/`.difficulty`/`.credibilities` through the
same API (see [Fitting models](../user_guide/fitting_models.md)) —
`difficulty` falls back to posterior entropy for models with no native
difficulty parameter, and `credibilities` is computed the same way
(post-inference agreement with the model's own inferred labels) for every
model, including the two with no annotator dimension at all.

Two non-Bayesian baselines are also registered, for comparison:
[`majority_voting` and `kappa_majority_voting`](baselines.md).

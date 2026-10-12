# API reference

## Data and models

::: pyrater.data.annotation_data.AnnotationData

::: pyrater.models.base.PyRaterModel

## NumPyro backend

Needs the `jax` extra; see [NumPyro backend](models/numpyro.md).

::: pyrater.models._numpyro_backed.NumpyroBackedModel
    options:
      members:
        - fit

::: pyrater.models._numpyro_backed.map_loss_converged

::: pyrater.models.numpyro_models.NumpyroDawidSkene
    options:
      members: false

::: pyrater.models.numpyro_models.NumpyroMace
    options:
      members: false

::: pyrater.models.numpyro_models.NumpyroHierDawidSkene
    options:
      members: false

::: pyrater.models.numpyro_models.NumpyroNested
    options:
      members: false

## Diagnostics

::: pyrater.diagnostics.label_switching
    options:
      members:
        - align_chains
        - align_to_gold
        - match_permutation

::: pyrater.diagnostics.convergence
    options:
      members:
        - check_convergence
        - ConvergenceReport

## Plotting

::: pyrater.plotting
    options:
      members:
        - credibility_plot
        - posterior_plot

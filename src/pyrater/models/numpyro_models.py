"""Annotation models on the NumPyro backend.

    np_d&s        Dawid & Skene (1979): one confusion matrix per annotator
    np_mace       Hovy et al. (2013): a competence and a spamming distribution
    np_hier_d&s   confusion matrices partially pooled across annotators
    np_nested     annotators grouped; a group holds one opinion per item and
                  its annotators are noisy readings of it

All four need the `jax` extra (`pip install "pyrater[jax]"`) and share the
`fit` signature of :class:`~pyrater.models._numpyro_backed.NumpyroBackedModel`.

The first three have the same likelihood as the Stan models of the same name
without the `np_` prefix (`d&s`, `mace`, `hier_d&s`), but they are NOT the same
models: the priors differ, and so do some of the parameters and their names.
Posteriors from the two backends are therefore not interchangeable, and a
number reported for `np_mace` is not a number for `mace`. Every class docstring
below lists the differences from its Stan counterpart. What they have in
common:

* The Stan files are the implementations of Paun et al. (2018). The priors of
  `d&s` and `hier_d&s` are symmetric in the classes, so their class axis is
  only identified up to a permutation and pyrater aligns it after sampling
  (see `pyrater.diagnostics.label_switching`). The priors here favour
  reporting the true class instead, which is what holds the class axis in
  place. No alignment runs, and `fit(gold=...)` has no effect.
* The class prior `pi` is always learnt, under a Dirichlet(2, ..., 2) prior.
* The posterior always holds `theta`, one summary number per annotator, next
  to the model's own parameters. In Stan's `mace`, `theta` is a parameter.
* There are no posterior predictive draws (`z_rep`, `rep_vote_hist`).
* `fit(method="map")` returns the posterior mode; Stan only samples.

`np_nested` has no Stan counterpart: it is for raters that are not independent
given the class because several of them are the same model under different
prompts.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, ClassVar

import numpy as np

from pyrater.data.annotation_data import AnnotationData
from pyrater.models._numpyro_backed import NumpyroBackedModel, dense_votes, vote_loglik
from pyrater.models.base import PyRaterModel


def _diagonal_prior(n_classes: int, strength: float) -> Any:
    """Dirichlet concentrations for the rows of a confusion matrix: 1 off the
    diagonal and `1 + strength` on it, so reporting the true class is favoured."""
    import jax.numpy as jnp

    return jnp.ones((n_classes, n_classes)) + strength * jnp.eye(n_classes)


@PyRaterModel.register("np_d&s")
class NumpyroDawidSkene(NumpyroBackedModel):
    """Dawid & Skene's model: every annotator has their own confusion matrix.

    Generative model, for items i, annotators j and K classes::

        pi              ~ Dirichlet(2, ..., 2)
        confusion[j, k] ~ Dirichlet(1, ..., 3, ..., 1)     the 3 at position k
        z[i]            ~ Categorical(pi)
        y[i, j]         ~ Categorical(confusion[j, z[i]])  for each observed vote

    The class `z[i]` is summed out; its posterior given the parameters is
    `q_z`. `theta[j]` is the mean of the diagonal of `confusion[j]`.

    Differences from the Stan model `d&s` (`DS.stan`):

    * `confusion[j, k]` there is called `beta[j, k]`, and its prior is the
      uniform Dirichlet(1, ..., 1), not one that favours the diagonal.
    * `pi` there has a Dirichlet(1, ..., 1) prior, not Dirichlet(2, ..., 2).
    * `theta` does not exist there.

    Reference: A. P. Dawid and A. M. Skene (1979), "Maximum Likelihood
    Estimation of Observer Error-Rates Using the EM Algorithm", Applied
    Statistics 28(1), 20-28.
    """

    def _priors(self, n_classes: int, tensors: dict[str, Any]) -> dict[str, Any]:
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        pi = numpyro.sample("pi", dist.Dirichlet(2.0 * jnp.ones(n_classes)))
        confusion = numpyro.sample(
            "confusion",
            dist.Dirichlet(_diagonal_prior(n_classes, 2.0))
            .expand([tensors["n_annotators"], n_classes])
            .to_event(2),
        )
        return {"pi": pi, "confusion": confusion}

    def _confusion(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        return params["confusion"]

    def _item_loglik(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        import jax.numpy as jnp

        log_confusion = jnp.log(self._confusion(params, tensors))
        return vote_loglik(tensors["indicators"], log_confusion) + jnp.log(params["pi"])


@PyRaterModel.register("np_mace")
class NumpyroMace(NumpyroDawidSkene):
    """MACE: an annotator either reports the true class or spams.

    Generative model, for items i, annotators j and K classes::

        pi            ~ Dirichlet(2, ..., 2)
        competence[j] ~ Beta(2, 2)
        spam[j]       ~ Dirichlet(5, ..., 5)
        z[i]          ~ Categorical(pi)
        s[i, j]       ~ Bernoulli(competence[j])
        y[i, j]       = z[i]                   if s[i, j] = 1
        y[i, j]       ~ Categorical(spam[j])   otherwise

    Both `z` and `s` are summed out, which gives each annotator the confusion
    matrix `competence[j] * I + (1 - competence[j]) * spam[j]` (every row
    shares the same `spam[j]`). `theta[j]` is `competence[j]` itself.

    Differences from the Stan model `mace` (`MACE.stan`):

    * `competence` there is called `theta` and has a Beta(0.5, 0.5) prior,
      which puts its mass near 0 and 1; Beta(2, 2) here puts it near 0.5.
    * `spam` there is called `epsilon` and has a Dirichlet(10, ..., 10)
      prior, not Dirichlet(5, ..., 5).
    * The class prior there is not a parameter: it is fixed at uniform
      (1/K each), as in the original MACE. Here `pi` is learnt.

    Reference: D. Hovy, T. Berg-Kirkpatrick, A. Vaswani and E. Hovy (2013),
    "Learning Whom to Trust with MACE", NAACL-HLT 2013, 1120-1130.
    """

    def _priors(self, n_classes: int, tensors: dict[str, Any]) -> dict[str, Any]:
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        n_annotators = tensors["n_annotators"]
        pi = numpyro.sample("pi", dist.Dirichlet(2.0 * jnp.ones(n_classes)))
        competence = numpyro.sample(
            "competence", dist.Beta(2.0, 2.0).expand([n_annotators]).to_event(1)
        )
        spam = numpyro.sample(
            "spam", dist.Dirichlet(5.0 * jnp.ones(n_classes)).expand([n_annotators]).to_event(1)
        )
        return {"pi": pi, "competence": competence, "spam": spam}

    def _confusion(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        import jax.numpy as jnp

        competence = params["competence"][:, None, None]
        n_classes = params["spam"].shape[-1]
        return competence * jnp.eye(n_classes)[None] + (1 - competence) * params["spam"][:, None, :]

    def _competence(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        return params["competence"]


@PyRaterModel.register("np_hier_d&s")
class NumpyroHierDawidSkene(NumpyroDawidSkene):
    """Hierarchical Dawid & Skene: confusion matrices drawn from a population.

    Generative model, for items i, annotators j and K classes, with every
    confusion-matrix row a softmax of K logits (non-centred)::

        pi              ~ Dirichlet(2, ..., 2)
        mean[k, l]      ~ Normal(2 if k = l else 0, 2)
        scale[k, l]     ~ HalfNormal(1)
        raw[j, k, l]    ~ Normal(0, 1)
        confusion[j, k] = softmax(mean[k] + scale[k] * raw[j, k])
        z[i]            ~ Categorical(pi)
        y[i, j]         ~ Categorical(confusion[j, z[i]])

    The posterior holds `mean`, `scale` and `raw`, not `confusion`, which is
    a deterministic function of them. `theta[j]` is the mean of the diagonal
    of `confusion[j]`.

    Differences from the Stan model `hier_d&s` (`HDS-NC.stan`):

    * `mean`, `scale` and `raw` there are called `zeta`, `Omega` and
      `beta_raw`.
    * Each row there has K - 1 free logits, the last one pinned to 0. Here all
      K are free, so a row's logits are only identified up to an additive
      constant, and that constant is held in place by the priors alone.
    * `zeta` there has a Normal(0, 1) prior in every cell: narrower, and with
      no preference for the diagonal. `Omega` (half-normal(1)) and `beta_raw`
      (Normal(0, 1)) have the same priors as `scale` and `raw`.
    * `pi` there has a Dirichlet(1, ..., 1) prior, not Dirichlet(2, ..., 2).
    * `theta` does not exist there; the log confusion matrices are stored as
      `beta_norm`.

    Reference: S. Paun, B. Carpenter, J. Chamberlain, D. Hovy, U. Kruschwitz
    and M. Poesio (2018), "Comparing Bayesian Models of Annotation",
    Transactions of the ACL 6, 571-585.
    """

    def _priors(self, n_classes: int, tensors: dict[str, Any]) -> dict[str, Any]:
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        shape = [n_classes, n_classes]
        pi = numpyro.sample("pi", dist.Dirichlet(2.0 * jnp.ones(n_classes)))
        mean = numpyro.sample("mean", dist.Normal(2.0 * jnp.eye(n_classes), 2.0).to_event(2))
        scale = numpyro.sample("scale", dist.HalfNormal(1.0).expand(shape).to_event(2))
        raw = numpyro.sample(
            "raw", dist.Normal(0.0, 1.0).expand([tensors["n_annotators"], *shape]).to_event(3)
        )
        return {"pi": pi, "mean": mean, "scale": scale, "raw": raw}

    def _confusion(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        import jax

        return jax.nn.softmax(params["mean"] + params["scale"] * params["raw"], axis=-1)


@PyRaterModel.register("np_nested")
class NumpyroNested(NumpyroBackedModel):
    """Annotators in the same group are not independent given the class.

    A group (one model, say) forms one opinion per item, drawn from the true
    class through the group's confusion matrix. Each of its annotators (that
    model under one prompt) reports the opinion through its own confusion
    matrix. Summing the opinion out costs classes x classes per item and group.
    Pass `groups={annotator_id: group_id}` to `fit`; every annotator in the
    data needs an entry.

    A group id may be a `(kind, name)` tuple. Groups of the same kind then
    share one confusion matrix, which is how anonymous annotators drawn from
    one population are written: each is its own group, so they stay
    independent, but there is one set of parameters for all of them. A group
    id that is not a tuple is a kind by itself (one it shares with any
    `(kind, name)` group whose kind equals it).

    Generative model, for items i, annotators j, groups g and K classes, with
    g(j) the group of annotator j and kind(g) the kind of group g::

        pi                   ~ Dirichlet(2, ..., 2)
        kind_confusion[t, k] ~ Dirichlet(1, ..., 3, ..., 1)   the 3 at position k
        confusion[j, k]      ~ Dirichlet(1, ..., 3, ..., 1)   the 3 at position k
        z[i]                 ~ Categorical(pi)
        o[i, g]              ~ Categorical(kind_confusion[kind(g), z[i]])
        y[i, j]              ~ Categorical(confusion[j, o[i, g(j)]])

    Both the class `z` and the opinions `o` are summed out. Note that
    `confusion[j]` here is conditional on the group's opinion, not on the true
    class. `theta[j]` is still about the true class: the mean of the diagonal
    of `kind_confusion[kind(g(j))] @ confusion[j]`. A group forms an opinion
    only on the items at least one of its annotators labelled; with every
    annotator in a group of its own and of its own kind, the model is Dawid &
    Skene's with each confusion matrix written as a product of two.

    Reference: the model was written for this library and has no counterpart
    among the Stan models. Its dependence structure, a latent variable per
    group between the true class and the group's annotators, is that of A.
    Jaffe, E. Fetaya, B. Nadler, T. Jiang and Y. Kluger (2016), "Unsupervised
    Ensemble Learning with Dependent Classifiers", AISTATS 2016, PMLR 51,
    351-360, who detect the groups from the data; here they are given, and
    every parameter has a prior.
    """

    uses_groups: ClassVar[bool] = True

    def _tensors(
        self, data: AnnotationData, groups: Mapping[Hashable, Hashable] | None
    ) -> dict[str, Any]:
        import jax.numpy as jnp

        if groups is None:
            raise ValueError("np_nested needs `groups`, a mapping from annotator id to group id.")
        votes = dense_votes(data)  # also refuses data without annotator ids
        annotator_ids = [] if data.annotator_ids is None else list(data.annotator_ids)
        missing = [a for a in annotator_ids if a not in groups]
        if missing:
            shown = ", ".join(repr(a) for a in missing[:10])
            more = f" and {len(missing) - 10} more" if len(missing) > 10 else ""
            raise ValueError(
                f"`groups` has no entry for {len(missing)} of the {len(annotator_ids)} "
                f"annotator(s) in the data: {shown}{more}."
            )
        labels = [groups[a] for a in annotator_ids]
        names = list(dict.fromkeys(labels))
        membership = np.array([names.index(g) for g in labels])
        kinds = [g[0] if isinstance(g, tuple) else g for g in names]
        unique_kinds = list(dict.fromkeys(kinds))
        return {
            "kind_of_group": np.array([unique_kinds.index(k) for k in kinds]),
            "n_kinds": len(unique_kinds),
            "membership": membership,
            "members": [np.flatnonzero(membership == g) for g in range(len(names))],
            "n_annotators": data.n_annotators,
            "group_indicators": [
                [
                    jnp.asarray((votes[:, membership == g] == label).astype(np.float64))
                    for label in range(data.n_classes)
                ]
                for g in range(len(names))
            ],
        }

    def _priors(self, n_classes: int, tensors: dict[str, Any]) -> dict[str, Any]:
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        prior = dist.Dirichlet(_diagonal_prior(n_classes, 2.0))
        pi = numpyro.sample("pi", dist.Dirichlet(2.0 * jnp.ones(n_classes)))
        kind_confusion = numpyro.sample(
            "kind_confusion", prior.expand([tensors["n_kinds"], n_classes]).to_event(2)
        )
        confusion = numpyro.sample(
            "confusion", prior.expand([tensors["n_annotators"], n_classes]).to_event(2)
        )
        return {"pi": pi, "kind_confusion": kind_confusion, "confusion": confusion}

    def _group_confusion(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        return params["kind_confusion"][tensors["kind_of_group"]]

    def _confusion(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        # What an annotator reports given the TRUE class: its group's opinion,
        # then its own reading of it.
        import jax.numpy as jnp

        group = self._group_confusion(params, tensors)[tensors["membership"]]
        return jnp.einsum("jcb,jbl->jcl", group, params["confusion"])

    def _item_loglik(self, params: dict[str, Any], tensors: dict[str, Any]) -> Any:
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp

        total = jnp.log(params["pi"])[None]
        log_confusion = jnp.log(params["confusion"])
        log_group = jnp.log(self._group_confusion(params, tensors))
        for g, indicators in enumerate(tensors["group_indicators"]):
            readings = vote_loglik(indicators, log_confusion[tensors["members"][g]])
            opinion = log_group[g][None] + readings[:, None, :]
            total = total + logsumexp(opinion, axis=-1)
        return total

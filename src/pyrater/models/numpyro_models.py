"""Annotation models on the NumPyro backend.

    np_d&s        Dawid & Skene (1979): one confusion matrix per annotator
    np_mace       Hovy et al. (2013): a competence and a spamming distribution
    np_hier_d&s   confusion matrices partially pooled across annotators
    np_nested     annotators grouped; a group holds one opinion per item and
                  its annotators are noisy readings of it

The first three match the Stan models of the same name without the `np_`
prefix. `np_nested` has no Stan counterpart: it is for raters that are not
independent given the class because several of them are the same model under
different prompts.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np

from pyrater.data.annotation_data import AnnotationData
from pyrater.models._numpyro_backed import NumpyroBackedModel, dense_votes, vote_loglik
from pyrater.models.base import PyRaterModel


def _diagonal_prior(n_classes: int, strength: float) -> Any:
    import jax.numpy as jnp

    return jnp.ones((n_classes, n_classes)) + strength * jnp.eye(n_classes)


@PyRaterModel.register("np_d&s")
class NumpyroDawidSkene(NumpyroBackedModel):
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
    Pass `groups={annotator_id: group_id}` to `fit`.

    A group id may be a `(kind, name)` tuple. Groups of the same kind then
    share one confusion matrix, which is how anonymous annotators drawn from
    one population are written: each is its own group, so they stay
    independent, but there is one set of parameters for all of them.
    """

    def _tensors(
        self, data: AnnotationData, groups: Mapping[Hashable, Hashable] | None
    ) -> dict[str, Any]:
        import jax.numpy as jnp

        if groups is None:
            raise ValueError("np_nested needs `groups`, a mapping from annotator id to group id.")
        assert data.annotator_ids is not None
        labels = [groups[a] for a in data.annotator_ids]
        names = list(dict.fromkeys(labels))
        membership = np.array([names.index(g) for g in labels])
        votes = dense_votes(data)
        kinds = [g[0] if isinstance(g, tuple) else g for g in names]
        unique_kinds = list(dict.fromkeys(kinds))
        return {
            "groups": names,
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

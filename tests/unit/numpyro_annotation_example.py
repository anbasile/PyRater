# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
#
# Three models of NumPyro's example "Bayesian Models of Annotation",
# https://github.com/pyro-ppl/numpyro/blob/1e5c0bf8bfbb0bb5c0278e83a530b7167dc8e5fb/examples/annotation.py
# copied unchanged but for the formatting: `dawid_skene`, `mace` and
# `hierarchical_dawid_skene`. They are the reference that the `np_stan_*`
# models are tested against (test_numpyro_stan_twins.py). They enumerate the
# class, which needs `funsor`, and take a complete items x positions matrix.
"""NumPyro's implementations of three models of Paun et al. (2018)."""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import nn
from numpyro import handlers
from numpyro.infer.reparam import LocScaleReparam
from numpyro.ops.indexing import Vindex


def dawid_skene(positions, annotations):
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            beta = numpyro.sample("beta", dist.Dirichlet(jnp.ones(num_classes)))

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi), infer={"enumerate": "parallel"})

        with numpyro.plate("position", num_positions):
            numpyro.sample("y", dist.Categorical(Vindex(beta)[positions, c, :]), obs=annotations)


def mace(positions, annotations):
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("annotator", num_annotators):
        epsilon = numpyro.sample("epsilon", dist.Dirichlet(jnp.full(num_classes, 10)))
        theta = numpyro.sample("theta", dist.Beta(0.5, 0.5))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample(
            "c",
            dist.DiscreteUniform(0, num_classes - 1),
            infer={"enumerate": "parallel"},
        )

        with numpyro.plate("position", num_positions):
            s = numpyro.sample(
                "s",
                dist.Bernoulli(1 - theta[positions]),
                infer={"enumerate": "parallel"},
            )
            probs = jnp.where(s[..., None] == 0, nn.one_hot(c, num_classes), epsilon[positions])
            numpyro.sample("y", dist.Categorical(probs), obs=annotations)


def hierarchical_dawid_skene(positions, annotations):
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        zeta = numpyro.sample("zeta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1))
        omega = numpyro.sample("Omega", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1))

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            with handlers.reparam(config={"beta": LocScaleReparam(0)}):
                beta = numpyro.sample("beta", dist.Normal(zeta, omega).to_event(1))
            beta = jnp.pad(beta, [(0, 0)] * (jnp.ndim(beta) - 1) + [(0, 1)])

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi), infer={"enumerate": "parallel"})

        with numpyro.plate("position", num_positions):
            logits = Vindex(beta)[positions, c, :]
            numpyro.sample("y", dist.Categorical(logits=logits), obs=annotations)

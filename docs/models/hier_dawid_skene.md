# Hierarchical Dawid & Skene (`hier_d&s`)

Same generative story as D&S, but each annotator's confusion matrix is now
drawn from a shared population distribution instead of estimated fully
independently — an annotator who's only labeled 3 items still gets a
sensible, regularized estimate (shrunk toward the population mean) instead
of a wild, overfit one.

Stan file: `HDS-NC.stan` (non-centered parameterization, for sampling
efficiency — see [Diagnostics](../user_guide/diagnostics.md)) —
implementation from Paun et al. (2018)'s supplementary material (CC-BY
4.0), not original to this library; see [Contributing](../contributing.md).

Plate diagram (redrawn from Paun et al. 2018, Figure 4) — same as D&S,
except each annotator's confusion row `β_j,k` is now itself drawn from a
shared, class-level population mean `ζ_k` and scale `Ω_k`:

```mermaid
flowchart LR
    classDef observed fill:#999,stroke:#333,color:#fff;
    classDef latent fill:#fff,stroke:#333,color:#000;

    pi(("π"))
    class pi latent

    subgraph I["items  i = 1..I"]
        c(("c_i"))
        class c latent
        subgraph Ni["n = 1..N_i"]
            y(("y_i,n"))
            class y observed
        end
        c --> y
    end

    subgraph J["annotators  j = 1..J"]
        subgraph K1["classes  k = 1..K"]
            beta(("β_j,k"))
            class beta latent
        end
    end

    subgraph K2["classes  k = 1..K"]
        zeta(("ζ_k"))
        Omega(("Ω_k"))
        class zeta,Omega latent
    end

    pi --> c
    beta --> y
    zeta --> beta
    Omega --> beta
```

**Good for:** datasets with a long tail of annotators contributing very few
labels each — exactly the shape open crowdsourcing produces, and exactly
the case where plain D&S's per-annotator estimates get noisy.

**`difficulty`:** the generic entropy fallback. **`credibilities`:** the
uniform post-inference formula.

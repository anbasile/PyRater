# Item Difficulty (`item_difficulty`)

Flips the emphasis of D&S: instead of modeling *who's* annotating, it
models *what's being annotated*. Every item has its own latent per-class
confusion profile, drawn from a class-level hierarchical prior, with no
annotator structure at all.

Stan file: `ID-NC.stan` — implementation from Paun et al. (2018)'s
supplementary material (CC-BY 4.0), not original to this library; see
[Contributing](../contributing.md).

Plate diagram (redrawn from Paun et al. 2018, Figure 5) — no annotator
plate; instead every item gets its own difficulty profile `θ_i`, drawn from
class-level hierarchical means `η_k` and scales `χ_k`:

```mermaid
flowchart LR
    classDef observed fill:#999,stroke:#333,color:#fff;
    classDef latent fill:#fff,stroke:#333,color:#000;

    pi(("π"))
    class pi latent

    subgraph I["items  i = 1..I"]
        c(("c_i"))
        theta(("θ_i"))
        class c,theta latent
        subgraph Ni["n = 1..N_i"]
            y(("y_i,n"))
            class y observed
        end
        c --> theta
        theta --> y
    end

    subgraph K["classes  k = 1..K"]
        eta(("η_k"))
        chi(("χ_k"))
        class eta,chi latent
    end

    pi --> c
    eta --> theta
    chi --> theta
```

**Good for:** when Q2 (item difficulty) is the primary interest — dataset
curation, error analysis, finding genuinely ambiguous items rather than
just badly-labeled ones.

**`difficulty`:** model-native, not the generic entropy fallback — derived
from the item's own fitted per-class confusion structure
(`theta_norm[I, K, K]`): `1 - sum_h pi[h] * exp(theta_norm[i, h, h])`, the
item's expected "own-class" inaccuracy. **`credibilities`:** still works
via the uniform post-inference formula, even though this Stan file has no
annotator dimension at all.

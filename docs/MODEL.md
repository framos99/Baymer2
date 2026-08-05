# Model options

How the layers are linked, how a node's rate is parameterised, and which
likelihood to use for which kind of data. The mathematics behind all three
is in [METHODS.md](METHODS.md); this page is about choosing between them.

## Linkage: `cut` or `joint`

`cut` (default) is legacy's structure — each layer carries its own multinomial on
pooled counts, so observations enter once per layer and information flows only
downward. `joint` samples every layer in one chain with the data entering
**once**, at the leaves, so a deep context can inform its ancestors.

At k=5 joint is in fact the *faster* of the two (9.8 s vs 16.0 s), because it
never recomputes pooled likelihoods. Its root lands exactly on the pooled
empirical rate, as it must when the data enter once. Cost grows with depth
though: a proposal at layer `l` rescores that node's whole leaf subtree, and all
layers stay resident, so beyond about k=7 prefer `cut`. Currently binomial-only —
the CTMC mode is quartet-local and has no joint kernel.

## Parameterisation: `alr` or `multiplicative`

ALR is the default and removes the simplex constraint structurally.
`multiplicative` keeps legacy's `phi = log(p_v / p_parent)` for anyone comparing
against published phi values. They agree closely — median 0.4–1.2% on rates,
correlation ≥0.997 — because the reference probability is within 1e-3 of one at
mutation rates.

This is **not** legacy's sampler. Legacy redrew an offending proposal up to five
times, which makes the effective proposal a truncated normal whose normalising
constant depends on the current state and never enters the acceptance ratio.
Here a proposal is drawn once and rejected if it leaves the simplex, which is
ordinary Metropolis. Reproducing legacy's numbers bit-for-bit means running
[legacy itself](https://github.com/bvoightlab/Baymer), which still works.

One practical note: at the root of this parameterisation `phi` *is* the rate, so
`phi_proposal_sd` is read as a fraction of the rate rather than an absolute step.
An absolute 0.02 step on a rate of 1e-3 is rejected essentially always, which is
why legacy hardcodes `phi_sigma: 0.00005` for layer 0 in every shipped config.

## Substitution and recurrence modes

`likelihood="ctmc"` models the focal branch as a continuous-time Markov chain, so
back mutation, parallel mutation, and saturation are inverted rather than read as
rate. Simulated branches, 3-mer centred tree, fitted both ways
(`tests/test_substitution_modes.py`):

| branch (subs/site) | binomial bias | CTMC bias | binomial error | CTMC error |
|---|---|---|---|---|
| 0.006 | −0.8% | −0.5% | 2.9% | 3.1% |
| 0.061 | −3.7% | +0.2% | 3.7% | 0.7% |
| 0.182 | −11.2% | +0.1% | 11.2% | 0.5% |
| 0.340 | −19.6% | −0.1% | 19.6% | 0.4% |

The CTMC is essentially unbiased throughout; the binomial degrades roughly with
branch length and is indistinguishable only on very short branches. That
crossover is why the mode is selectable rather than the default.

**The CTMC needs a centred tree.** A 4×4 generator requires rates out of all four
bases at fixed flanks. Folding supplies the `G`/`T` rows — the rate out of `G` in
context `c` is the rate out of `C` in `revcomp(c)` — but only when the focal base
is centred, since reverse complementing moves it from index `fi` to `k−1−fi`. The
alternating tree satisfies that on odd layers only; at k=4, 128 of 512
(context, base) pairs are unreachable. So substitution mode uses
`growth="centred"`, which adds one base on *each* side per layer. It reaches the
same 9-mer leaf set (131,072 contexts) in 5 layers instead of 9.

Because a node's generator draws on three neighbours, the CTMC likelihood is
*quartet-local*. The sampler therefore parallelises over closed quartets rather
than nodes — updating two members of one quartet concurrently would race on a
shared generator, and an acceptance ratio from a single member would be wrong.

`likelihood="recurrent"` models the three alleles as competing Poisson processes,
so the observed mutated fraction is `1 − exp(−L)` rather than `L`. Rates become
expected event counts, removing the `sum < 1` ceiling; at λ ≈ 1 a binomial fit
understates by >15%. Node-local, so it needs no quartet structure.

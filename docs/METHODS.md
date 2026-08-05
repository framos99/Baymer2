# Methods

## 1. Overview

Baymer (Adams et al.) estimates context-dependent germline mutation rates by
fitting a Bayesian hierarchical model over a tree of nested k-mer contexts, in
which each additional flanking nucleotide defines a new layer and the rate at
each node is shrunk toward its parent through a spike-and-slab prior. We
reimplemented the method as `baymer2`, retaining the hierarchical structure and
the spike-and-slab shrinkage while (i) correcting nine defects in the sampler,
(ii) reparameterising the rate vector to eliminate a violation of detailed
balance, (iii) replacing the per-proposal likelihood with precomputed sufficient
statistics, (iv) adding selectable likelihoods for substitution data and for
recurrent mutation, and (v) adding an input path that calls substitutions
directly from a Cactus HAL whole-genome alignment. The original implementation
was retained unmodified and used as a validation oracle throughout.

## 2. Model

### 2.1 Context tree

Contexts are strand-folded so that the focal base is always A or C; contexts
centred on G or T are replaced by their reverse complements, with alternate
alleles complemented accordingly. Layer l contains contexts of length k, and the
node index space is a contiguous B-ary heap in which the children of node i
occupy positions [B*i, B*i + B) of the next layer. Two growth schedules are
supported:

```
alternating growth (legacy):   k = l + 1,      B = 4
                               one flanking base added per layer,
                               alternating sides

centred growth:                k = 2*l + 1,    B = 16
                               one flanking base added on each side per layer
```

Both schedules terminate on the same set of deepest-layer contexts. A 9-mer
model contains 2 * 4^8 = 131,072 folded contexts under either schedule, reached
in nine alternating layers or five centred layers. The motivation for centred
growth is given in Section 5.2.

For each node v and each of the three alternate alleles j, the model carries a
mutation probability p[v,j]. Slot 0 of every count vector holds the unmutated
(reference or ancestral) count and slots 1 to 3 hold the alternate alleles in a
fixed order, so that an A-focal context carries (A, A>C, A>G, A>T) and a C-focal
context carries (C, C>A, C>G, C>T).

### 2.2 Likelihood on sufficient statistics

All leaves descending from node v share that node's rate vector, so the product
of per-leaf multinomials collapses onto four pooled totals:

```
S[v,a] = sum over leaves u descending from v of n[u,a]

log L_v = S[v,0] * log(1 - sum_j p[v,j]) + sum_{j=1..3} S[v,j] * log(p[v,j])
```

S does not depend on any parameter and is therefore a constant of the data. The
original implementation nonetheless recomputed it inside the innermost
Metropolis loop, evaluating four strided sums over the node's full leaf-count
vector on every proposal. In `baymer2`, S is computed once per fit by successive
B-way reduction up the heap:

```
S_(L-1) = per-leaf counts
S_l     = reshape(S_(l+1), (-1, B, 4)).sum(axis = 1)
```

This is O(B^(L-1)) in total for all layers, replacing an O(L * B^(L-1))
re-enumeration per layer, and reduces each likelihood evaluation to O(1) in the
context depth. The reference term is evaluated as log1p(-sum_j p[v,j]) rather
than log(1 - sum_j p[v,j]); at mutation probabilities of 1e-3 to 1e-8 the latter
discards the low-order bits of the argument before the logarithm is taken.

Sufficient statistics were validated against two independent reconstructions
from the original codebase: direct leaf enumeration with strided summation as
performed by the original sampler, and the separately implemented down-tree
marginalisation that generated the shipped lower-k count dictionaries. Agreement
was exact in both cases.

### 2.3 Reparameterisation

The original model wrote

```
p[v,j] = p[pa(v),j] * exp(phi[v,j])       subject to   sum_j p[v,j] < 1
```

and enforced the constraint by redrawing an offending proposal up to five times
before abandoning the update. The effective proposal is then a truncated normal
whose normalising constant depends on the current state and does not appear in
the acceptance ratio, so the chain does not target the stated posterior. The
same pattern in the parent-resampling step silently retained the previous parent
whenever a redrawn parent fell outside the simplex, biasing the plug-in prior
toward small rates.

We reparameterise to additive log-ratio (ALR) coordinates:

```
eta[v,j] = log( p[v,j] / p[v,ref] )

p[v,j]   = exp(eta[v,j]) / (1 + sum_k exp(eta[v,k]))
p[v,ref] =            1  / (1 + sum_k exp(eta[v,k]))

phi[v,j] = eta[v,j] - eta[pa(v),j]
```

Every real-valued eta maps to an interior point of the simplex, so the rejection
region is removed structurally rather than patched. Because mutation
probabilities are small, p[v,ref] lies within about 1e-3 of unity and phi
retains its original interpretation as the log fold change in rate contributed
by one additional flanking base.

At the root there is no parent. The original model placed a flat prior on p over
the admissible region; this is preserved by adding the log-Jacobian of the ALR
map to the root target,

```
log |d p / d eta| = log(p_ref) + sum_j log(p[j])
```

so the root target remains flat in p rather than becoming flat in eta.

The original multiplicative parameterisation is retained as an option for
comparability of phi values with published results. It is implemented with a
single-shot proposal that is rejected if it leaves the simplex, which is
ordinary Metropolis; it does not reproduce the original sampler's behaviour. On
the tutorial data the two parameterisations give rates agreeing to a median of
0.4 to 1.2 per cent with Pearson correlation at or above 0.997.

### 2.4 Priors and Gibbs steps

The spike-and-slab prior on the log fold changes is unchanged:

```
phi[v,j] | z[v,j], sigma_l  ~  N(0, sigma_l^2)          if z[v,j] = 1
                               N(0, (c * sigma_l)^2)    if z[v,j] = 0

z[v,j] | alpha_l            ~  Bernoulli(alpha_l)
alpha_l                     ~  Beta(1, 1)
```

with c the spike-to-slab scale ratio. The indicator full conditional is drawn in
logit form,

```
logit = log(alpha_l / (1 - alpha_l)) + log(c)
        + (phi^2 / 2) * ( 1/(c*sigma_l)^2 - 1/sigma_l^2 )
```

so that the log(c) term favours the spike at phi = 0 while the quadratic term
favours the slab as |phi| grows.

The inclusion probability has the exact conjugate full conditional

```
alpha_l | z  ~  Beta(1 + sum z, 1 + n_active - sum z)
```

where n_active counts only contexts with non-zero opportunity. The original
implementation drew from exactly this distribution and then passed the draw
through a Metropolis filter based on the Bernoulli likelihood ratio alone. For
an independence sampler with proposal proportional to the likelihood the correct
acceptance probability is identically one; the filter therefore rejected any
downhill move, with a reported acceptance rate of 0.79, and concentrated the
posterior for alpha at its mode. The filter was removed.

## 3. Sampler corrections

Nine defects were identified in the original sampler and are listed here because
several alter posterior inference rather than only performance. Each is covered
by a regression test.

| # | Defect | Consequence |
|---|---|---|
| a | `sample_new_sigmas` invoked with fourteen arguments against a thirteen-argument definition | `TypeError` whenever sigma is estimated rather than fixed; latent because all shipped configurations fix sigma |
| b | Out-of-bounds write into a length-three acceptance array at the root layer, under `@njit` with bounds checking disabled | silent memory corruption |
| c | Exact Beta-Gibbs draw for alpha passed through a Metropolis filter | approximately 21 per cent of valid draws rejected; posterior for alpha concentrated at its mode |
| d | Truncated proposals for phi; invalid parent draws silently retained the previous parent | detailed balance violated; plug-in prior biased toward small rates |
| e | Zero-context flag reset per parent rather than per child | siblings of an unobserved context mis-initialised without being registered as unobserved |
| f | Acceptance counter indexed `[i + j]` rather than `[3*i + j]`, with a concurrent write under `prange` | reported acceptance rates incorrect, which is the quantity used to tune proposals |
| g | Adaptive proposal mixing weight overwritten by a constant | adaptive component inert |
| h | Global seeding of a thread-local generator inside parallel regions | results depend on thread count despite a fixed seed |
| i | Falsy test on the sigma-fixing option | a configuration value of zero read as "do not fix", which then triggers defect (a) |

Defect (h) was addressed by replacing stateful generators with a counter-based
scheme. Every random draw is a pure function of an explicit coordinate --
seed, layer, iteration, node, allele, and a purpose tag -- hashed with
splitmix64. Draws are therefore identical regardless of thread count, scheduling
order, or loop order, and any individual draw can be reproduced in isolation.
Identity across one, two, four and eight threads is asserted by test.

## 4. Computational implementation

### 4.1 Membership testing

The original sampler stored the indices of unobserved contexts in an array and
tested membership with `if i in zero_contexts`, which numba lowers to a linear
scan. In the alpha-probability update this test is nested inside the per-allele
loop, giving a cost of O(n_nodes * 3 * n_zero) per call, and the update runs
whenever the alpha step accepts. Replacing the index list with a boolean mask
reduces the test to O(1). Measured on the deepest layer of a 9-mer model
(131,072 nodes), for a single call:

| Unobserved contexts | Linear scan | Boolean mask | Ratio |
|---|---|---|---|
| 0.1 % | 16.0 ms | 0.267 ms | 60 |
| 1 % | 163.5 ms | 0.259 ms | 631 |
| 5 % | 764.6 ms | 0.294 ms | 2600 |
| 10 % | 1509.3 ms | 0.350 ms | 4306 |

### 4.2 Trace storage

Posterior draws were previously held in memory for the duration of a layer. At
`max_mer = 9` the deepest layer requires approximately 3.1 GB for each of four
per-node quantities. In `baymer2`, traces are allocated as memory-mapped `.npy`
files and are written back and released with `madvise(MADV_DONTNEED)` every 50
thinned samples. Memory mapping alone is insufficient: written pages remain
resident until the kernel has cause to reclaim them, and on an unloaded node it
does not. For a complete 8-mer fit with 2000 retained samples, peak resident set
size was 3.45 GB when held in memory and 0.49 GB with mapping and release, with
3.41 GB of traces written to disk and approximately 20 per cent additional
runtime.

### 4.3 Aggregate performance

Both implementations were run on identical inputs with identical iteration
counts on a single node with eight threads:

| Case | Original | baymer2 | Ratio | Original RSS | baymer2 RSS |
|---|---|---|---|---|---|
| k = 5, 8000 iterations, empirical data | 31.09 s | 8.04 s | 3.9 | 253 MB | 182 MB |
| k = 9, 400 iterations, 10 % unobserved | 841.76 s | 30.25 s | 27.8 | 489 MB | 371 MB |

The ratio increases with depth because the cost of the membership test scales
with the number of unobserved contexts, which is substantial for deep k-mers on
any realistic callable region. The sufficient-statistic likelihood contributes a
smaller factor in isolation (1.1, 2.1 and 2.8 at k = 5, 7 and 9 respectively),
since the original per-sweep cost is independent of layer while the
sufficient-statistic cost grows with layer, the two converging at the deepest
layer where each node has a single descendant.

## 5. Likelihood modes

The likelihood is selectable. All modes consume the same sufficient statistics.

### 5.1 Binomial

One multinomial trial per site, as in the original model. Appropriate for
polymorphism data when the per-site mutation probability is small enough that
recurrence is negligible.

### 5.2 Continuous-time Markov chain (substitution data)

Observed differences between a focal genome and its ancestor understate true
mutation because a site may be hit more than once, may revert, and saturates.
The focal branch is modelled as a continuous-time Markov chain,

```
log L = sum over (a,b) of  N[c,a,b] * log( [ expm(Q_c * t) ][a,b] )
```

where a is the ancestral base, b the focal base, N[c,a,b] the count of such
sites in ancestral context c, Q_c the context-conditional 4x4 instantaneous rate
matrix, and t the branch length. Only the product Q*t is identified from a
single branch, so t is fixed at unity by default and the rates are interpreted
as expected substitutions per site along the focal branch.

A full Q_c requires the rates out of all four bases at fixed flanks, whereas the
model parameterises three rates per folded context. Folding supplies the missing
rows without introducing parameters:

```
Q_c[ G -> X ]  =  Q_revcomp(c)[ C -> comp(X) ]
Q_c[ T -> X ]  =  Q_revcomp(c)[ A -> comp(X) ]
```

This requires that reverse complementation map a context back into the same
layer, which holds if and only if the focal base is centred, that is if
k - 1 - i = i where i is the focal index. Under alternating growth this holds on
odd layers only; at k = 4, 128 of 512 (context, base) pairs are unreachable.
Substitution mode therefore uses centred growth, under which the condition holds
at every layer.

A consequence is that the likelihood is not node-local. The four contexts
sharing a set of flanks form a closed group, and a proposal at one member alters
the likelihood of all four. Sampling is therefore parallelised over groups
rather than nodes; groups partition the layer exactly, with groups of size two
arising where the flanks are palindromic.

The flanking context is treated as fixed over the branch. A fully
context-coupled process is not independent across sites and is intractable at
this scale.

Matrix exponentials are computed by scaling and squaring with a Taylor series,
scaled so that the infinity norm of the scaled matrix does not exceed 0.5. This
avoids both LAPACK inside compiled kernels and the failure of eigendecomposition
on defective rate matrices. Agreement with `scipy.linalg.expm` was verified to a
relative tolerance of 1e-10 across seven orders of magnitude in branch length.

As t approaches zero, expm(Q*t) approaches I + Q*t and the mode converges to the
binomial likelihood; this limit is asserted numerically. Data were simulated
under known Q and t and fitted under both likelihoods:

| Branch length (substitutions/site) | Binomial bias | CTMC bias | Binomial error | CTMC error |
|---|---|---|---|---|
| 0.006 | -0.8 % | -0.5 % | 2.9 % | 3.1 % |
| 0.061 | -3.7 % | +0.2 % | 3.7 % | 0.7 % |
| 0.182 | -11.2 % | +0.1 % | 11.2 % | 0.5 % |
| 0.340 | -19.6 % | -0.1 % | 19.6 % | 0.4 % |

The correction is not uniform across contexts. Because faster contexts saturate
sooner, the bias is largest where the rate is highest, so the binomial
likelihood compresses the contrast between fast and slow contexts. In simulation
with CpG contexts assigned a tenfold rate, the binomial shortfall at CpG
exceeded that at other contexts and the estimated CpG elevation fell short of
the true value, while the CTMC recovered it to within 6 per cent.

### 5.3 Recurrent mutation

The three alternate alleles are treated as competing Poisson processes with
expected event counts lambda[j] over the sampled history:

```
L = sum_j lambda[j]

P(site unmutated)      = exp(-L)
P(site carries allele j) = ( lambda[j] / L ) * ( 1 - exp(-L) )
```

The probability of observing any mutation is therefore 1 - exp(-L) rather than
L. Parameters retain their shape but are expected event counts rather than
probabilities, which removes the constraint that they sum to less than one. As L
approaches zero the mode converges to the binomial likelihood; the difference is
second order in L and is material only where recurrence is plausible. At L of
approximately one, a binomial fit understates the rate by more than 15 per cent.
The term log(1 - exp(-L)) is evaluated as log(-expm1(-L)) to retain accuracy at
small L. This mode is node-local and requires no change to the sampler.

## 6. Layer linkage

Two linkage schemes are provided.

Under **cut** linkage, which reproduces the original structure, each layer is
run to completion and the next layer treats the stored draws as an empirical
prior, resampling a parent state independently per parent node at every
iteration. Each layer carries its own multinomial likelihood on pooled counts,
so the same observations enter the target once per layer, and information flows
from shallow layers to deep ones but not in reverse. Memory is bounded to two
layers.

Under **joint** linkage the data enter once, at the leaves:

```
log target = sum over leaves u of log Multinomial( n[u] | p[u] )
             + sum over (v,j) of log P( phi[v,j] | z[v,j], sigma_l )
             + sum over (v,j) of log P( z[v,j] | alpha_l )

with  eta[u] = eta[root] + sum of phi along the path from root to u
```

A proposal at layer l therefore alters every leaf beneath the corresponding
node, and the acceptance ratio is computed over that subtree. The heap layout
makes this tractable: node v at layer l owns the contiguous leaf range
[v*span, (v+1)*span) with span = B^(L-1-l), and nodes within a layer own
disjoint ranges, so a layer can still be swept in parallel. Cost is
O(B^(L-1)) per layer sweep regardless of layer, and all layers remain resident;
joint linkage is practical to approximately k = 7. At k = 5 it is faster than
cut linkage (9.8 s against 16.0 s) because pooled likelihoods are never
recomputed. Joint linkage is currently implemented for the binomial likelihood
only, the CTMC mode being group-local.

## 7. Input from polymorphism data

The FASTA and VCF counting stages were reimplemented. On the reference tutorial
data they reproduce the original totals exactly (799,381 context occurrences and
12,000 retained variants).

Positions and variants are labelled by parity, position mod 2, so that the data
can be split into two interleaved, size-matched halves for cross-validation. In
the original implementation the two halves are mislabelled relative to one
another: the context counter writes parity-zero positions to a column named for
parity one, while the count-dictionary builder selects parity-zero variants for
the numerator but reads the parity-one column for the denominator. Every ODD and
EVEN analysis therefore combines a numerator from one parity class with a
denominator from the other. Combined analyses are unaffected.

Reproducing both wirings on the tutorial data, the aggregate rate changes by
0.03 per cent, because the two parity classes carry almost identical total
opportunity. Per-context rates change by a median of 3.3 per cent, a 95th
percentile of 15 per cent and a maximum of 39 per cent, individual contexts
being unevenly distributed between parities. In `baymer2` parity is a single
concept applied identically to numerators and denominators, and the count-table
constructor raises if the two are inconsistent.

Two further changes were made. Gzip-compressed FASTA is accepted by both
counters, whereas the original accepted it in one and not the other.
Multiallelic sites, which the original discards unconditionally, may be retained
and are required by the recurrent-mutation mode.

## 8. Input from whole-genome alignment

### 8.1 Comparator selection

Comparators are derived from the alignment's own phylogeny, read from
`halStats --tree`. Walking rootward from the focal genome, each successive
ancestor contributes the least-diverged member of its sibling clade, giving the
nested ladder

```
( ( ( ( focal, sister ), outgroup_1 ), outgroup_2 ), ... )
```

Two guards exclude degenerate comparators. A minimum-divergence threshold
excludes comparators below a specified patristic distance. A species map
excludes comparators of the same species as the focal, with labels reduced to
genus and species so that subspecies collapse onto one another.

The species guard cannot be replaced by a distance threshold. In the 577-genome
Vertebrate Genomes Project alignment used here, two human assemblies lie 0.00054
substitutions per site apart and must not be paired, whereas two distinct gull
species lie 0.00006 apart and are a valid comparison; the pair that must be
excluded is the more divergent of the two. Compared against a manually curated
table of 577 trios, automated selection agreed on 569 of 577 sisters (98.6 per
cent) using topology alone and 576 of 577 (99.83 per cent) with species labels,
the single residual being an exact distance tie between two equally close
congeners.

### 8.2 Ancestral state inference

Three methods are provided.

**Parsimony.** Fitch parsimony over the focal genome and its nested outgroups,
by down-pass and up-pass on state bitmasks, taking the state at the most recent
common ancestor of focal and sister. For the three-taxon case this reproduces
the rules used by the existing trio caller on all 64 configurations of focal,
sister and outgroup states.

**Maximum likelihood.** Felsenstein pruning over the selected subtree, using the
branch lengths carried by the alignment tree, giving a posterior distribution
over the ancestral base rather than a point call. Jukes-Cantor is the default
substitution model; an arbitrary rate matrix may be supplied. Sites contribute
fractionally, weighted by the posterior, so that ambiguous sites are retained
rather than discarded. Flanking positions are point-called above a confidence
threshold, since a k-mer has 4^(k-1) flank combinations.

Maximum likelihood agrees with parsimony at every configuration where parsimony
is confident, and differs in two respects: it weights comparators by branch
length, so that a difference on a long branch reduces confidence less than one
on a short branch; and it retains sites that parsimony discards.

The second difference is material because parsimony's discard rule is not
neutral with respect to context. A context is counted only when every base
within it resolves unambiguously, and ambiguity is produced by mutation, so
parsimony preferentially removes the contexts that mutate fastest. On a test
alignment, parsimony recovered 1,346 CpG opportunities (0.70 per cent of counted
sites) against 3,188 (1.4 per cent) under maximum likelihood, the latter
consistent with genomic CpG frequency; the corresponding CpG numerators differed
by a factor of 4.98 against 2.37 for the denominators. Estimated CpG elevation
was 1.75-fold under parsimony and approximately 3-fold under maximum likelihood,
stable across the full range of the flank-confidence threshold.

**Stored ancestor.** Progressive Cactus reconstructs an ancestral genome at each
internal node and stores it in the HAL. Where present, the ancestor of the focal
branch is read directly. In the 577-genome alignment, 577 extant genomes and 576
reconstructed ancestors are present, one per internal node, so this method is
available for every focal genome. Because it requires only the focal genome and
its ancestor in each alignment block rather than a complete trio, it recovers
more aligned sites; on a test alignment it recovered 515,833 counted sites
against 191,491 for parsimony, with spectra correlating at 0.93 and CpG
elevation agreeing to within 2 per cent. Absolute divergence differed
substantially between the two (0.057 against 0.090) and the source of this
discrepancy has not been established; the method should be validated against a
lineage of known divergence before being used for absolute rates.

### 8.3 Ancestral context assignment

Context k-mers are constructed from the reconstructed ancestral sequence rather
than from the focal genome. The two agree at invariant sites and differ at
exactly the sites carrying a substitution, and the resulting error is systematic
and largest at hypermutable contexts: a CpG that has mutated to TpG is otherwise
counted as a TpG opportunity when the mutation occurred in a CpG context. A site
is counted only when the entire ancestral k-mer resolves, so an unresolved
flanking position removes the site rather than contaminating another context.

This was quantified against species already processed by the existing
focal-context caller, whose outputs record both the derived and the ancestral
3-mer, making the correction exact:

```
corrected_opportunity[c] = focal_opportunity[c]
                         - ( substitutions whose derived   3-mer is c )
                         + ( substitutions whose ancestral 3-mer is c )
```

Across eight species, the aggregate rate is unchanged (ratio 1.0000 in every
case), since reassignment conserves the total. Per-context rates change by a
median of 14.6 per cent, with a 95th percentile near 40 per cent and a maximum
of 55 per cent. Estimated CpG elevation changes by a median of 16.8 per cent,
though not uniformly in direction: it increased in five species and decreased in
three. These figures are a lower bound, since the archived outputs record only
the centre base of the ancestral k-mer; `baymer2` reconstructs the full
ancestral k-mer including flanks.

## 9. Validation

### 9.1 Agreement with the original implementation

The counting stages are deterministic and reproduce the original outputs
exactly. The sampler cannot be compared bit-for-bit, since the reparameterisation
alters the proposal geometry and the corrected alpha step alters the chain. We
therefore asked whether the two implementations differ by more than one
implementation differs from itself under a different seed. On the tutorial data
(5-mer, 40,000 iterations, 20,000 burn-in), the median relative difference
between implementations divided by the median relative difference between seeds
was 1.03, 1.22, 1.91, 1.44 and 1.00 for layers 0 to 4 respectively, indicating
agreement within Monte Carlo error at every layer. The root layer, which carries
a flat prior on p, additionally recovers the empirical maximum-likelihood
estimate.

### 9.2 Calibration

Recovery of point estimates does not establish that credible intervals have the
stated width. We simulated from the model's own generative process, drawing
root coordinates, descending the tree with indicator and log-fold-change draws,
and sampling multinomial counts at the leaves; the model was then fitted with
the generating hyperparameters and the coverage of 95 per cent equal-tailed
credible intervals recorded. Coverage was between 0.966 and 1.000 at every
layer.

Coverage must be assessed against the estimand the model identifies. Because the
likelihood at an internal node is a multinomial on that node's pooled descendant
counts, the parameter it identifies is the arithmetic mean of its descendants'
rates, not the multiplicative cascade value placed at that node by a top-down
simulation. The two differ because the log fold changes are Gaussian on the log
scale, so by Jensen's inequality

```
E[ exp(phi) ] = alpha * exp(sigma^2 / 2) + (1 - alpha) * exp((c*sigma)^2 / 2) > 1
```

and each layer of descent inflates the aggregate above the cascade value.
Predicted and observed inflation agreed to approximately 1 per cent (1.0277
against 1.0279 at one layer of descent, 1.1153 against 1.1215 at four). At
k = 5 with sigma = 0.3 the root parameter therefore exceeds the corresponding
cascade value by approximately 12 per cent. This is a property of the model
formulation, present equally in the original implementation, and bears on the
interpretation of shallow-layer rates.

### 9.3 Test coverage

The original implementation had no automated tests. `baymer2` is covered by 368
tests, including exact reproduction of the original counting outputs, agreement
of every Gibbs step with its analytic full conditional, parameter recovery under
simulation, interval calibration, identity of results across thread counts,
convergence of each likelihood mode to its limiting case, and end-to-end
execution on a whole-genome alignment.

## 10. Implementation and availability

`baymer2` is implemented in Python with numba-compiled kernels and depends only
on numpy, scipy, numba, pyyaml and pandas for its core; optional dependencies
cover the alignment path. Alignment input requires the HAL toolkit, invoked
either from native binaries or through an Apptainer image. A command-line
interface exposes comparator selection, substitution calling from a HAL
alignment, and model fitting.

Source is at https://github.com/framos99/ReBaymer under GPL-3.0-or-later. The
original implementation it reimplements, used unmodified as the validation
oracle throughout §9, is at https://github.com/bvoightlab/Baymer.

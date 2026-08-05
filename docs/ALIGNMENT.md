# Alignment input

Going from a Cactus HAL alignment and a focal species to a count table:
comparator selection, ancestral state inference, and the biases each choice
introduces. See [POLYMORPHISM.md](POLYMORPHISM.md) for the FASTA/VCF route.

```bash
baymer2 hal-counts --hal aln.hal --list-species
baymer2 hal-counts --hal aln.hal --focal SPECIES --max-mer 5 -o counts.json
baymer2 fit --counts counts.json --likelihood ctmc -o traces/
baymer2 summarize --traces traces/ -o rates/
```

Comparators are chosen from the alignment's own tree (`halStats --tree`), walking
rootward and taking the least-diverged representative of each successively more
distant clade — automating what would otherwise be a hand-curated trio table
with one row per species. Ancestral states come from Fitch parsimony generalised
to arbitrarily many nested outgroups; it reproduces the three-taxon rules of the
earlier trio caller on all 64 configurations, so species already processed with
that script stay comparable.

**Contexts are counted on the ancestral sequence, not the focal one.** The
earlier script tallies focal k-mers, which agree at invariant sites but differ
at exactly the sites carrying a substitution: a CpG that mutated to TpG is
charged to `TpG` when the mutation actually occurred in `CpG`. Since CpG is where
context dependence is strongest, the bias lands where it does most damage. A site
is counted only when its *whole* ancestral k-mer resolves, so an ambiguous
neighbour removes the site rather than contaminating it.

### Verified against completed production runs

The focal-versus-ancestral context claim is checked directly against species
already processed by the earlier trio caller, no HAL access needed: its outputs
record both the derived and the ancestral 3-mer, so the correction is exact
(`validation/reconcile_completed_species.py`, which takes `--runs` pointing at
that output). Eight species:

| | median per-context change | p95 | max |
|---|---|---|---|
| rate shift from re-assigning context | **14.6%** | ~40% | 55% |

The **aggregate rate is unchanged in every species** (ratio 1.0000) — moving a
site between contexts conserves the total. The damage is entirely per-context,
which is what the model estimates. CpG elevation moves by a median of +16.8%,
though not uniformly: it rose in 5 of 8 species and fell in 3, so the direction
is species-specific even though the magnitude is consistently large.

Two caveats, both verified rather than assumed. The `hal_coords` and
`3mer_substitutions` routes were cross-checked on a species that has both files
and agree **exactly** (0.00%). And neither file corrects *flanking* bases, only
the centre — so these figures are a lower bound on the full bias, which baymer2's
own counter does correct by reconstructing the whole ancestral k-mer.

### Result on a real alignment

`evolverMammals.hal`, focal `simHuman_chr6` (tree branch length 0.144, apparent
divergence 0.090), 5-mer centred tree, both likelihoods:

| | binomial | CTMC | ratio |
|---|---|---|---|
| all contexts | 0.0980 | 0.1079 | 1.078 |
| CpG contexts | 0.1568 | 0.1862 | **1.192** |
| other C contexts | 0.0940 | 0.1012 | 1.075 |
| **CpG elevation** | **1.67×** | **1.84×** | |

The correction is not uniform — it is largest exactly where the rate is highest,
because those contexts saturate first. So multiple-hit bias does not merely shrink
rates, it *compresses the contrast between fast and slow contexts*: the binomial
puts CpG at 1.67× other C sites, the CTMC at 1.84×. That contrast is usually the
quantity of interest in a spectrum study, which is the argument for using the
substitution likelihood on any branch that is not very short.

### The VGP 577-way alignment

Probed with `halStats` on a compute node (the alignment is 900 GB and has to sit
on node-local scratch):

- **577 extant genomes + 576 reconstructed ancestors = 1153**, exactly the
  leaf/internal split of a rooted binary tree. Every ancestor named in the tree
  is present as a genome, so `--method hal-ancestor` is available for *every*
  focal species.
- Cactus names ancestors by clade (`PrimatesAnc0`, `HominidaeAnc4`,
  `VertebratesAnc8`), not `Anc<N>`, so identify them from the tree's internal
  nodes rather than by name pattern.

`baymer2 select-trios` reproduces a hand-curated trio table for the same 577
species:

| configuration | sister agreement |
|---|---|
| topology only | 569/577 (98.6%) |
| \+ species labels (`--species-table`) | **576/577 (99.83%)** |

The single residual is an exact distance tie between two equally close sparids.

**Why species labels are needed.** Large alignments carry several assemblies of
one species, and a sister that close is useless — nearly every site is invariant
between focal and sister, so the trio degenerates. Distance alone cannot
separate the cases:

| pair | distance | correct action |
|---|---|---|
| GRCh38 / T2T-CHM13 | 0.00054 | skip — both *Homo sapiens* |
| two mouse assemblies | 0.00026 | skip — both *Mus musculus* |
| `Canis_lupus_orion` / `_baileyi` / `_familiaris` | 0.00005 | skip — one species |
| *Larus argentatus* / *L. michahellis* | 0.00006 | **keep** — distinct species |

The gulls are *closer* than the humans, so no distance threshold works. `binomial()`
reduces labels to genus+species (collapsing subspecies, resolving `hg38` and
`mm39` via an alias table) and `--species-table` applies it. Across the 577
species, 20 have a sister under 0.001 substitutions/site and 64 have a sister
more diverged than their own outgroup — both flagged in the output.

### Ancestral inference: `--method parsimony`, `ml`, or `hal-ancestor`

Parsimony is fast and, on short branches, close to optimal. It has two costs.
It ignores branch lengths, so a distant outgroup counts the same as a close
sister; and it is all-or-nothing, so an ambiguous site is discarded outright.

That second cost is not neutral. **A context is counted only when every base in
it resolves unambiguously — and ambiguity is caused by mutation. So parsimony
preferentially deletes exactly the contexts that mutate fastest.** Measured on
`evolverMammals.hal`, focal `simHuman_chr6`:

| | parsimony | ML (soft weights) | ratio |
|---|---|---|---|
| CpG denominator | 1,346 | 3,188 | **2.37×** |
| CpG numerator | 222 | 1,105 | 4.98× |
| other-C denominator | 74,542 | 84,897 | 1.14× |
| other-C numerator | 7,004 | 9,746 | 1.39× |
| **CpG elevation** | **1.75×** | **~3×** | |

Parsimony finds CpG at 0.70% of sites; ML finds 1.4%, which is the frequency a
genome actually has. The elevation estimate is stable at 2.8–3.5× across the
whole `--flank-threshold` range (0.0001 to 0.99), so it is not a threshold
artefact.

This is a *third* bias flattening the spectrum, independent of the other two and
larger than either: focal-versus-ancestral context assignment, binomial
saturation, and now parsimony's ascertainment. All three compress the contrast
between fast and slow contexts, and all three are corrected here.

ML agrees with parsimony on every one of the 64 three-taxon configurations where
parsimony is confident, so the modes do not contradict each other — ML adds
graded confidence and retains what parsimony throws away. Flanks are still
hard-called above `--flank-threshold`, since a k-mer has `4**(k-1)` flank
combinations and carrying them all softly is neither tractable nor worthwhile;
only the centre, where the mutation call lives, is spread across candidate
states.


### `--method hal-ancestor`

Reads the ancestral sequence Cactus already reconstructed instead of re-inferring
it, so a block needs only the focal and its ancestor — a duo, not a trio. On
`evolverMammals.hal`, focal `simHuman_chr6`:

| | sites counted | apparent divergence | CpG elevation |
|---|---|---|---|
| parsimony (trio) | 191,491 | 0.0902 | 1.75× |
| hal-ancestor (duo) | **515,833** | 0.0569 | 1.78× |

2.7× more data, and the spectrum shape agrees (r = 0.93, CpG elevation within
2%). **But the absolute divergence differs substantially (0.057 vs 0.090) and I
have not established which is right.** A plausible mechanism is that Cactus's
reconstruction is drawn toward the reference genome in regions where the focal
*is* the reference, which would understate focal-branch divergence — but that is
a hypothesis, not a measurement. Validate on a species with known divergence
before using this method for absolute rates; for *relative* context effects the
three methods already agree closely.

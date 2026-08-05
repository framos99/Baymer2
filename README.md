# ReBaymer

**Bayesian hierarchical inference of context-dependent mutation spectra, from
polymorphism, substitution, or whole-genome alignment data.**

A reimplementation of [Baymer](https://github.com/bvoightlab/Baymer)
(Adams et al.) with a corrected sampler, selectable likelihoods for recurrent
and multiple-hit mutation, and a HAL alignment input path. The importable
package and CLI are named `baymer2`; "ReBaymer" — recurrent-mutation-aware
Baymer — is the same thing.

Mutation rates depend on sequence context, and the deeper the context the harder
it is to estimate: a 9-mer model has 131,072 contexts, most of them rare. Baymer
handles that by arranging contexts in a tree and letting each node shrink toward
its parent, so a 9-mer with little data borrows strength from its 7-mer. This
implementation keeps that model and changes what surrounds it — sufficient
statistics instead of per-leaf sweeps, likelihoods that account for mutations
you cannot see, and inputs that do not require a curated VCF.

## What is different from the original

- **Nine sampler defects fixed**, each pinned by a test — including an
  exact Beta-Gibbs draw that was being filtered through a Metropolis ratio and
  wrongly rejecting ~21% of valid draws, truncated proposals that broke detailed
  balance, and a thread-count dependence that made a fixed seed non-reproducible.
  Full list in [docs/VALIDATION.md](docs/VALIDATION.md#corrections-to-legacy).
- **Sufficient statistics**, so a rate evaluation is O(1) in k-mer depth rather
  than O(4^k). End to end this is 3.9× faster at k=5 and 27.8× at k=9.
- **Selectable likelihoods.** `binomial` (polymorphism), `ctmc` (substitution
  data, inverts back mutation and saturation), `recurrent` (competing Poisson
  processes, no `sum < 1` ceiling).
- **Alignment input.** A Cactus HAL plus a focal species is a valid starting
  point: comparators are chosen from the alignment's own tree, and ancestral
  states come from parsimony, ML, or Cactus's own reconstruction.
- **Contexts counted on the ancestral sequence**, not the focal one — the two
  differ at exactly the sites carrying a substitution, which is where the bias
  does most damage.
- **Memory-mapped traces** with page release, so a whole 8-mer fit peaks at
  0.49 GB instead of 3.45 GB.

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/framos99/ReBaymer
cd ReBaymer
pip install -e .
```

Core dependencies are numpy, scipy, numba, pyyaml and pandas. The optional
extras cover the input paths:

| extra | pulls in | needed for |
|---|---|---|
| `bio` | biopython, pysam | `vcf-counts` — the FASTA/VCF polymorphism path |
| `phylo` | dendropy | `hal-counts`, `select-trios` — alignment phylogenetics |
| `plot` | matplotlib, seaborn | plotting helpers |
| `dev` | pytest, pytest-xdist | running the test suite |

```bash
pip install -e '.[bio,phylo,dev]'
```

**Alignment input additionally needs the [HAL toolkit](https://github.com/ComparativeGenomicsToolkit/hal)**
(`halStats`, `hal2maf`), which is not a Python package. Either put it on `PATH`
or point at it:

```bash
export BAYMER2_CACTUS_BIN=/path/to/cactus-bin/bin      # native binaries
export BAYMER2_CACTUS_SIF=/path/to/cactus.sif          # or an Apptainer image
```

Nothing else in the package needs these; the polymorphism path and the sampler
run without them.

## Quick start

Counts come from either an alignment or a FASTA/VCF pair; both write the same
count JSON, so `fit` consumes them identically.

```bash
# substitutions, from a Cactus HAL alignment
baymer2 hal-counts --hal aln.hal --focal SPECIES --max-mer 5 --outgroups 1 \
        --method parsimony -o counts.json
baymer2 fit --counts counts.json --likelihood ctmc -o traces.rs14/

# polymorphism, from a feature FASTA plus a VCF
baymer2 vcf-counts --fasta feature.fa --vcf variants.vcf.bgz \
        --max-mer 9 --dataset ALL -o counts.json
baymer2 fit --counts counts.json --likelihood binomial -o traces.rs14/

# every layer, per seed and pooled across seeds, from traces already on disk
baymer2 summarize --traces traces.rs14/ traces.rs15/ -o out/

# comparators for every genome in a tree
baymer2 select-trios --hal aln.hal --species-table species.tsv -o trios.tsv
```

### Rate outputs

`fit --rates` writes one file, for the deepest layer only, as
`{context: [p_ref, p_alt1, p_alt2, p_alt3]}`.

`summarize` is the fuller product. It reads finished traces and refits nothing,
so it can be rerun at any time, and it writes one file per layer per seed plus a
pooled `final` — for a centred 9-mer run that is
`rates.{1,3,5,7,9}mer.{rs14,rs15,final}.json`. Pooling combines every chain's
draws and then averages, which is what legacy Baymer did, and is **not** the same
as averaging the per-seed means when the chains differ in length.

Rates are slotted by destination base, `[p_A, p_C, p_G, p_T]`, matching legacy's
`rate_dict` rather than the reference-first order above; the reference slot holds
`1 - sum(alts)`. Alongside them each context carries `rate_sd`, `phi_mean`, and
`pip` — the posterior inclusion probability, i.e. the mean spike-and-slab
indicator.

Two things are `null` rather than zero, deliberately: the reference slot of
`pip` and `phi_mean`, since there is no indicator or phi for a
non-substitution; and *every* slot of `pip` at layer 0, since the root has no
spike-and-slab at all, so `0.0` there would assert "never in the slab", which
the sampler never measured.

Contexts sit under a `"contexts"` key rather than at the top level, so the file
can also carry provenance. Consumers expecting the legacy layout need one line:
`d = json.load(f)["contexts"]`.

**Tree depth is configurable in two senses:**

| flag | meaning | where | default |
|---|---|---|---|
| `--max-mer` | k-mer context depth, i.e. number of model layers | `hal-counts`, `vcf-counts` | 5 |
| `--outgroups` | phylogenetic depth, successive outgroups beyond the sister | `hal-counts`, `select-trios` | 1 |

`fit` takes neither: the geometry is read from the counts file so the two
cannot disagree. It does take `--linkage cut|joint` and
`--parameterization alr|multiplicative` — see [docs/MODEL.md](docs/MODEL.md).

### Python API

```python
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.backends.numba_mh import LayerConfig
from baymer2.model import ModelConfig, fit

tree = ContextTree(max_mer=5, alternation="right")
table = CountTable.from_json("5mer.….folded_count_dict.json", tree)
result = fit(table, ModelConfig(
    max_mer=5, seed=14,
    default_layer=LayerConfig(n_iterations=40000, burnin=20000, thin=10,
                              set_sigma=0.8540486, c=0.04472136),
))
print(result.summary())
rates = result.rate_dict(4)      # {context: [p_ref, p_alt1, p_alt2, p_alt3]}
```

The three seams that matter are `CountTable` (data in), the `Likelihood`
protocol (the model), and the `Backend` protocol (the sampler). Adding a
likelihood never requires touching a backend.

## Documentation

| | |
|---|---|
| [docs/METHODS.md](docs/METHODS.md) | The model, priors, Gibbs steps, and every likelihood written out |
| [docs/MODEL.md](docs/MODEL.md) | Choosing between `cut`/`joint` linkage, `alr`/`multiplicative` parameterisation, and the three likelihoods |
| [docs/ALIGNMENT.md](docs/ALIGNMENT.md) | HAL input: comparator selection, parsimony vs ML vs `hal-ancestor`, and the biases each introduces |
| [docs/POLYMORPHISM.md](docs/POLYMORPHISM.md) | FASTA + VCF input, and the EVEN/ODD parity mismatch it fixes |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Agreement with the original, measured performance, calibration, and the nine corrections |
| [docs/NOTES.md](docs/NOTES.md) | Failure modes that cost real time, and one fully measured production run |

## Validation

The count pipeline is deterministic and reproduces the original **exactly** —
TSVs are byte-identical, count JSONs differ only in key ordering.

The sampler cannot be compared bit-for-bit, because the ALR reparameterisation
changes the proposal geometry and the corrected alpha step changes the chain.
So the test asks whether the two implementations differ by more than one
implementation differs from *itself* under another seed:

| layer | seed-to-seed | vs legacy | ratio |
|---|---|---|---|
| 0 | 0.0005 | 0.0005 | 1.03 |
| 1 | 0.0016 | 0.0020 | 1.22 |
| 2 | 0.0032 | 0.0061 | 1.91 |
| 3 | 0.0069 | 0.0099 | 1.44 |
| 4 | 0.0122 | 0.0122 | 1.00 |

Every layer sits at or near 1.0 — agreement within Monte Carlo error. The
sampler is also calibrated: 95% intervals contain the truth ~95% of the time
against the estimand the model actually fits. Details, and the caveat hiding in
that last clause, in [docs/VALIDATION.md](docs/VALIDATION.md).

## Testing

```bash
pip install -e '.[dev]'
pytest -q
```

411 tests. 390 run from a bare clone; the remaining 21 need external data and
skip cleanly with a message naming the variable to set:

| variable | unlocks |
|---|---|
| `BAYMER2_LEGACY_ROOT` | tests marked `legacy` — a [Baymer](https://github.com/bvoightlab/Baymer) checkout whose tutorial has been run, used as the oracle |
| `BAYMER2_EVOLVER_HAL` | tests marked `hal` — `evolverMammals.hal`, shipped with Cactus |
| `BAYMER2_CACTUS_BIN` | tests marked `hal` — directory holding `halStats`/`hal2maf` |

`benchmarks/` and `validation/` hold the scripts behind the performance,
calibration, and legacy-agreement tables; `validation/check_calibration.py`
needs nothing external.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE). The original Baymer is GPL-3.0 and
this is a derivative work of it.

## Citation

See [CITATION.cff](CITATION.cff). If you use this, please also cite the original
Baymer (Adams et al., https://github.com/bvoightlab/Baymer), whose model this
implements and whose tutorial data underlies the golden fixtures in `tests/`.

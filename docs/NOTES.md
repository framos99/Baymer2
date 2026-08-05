# Development notes

Failure modes that cost real time, and one fully measured production run.
Nothing here is required to use the package; it is recorded because each
item was expensive to learn and is invisible from the code.

## Gotchas found the hard way

**MAF `src` fields cannot be split on a dot.** The field is nominally
`genome.sequence`, but genome names may contain dots of their own — every VGP
assembly accession does. A real s-line reads `GCA_048771995.1.CM088821.1`, and
splitting on the first dot yields `GCA_048771995`, which matches no requested
genome. Resolution must be against the *known* set of genome names, longest
first so a name that prefixes another cannot capture it
(`MafRecord.genome_in`).

This produced zero usable blocks for every species on the 577-way alignment
while passing every test, because `evolverMammals.hal` names (`simHuman_chr6`)
contain no dots. `tests/test_alignment.py` now uses accession-formatted names
throughout.

**Counting failures must be loud.** The same run wrote an all-zero count table,
fitted 40,000 iterations against it, and emitted rate files in which every
context held the initialisation floor `1e-06` — exit status `Done`, empty
stderr, 21 GB of output. `hal-counts` and `vcf-counts` now exit non-zero and
write nothing when no sites are counted, printing the requested genome names
beside the `src` fields actually present; `model.fit` refuses an empty table so
the API path is guarded too.

`hal-counts` also reports why blocks were dropped:

```
block tally:
  blocks read                       14,315
  kept                               3,617
  dropped: unresolved src name           0
  dropped: focal absent                  0
  dropped: comparator absent         1,031
  dropped: focal below min-block     9,667
  dropped: duplicated genome             0
```

A non-zero `unresolved src name` is the signature of a name-format mismatch
rather than absent coverage — the distinction that took a wasted run to spot.

## Production run: Taeniopygia guttata

First full species on the 577-way alignment, k=9, `--outgroups 2`, CTMC, two
seeds. Everything below is measured, not projected.

| | |
|---|---|
| blocks read / kept | 8,239,599 / 4,335,131 |
| sites counted | 751,491,373 |
| **empty contexts** | **0 of 131,072** |
| apparent divergence | 0.01722 (tree branch length 0.01473) |
| counting | ~10 min, 116 MB peak |
| fitting | 2.1 h per seed, 215 MB peak, 11 GB traces per seed |

The fastest 9-mers are `ACTTCGAAT`, `TATACGCAG`, `CTAGCGCAC`, `GGTACGGTT`,
`TATACGTTA` — every one a CpG, recovered at 9-mer resolution with nothing
directing the model toward them.

### Three findings that should shape a 577-species sweep

**The second seed buys almost nothing here.** The two chains agree at
r = 0.99976, median 0.46% relative difference, p95 5.5%. Dropping to one seed
halves the cost; keep two on a subset for convergence assessment.

**CTMC costs ~100x the compute for a ~3.5% correction on a short branch.**
Fitted rates sit 3.4% above empirical overall and 4.9% above at CpG — the
predicted direction, but small, because saturation is negligible at 0.017
substitutions/site. CpG elevation barely moves (3.67x empirical, 3.69x fitted).
The correction earns its cost on deeply diverged branches, not on shallow ones;
`--likelihood binomial` is defensible for the shallow end of the tree.

**Reserve memory from measurement, not inheritance.** Our batch template
originally requested 200 GB per job, carried over from the old trio caller's
observed 162 GB peak. That script materialised MAF blocks; baymer2 streams them
and memory-maps its traces, so the real peak is 215 MB. On a 375 GB host a
200 GB reservation caps the node at **one** concurrent job when the cores allow
four:

| reservation | concurrent per host | 577 species at 4.2 h |
|---|---|---|
| 200 GB (as inherited) | 1 | 2,423 h — 101 days |
| 8 GB (measured + headroom) | 4 | 606 h — 25 days |

If you are writing a scheduler script, measure a single run first — 8 GB is
ample.


## Gotchas

- **Numba cache staleness.** `cache=True` does not invalidate a caller's on-disk
  cache when an `inline="always"` callee changes. If an edit appears to have no
  effect, `find . -name '__pycache__' -type d -exec rm -rf {} +` before drawing
  conclusions. This silently masked a real RNG fix during development.
- **Unsigned arithmetic.** Under `inline="always"` numba re-specialises the body
  at the caller's types, so a declared `uint64` parameter receiving an `int64`
  loop variable is *not* converted and the expression promotes to `float64`.
  Every function in `rng.py` casts internally as well as declaring a signature.
- **Interpreter.** On a managed cluster the interpreter on `PATH` may not be the
  one you want — a FIPS-mode build can abort on import, and a conda base may be
  missing optional dependencies. Fix the interpreter explicitly rather than
  relying on `python3`.

"""Measure the focal-versus-ancestral context bias on real completed runs.

Plan item 3.6 claims that tallying k-mer contexts on the *focal* genome
mis-assigns opportunity at exactly the sites carrying a substitution,
and that the error is systematic and worst at CpG. The existing trio
caller does tally focal k-mers (``_focal_kmer`` over ``ungapped_focal``),
and its outputs are on disk for species that have already been run --
so the claim can be checked directly rather than argued.

The check needs no HAL access. ``*_3mer_substitutions.tsv`` records both
``derived_3mer`` and ``ancestral_3mer`` per substitution, which makes the
correction exact::

    corrected[c] = focal[c]
                 - (substitutions whose derived_3mer   is c)
                 + (substitutions whose ancestral_3mer is c)

Every substituted site is moved from the context it ended in to the one
it mutated *from*. Invariant sites are unaffected, which is why the
aggregate barely moves while individual contexts shift a lot.

This is an audit of substitution-calling output that already exists on
disk, so unlike the rest of ``validation/`` it cannot run from a fresh
clone. ``--runs`` must point at a directory of completed per-species
output, each holding a context tally and a ``*_3mer_substitutions.tsv``::

    PYTHONPATH=src python validation/reconcile_completed_species.py \\
        --runs /path/to/substitution_calling_output
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from baymer2.contexts import ContextTree, reverse_complement


def completed_species(directory: Path) -> list[str]:
    """Species with a context tally and some per-substitution record.

    ``rc_`` duplicates are dropped: they are re-runs of the same data
    and would double-count in a cross-species summary.
    """
    out = []
    for path in sorted(directory.glob("*_trioalign.kmer_context_counts.tsv")):
        stem = path.name[: -len("_trioalign.kmer_context_counts.tsv")]
        if stem.startswith("rc_"):
            continue
        has_subs = (directory / f"{stem}_3mer_substitutions.tsv").exists()
        has_coords = (
            directory / f"{stem}_trioalign.substitutions.hal_coords.tsv.gz"
        ).exists()
        if has_subs or has_coords:
            out.append(stem)
    return out


def read_context_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    with open(path) as handle:
        next(handle, None)
        for line in handle:
            context, _, value = line.partition("\t")
            if value.strip():
                counts[context] += int(value)
    return counts


def read_hal_coords_contexts(path: Path) -> tuple[Counter, Counter, int]:
    """Derive both context assignments from a ``hal_coords`` file.

    That file records the *focal* k-mer plus the inferred ancestral
    allele, so the ancestral k-mer is the focal one with its centre
    replaced -- flanks are shared because only the focal base changed.
    Only ``derived_on_focal`` rows describe the focal branch; the others
    are substitutions on the sister or outgroup lineage.
    """
    import gzip

    derived: Counter = Counter()
    ancestral: Counter = Counter()
    total = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        header = next(handle).rstrip("\n").split("\t")
        c_i = header.index("context")
        a_i = header.index("ancestral")
        k_i = header.index("category")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(c_i, a_i, k_i):
                continue
            if parts[k_i] != "derived_on_focal":
                continue
            context = parts[c_i]
            anc = parts[a_i]
            if len(context) != 3 or anc not in "ACGT":
                continue
            derived[context] += 1
            ancestral[context[0] + anc + context[2]] += 1
            total += 1
    return derived, ancestral, total


def read_substitution_contexts(path: Path) -> tuple[Counter, Counter, int]:
    """Tally derived and ancestral 3-mers over every substitution."""
    derived: Counter = Counter()
    ancestral: Counter = Counter()
    total = 0
    with open(path) as handle:
        header = next(handle).rstrip("\n").split("\t")
        d_i = header.index("derived_3mer")
        a_i = header.index("ancestral_3mer")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(d_i, a_i):
                continue
            derived[parts[d_i]] += 1
            ancestral[parts[a_i]] += 1
            total += 1
    return derived, ancestral, total


def fold(counter: Counter, tree: ContextTree) -> dict[str, float]:
    """Collapse onto the folded context set, dropping anything with N."""
    valid = set(tree.contexts(tree.n_layers - 1))
    out: dict[str, float] = {c: 0.0 for c in valid}
    for context, value in counter.items():
        if len(context) != 3 or set(context) - set("ACGT"):
            continue
        folded = context if context[1] in "AC" else reverse_complement(context)
        if folded in out:
            out[folded] += value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory of completed per-species substitution "
                         "calling output")
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N species")
    ap.add_argument("--quiet", action="store_true",
                    help="one summary line per species")
    args = ap.parse_args()

    tree = ContextTree(max_mer=3, growth="centred")
    contexts = list(tree.contexts(tree.n_layers - 1))
    names = args.species or completed_species(args.runs)
    if args.limit:
        names = names[: args.limit]
    summary = []
    if not names:
        raise SystemExit(f"no completed species with both files under {args.runs}")

    print(f"reconciling {len(names)} completed species\n")

    for name in names:
        ctx_path = args.runs / f"{name}_trioalign.kmer_context_counts.tsv"
        subs_path = args.runs / f"{name}_3mer_substitutions.tsv"
        coords_path = (
            args.runs / f"{name}_trioalign.substitutions.hal_coords.tsv.gz"
        )

        focal_raw = read_context_counts(ctx_path)
        if coords_path.exists():
            derived_raw, ancestral_raw, n_subs = read_hal_coords_contexts(coords_path)
            source = "hal_coords"
        else:
            derived_raw, ancestral_raw, n_subs = read_substitution_contexts(subs_path)
            source = "3mer_subs"
        if n_subs == 0:
            print(f"--- {name}: no usable substitutions, skipped\n")
            continue

        focal = fold(focal_raw, tree)
        derived = fold(derived_raw, tree)
        ancestral = fold(ancestral_raw, tree)

        focal_den = np.array([focal[c] for c in contexts])
        derived_num = np.array([derived[c] for c in contexts])
        ancestral_num = np.array([ancestral[c] for c in contexts])
        corrected_den = focal_den - derived_num + ancestral_num

        usable = (focal_den > 0) & (corrected_den > 0)
        # As-published: numerator and denominator both in focal context.
        published = np.divide(derived_num, focal_den,
                              out=np.zeros_like(focal_den), where=usable)
        # Corrected: both in ancestral context, which is where the
        # mutation actually occurred.
        corrected = np.divide(ancestral_num, corrected_den,
                              out=np.zeros_like(focal_den), where=usable)

        ratio = np.divide(corrected, published,
                          out=np.ones_like(published), where=published > 0)
        cpg = np.array([c[1:] == "CG" for c in contexts])
        other_c = np.array([c[1] == "C" and c[1:] != "CG" for c in contexts])

        print(f"--- {name}  [{source}]")
        print(f"    substitutions {n_subs:,}   aligned sites "
              f"{focal_den.sum():,.0f}   contexts {int(usable.sum())}")
        print(f"    aggregate rate  published {derived_num.sum()/focal_den.sum():.6f}"
              f"   corrected {ancestral_num.sum()/corrected_den.sum():.6f}"
              f"   ratio {(ancestral_num.sum()/corrected_den.sum())/(derived_num.sum()/focal_den.sum()):.4f}")
        r = ratio[usable]
        print(f"    per-context |change|  median {np.median(np.abs(r-1)):.2%}"
              f"   p95 {np.percentile(np.abs(r-1),95):.2%}"
              f"   max {np.abs(r-1).max():.2%}")
        if cpg.any() and other_c.any():
            print(f"    CpG elevation   published "
                  f"{published[cpg].mean()/published[other_c].mean():.3f}x"
                  f"   corrected "
                  f"{corrected[cpg].mean()/corrected[other_c].mean():.3f}x")
            print(f"    CpG denominator moves by "
                  f"{corrected_den[cpg].sum()/focal_den[cpg].sum()-1:+.2%}"
                  f"   other-C by "
                  f"{corrected_den[other_c].sum()/focal_den[other_c].sum()-1:+.2%}")
        cpg_pub = published[cpg].mean() / published[other_c].mean()
        cpg_cor = corrected[cpg].mean() / corrected[other_c].mean()
        summary.append((name, float(np.median(np.abs(r - 1))), cpg_pub, cpg_cor))

        if not args.quiet:
            worst = np.argsort(-np.abs(ratio - 1) * usable)[:5]
            print("    largest per-context shifts:")
            for i in worst:
                print(f"      {contexts[i]}  published {published[i]:.5f} -> "
                      f"corrected {corrected[i]:.5f}  ({ratio[i]-1:+.1%})")
            print()

    if summary:
        print("=" * 74)
        print(f"{'species':>44} {'median |dr|':>12} {'CpG pub':>8} {'CpG corr':>9}")
        print("-" * 74)
        for name, median, pub, cor in summary:
            print(f"{name[:44]:>44} {median:>11.1%} {pub:>8.2f}x {cor:>8.2f}x")
        medians = np.array([s[1] for s in summary])
        pubs = np.array([s[2] for s in summary])
        cors = np.array([s[3] for s in summary])
        print("-" * 74)
        print(f"{'ACROSS SPECIES (n=%d)' % len(summary):>44} "
              f"{np.median(medians):>11.1%} {np.median(pubs):>8.2f}x "
              f"{np.median(cors):>8.2f}x")
        print(f"\nCpG elevation is understated by a median of "
              f"{np.median(cors/pubs)-1:+.1%} when contexts are tallied on the "
              f"focal genome.")


if __name__ == "__main__":
    main()

"""Command line interface.

Counts come from either an alignment or a FASTA/VCF pair, and both write
the same count JSON, so ``fit`` consumes them identically::

    baymer2 hal-counts --hal aln.hal --focal SPECIES --max-mer 5 -o counts.json
    baymer2 vcf-counts --fasta feature.fa --vcf vars.vcf.bgz -o counts.json
    baymer2 fit --counts counts.json --likelihood ctmc -o out/

``select-trios`` chooses comparators for every genome in a tree, and
``hal-counts --list-species`` inspects an alignment before committing to
a run -- which matters when the alignment is a 577-genome file on
node-local scratch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from baymer2.ancestral.felsenstein import MLAlignmentCounter
from baymer2.ancestral.hal_ancestor import AncestorAlignmentCounter
from baymer2.ancestral.parsimony import AlignmentCounter
from baymer2.ancestral.tree import binomial
from baymer2.backends.numba_mh import LayerConfig
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.io.hal import (
    BlockTally,
    HalRuntime,
    ancestral_genomes,
    extract_maf,
    hal_tree,
    iter_maf_blocks,
    sample_src_names,
)
from baymer2.io.vcf import (
    DATASETS,
    VariantFilter,
    build_count_table,
    count_contexts,
    count_mutations,
)
from baymer2.model import ModelConfig, fit


def _write_count_json(table: CountTable, path: Path, extra: dict) -> None:
    """Serialise a count table plus provenance."""
    tree = table.tree
    layer = tree.n_layers - 1
    focal_idx = tree.focal_index(tree.max_mer)
    payload = {
        "max_mer": tree.max_mer,
        "growth": tree.growth,
        "alternation": tree.alternation,
        "focal_index": focal_idx,
        "name": table.name,
        "counts": {
            ctx: [float(v) for v in table.leaf_counts[i]]
            for i, ctx in enumerate(tree.contexts(layer))
        },
        **extra,
    }
    path.write_text(json.dumps(payload, indent=1))


def _read_count_json(path: Path) -> CountTable:
    payload = json.loads(Path(path).read_text())
    tree = ContextTree(
        max_mer=payload["max_mer"],
        growth=payload.get("growth", "alternating"),
        alternation=payload.get("alternation", "right"),
    )
    contexts = tree.contexts(tree.n_layers - 1)
    counts = np.zeros((len(contexts), 4))
    for i, ctx in enumerate(contexts):
        counts[i] = payload["counts"][ctx]
    return CountTable(tree=tree, leaf_counts=counts, name=payload.get("name", ""))


def cmd_hal_counts(args: argparse.Namespace) -> int:
    runtime = HalRuntime.detect()
    if not runtime.available:
        print("error: no cactus binaries or .sif image found", file=sys.stderr)
        return 2

    tree = hal_tree(args.hal, runtime)

    if args.list_species:
        ancestors = set(ancestral_genomes(args.hal, runtime))
        print(f"{len(tree.leaf_names)} extant genomes:")
        for name in tree.leaf_names:
            print(f"  {name}")
        print(f"{len(ancestors)} reconstructed ancestors:")
        for name in sorted(ancestors):
            print(f"  {name}")
        return 0

    if args.focal is None:
        print("error: --focal is required unless --list-species", file=sys.stderr)
        return 2

    species = _load_species_map(args.species_table)
    selection = tree.select(
        args.focal,
        n_outgroups=args.outgroups,
        min_divergence=args.min_sister_divergence,
        species_of=species,
    )
    print(f"focal:     {selection.focal}")
    print(f"sister:    {selection.sister}")
    print(f"outgroups: {', '.join(selection.outgroups)}")
    print(f"ancestor:  {selection.ancestor}")
    print(f"branch:    {selection.focal_branch_length:.6f}")
    if selection.sister_more_diverged_than_outgroup:
        print(
            "note:      the sister is more diverged than an outgroup, which "
            "weakens parsimony polarisation"
        )

    if args.method == "hal-ancestor":
        if not selection.ancestor:
            print(
                "error: the tree does not name the focal's parent, so its "
                "reconstructed ancestor cannot be addressed; use --method "
                "parsimony or ml",
                file=sys.stderr,
            )
            return 2
        available = set(hal_genomes(args.hal, runtime))
        if selection.ancestor not in available:
            print(
                f"error: ancestor {selection.ancestor!r} is named in the tree "
                f"but is not a genome in the HAL; use --method parsimony or ml",
                file=sys.stderr,
            )
            return 2
        targets = [selection.ancestor]
    else:
        targets = selection.targets

    maf = Path(args.maf) if args.maf else Path(args.output).with_suffix(".maf")
    if args.reuse_maf and maf.exists():
        print(f"reusing    {maf}")
    else:
        print(f"extracting {maf} ...")
        extract_maf(
            args.hal, selection.focal, targets, maf,
            runtime=runtime, ref_sequence=args.ref_sequence,
            no_ancestors=(args.method != "hal-ancestor"),
        )

    context_tree = ContextTree(max_mer=args.max_mer, growth="centred")
    if args.method == "parsimony":
        counter = AlignmentCounter(
            context_tree, selection.focal, selection.sister, selection.outgroups
        )
    elif args.method == "ml":
        counter = MLAlignmentCounter.from_selection(
            context_tree, tree, selection,
            flank_threshold=args.flank_threshold,
        )
    else:
        counter = AncestorAlignmentCounter.from_selection(context_tree, selection)
    tally = BlockTally()
    counter.add_blocks(
        iter_maf_blocks(
            maf, selection.focal, targets,
            min_block=args.min_block, tally=tally,
        )
    )

    table = counter.to_count_table()
    print("block tally:")
    for name, value in tally.summary().items():
        print(f"  {name:32s} {value:,}")
    print(f"sites:     {table.total_sites:,.0f}")
    print(f"divergence:{counter.divergence: .5f}  (apparent, focal branch)")
    if hasattr(counter, "category_summary"):
        for name, value in counter.category_summary().items():
            print(f"  {name:22s} {value:,}")
    else:
        print(f"  dropped (low-confidence flank) {counter.n_dropped_flank[0]:,}")

    if table.total_sites <= 0:
        requested = [selection.focal, *targets]
        observed = sample_src_names(maf)
        print(
            "\nERROR: no usable alignment blocks, so no counts were produced.",
            file=sys.stderr,
        )
        if tally.unresolved_src:
            print(
                f"{tally.unresolved_src:,} blocks carried no src field matching "
                f"any requested genome, which points at a name-format "
                f"mismatch rather than absent coverage:",
                file=sys.stderr,
            )
        print("  requested genome names:", file=sys.stderr)
        for name in requested:
            print(f"    {name}", file=sys.stderr)
        print("  src fields present in the MAF:", file=sys.stderr)
        for name in observed:
            print(f"    {name}", file=sys.stderr)
        print(
            "Nothing was written; a table of zeros would fit without error "
            "and produce rates that look valid.",
            file=sys.stderr,
        )
        return 1

    _write_count_json(
        table,
        Path(args.output),
        {
            "focal": selection.focal,
            "sister": selection.sister,
            "outgroups": selection.outgroups,
            "ancestor": selection.ancestor,
            "focal_branch_length": selection.focal_branch_length,
            "method": args.method,
            "categories": (
                counter.category_summary()
                if hasattr(counter, "category_summary")
                else {}
            ),
            "apparent_divergence": counter.divergence,
            "n_blocks": counter.n_blocks,
            "block_tally": tally.summary(),
        },
    )
    print(f"wrote      {args.output}")
    return 0


def _load_species_map(path: str | None) -> dict[str, str] | None:
    """Assembly -> binomial, from a two-column TSV.

    Reduced to genus+species so subspecies collapse onto one another:
    three *Canis lupus* assemblies must not be each other's sister,
    while two distinct *Larus* species must remain usable.
    """
    if not path:
        return None
    import csv

    with open(path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        return {}
    fields = rows[0].keys()
    assembly_key = "assembly" if "assembly" in fields else "focal_assembly"
    species_key = "species" if "species" in fields else "SpeciesName"
    if assembly_key not in fields or species_key not in fields:
        raise SystemExit(
            f"{path}: expected columns (assembly, species) or "
            f"(focal_assembly, SpeciesName), found {sorted(fields)}"
        )
    return {r[assembly_key]: binomial(r[species_key]) for r in rows}


def cmd_select_trios(args: argparse.Namespace) -> int:
    """Emit comparator choices for every genome in an alignment."""
    import csv

    runtime = HalRuntime.detect()
    if args.tree:
        from baymer2.ancestral.tree import Phylogeny

        tree = Phylogeny.from_file(args.tree)
    else:
        if not runtime.available:
            print("error: no cactus binaries or .sif image found", file=sys.stderr)
            return 2
        tree = hal_tree(args.hal, runtime)

    species = _load_species_map(args.species_table)
    selections = tree.select_all(
        n_outgroups=args.outgroups,
        min_divergence=args.min_sister_divergence,
        species_of=species,
    )

    fields = [
        "focal", "sister", "outgroups", "ancestor", "focal_branch_length",
        "sister_distance", "sister_more_diverged_than_outgroup",
    ]
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for selection in selections:
            writer.writerow(selection.as_row())

    flagged = sum(1 for s in selections if s.sister_more_diverged_than_outgroup)
    close = sum(1 for s in selections if s.distances[s.sister] < 0.001)
    print(f"genomes:   {len(tree.leaf_names)}")
    print(f"selected:  {len(selections)}")
    print(f"flagged:   {flagged} with a sister more diverged than an outgroup")
    print(f"very close:{close} with a sister under 0.001 substitutions/site")
    print(f"wrote      {args.output}")
    return 0


def cmd_vcf_counts(args: argparse.Namespace) -> int:
    """Count contexts and mutations from a feature FASTA plus a VCF."""
    tree = ContextTree(
        max_mer=args.max_mer,
        growth=args.growth,
        alternation=args.alternation,
    )
    high_confidence = not args.no_high_confidence

    print(f"fasta:     {args.fasta}")
    print(f"vcf:       {args.vcf}")
    print(f"tree:      max_mer={args.max_mer} growth={args.growth}"
          + (f" alternation={args.alternation}" if args.growth == "alternating" else ""))
    print(f"dataset:   {args.dataset}")

    print("counting contexts ...")
    contexts = count_contexts(args.fasta, tree, high_confidence=high_confidence)

    filters = VariantFilter(
        min_ac=args.min_ac,
        max_ac=args.max_ac,
        min_af=args.min_af,
        max_af=args.max_af,
        min_quality=args.min_quality,
        ac_key=args.ac_key,
        an_key=args.an_key,
        quality_key=args.quality_key,
        keep_multiallelic=args.keep_multiallelic,
    )
    print("counting mutations ...")
    mutations = count_mutations(
        args.vcf, args.fasta, tree,
        filters=filters, high_confidence=high_confidence,
    )
    for name, value in mutations.summary().items():
        print(f"  {name:20s} {value:,}")

    # Selects the parity class from numerator and denominator together;
    # legacy drew them from opposite classes for ODD and EVEN.
    table = build_count_table(
        contexts, mutations.counts, tree,
        dataset=args.dataset, name=args.name or Path(args.vcf).stem,
    )
    print(f"sites:     {table.total_sites:,.0f}")
    print(f"variants:  {table.leaf_counts[:, 1:].sum():,.0f}")
    if table.total_sites <= 0:
        print(
            "\nERROR: no contexts were counted. Check that the FASTA headers "
            "and VCF CHROM values refer to the same sequences, and that the "
            "FASTA is longer than --max-mer.",
            file=sys.stderr,
        )
        return 1
    rate = table.leaf_counts[:, 1:].sum() / table.total_sites
    print(f"rate:      {rate:.6f}")

    _write_count_json(
        table,
        Path(args.output),
        {
            "source": "vcf",
            "fasta": str(args.fasta),
            "vcf": str(args.vcf),
            "dataset": args.dataset,
            "high_confidence": high_confidence,
            "keep_multiallelic": args.keep_multiallelic,
            "filters": {
                "min_ac": args.min_ac, "max_ac": args.max_ac,
                "min_af": args.min_af, "max_af": args.max_af,
                "min_quality": args.min_quality,
            },
            "qc": mutations.summary(),
        },
    )
    print(f"wrote      {args.output}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    table = _read_count_json(Path(args.counts))
    layer_config = LayerConfig(
        n_iterations=args.iterations,
        burnin=args.burnin if args.burnin is not None else args.iterations // 2,
        thin=args.thin,
        set_sigma=args.set_sigma,
        c=args.c,
        phi_proposal_sd=args.proposal_sd,
        seed=args.seed,
        parameterization=args.parameterization,
    )
    config = ModelConfig(
        max_mer=table.tree.max_mer,
        growth=table.tree.growth,
        alternation=table.tree.alternation,
        likelihood=args.likelihood,
        linkage=args.linkage,
        branch_length=args.branch_length,
        seed=args.seed,
        default_layer=layer_config,
    )
    result = fit(table, config, progress=args.progress, trace_dir=args.output)
    print(result.summary())

    if args.rates:
        layer = table.tree.n_layers - 1
        Path(args.rates).write_text(json.dumps(result.rate_dict(layer), indent=1))
        print(f"wrote      {args.rates}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    """Write per-layer rate and PIP JSONs from traces already on disk."""
    from baymer2.summarize import summarize_run

    paths = summarize_run(
        args.traces,
        args.output,
        prefix=args.prefix,
        final_label=args.final_label,
        digits=args.digits,
        progress=args.progress,
    )
    for path in paths:
        print(f"wrote      {path}")
    print(f"{len(paths)} files")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baymer2", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    hal = sub.add_parser(
        "hal-counts", help="call substitutions from a HAL alignment"
    )
    hal.add_argument("--hal", required=True, help="path to the .hal alignment")
    hal.add_argument("--focal", help="focal genome name")
    hal.add_argument(
        "--outgroups", type=int, default=1,
        help="number of successive outgroups beyond the sister (default 1)",
    )
    hal.add_argument("--max-mer", type=int, default=5,
                     help="context length; must be odd (default 5)")
    hal.add_argument(
        "--method", default="parsimony",
        choices=["parsimony", "ml", "hal-ancestor"],
        help="ancestral inference: Fitch parsimony (hard calls, fast), "
             "maximum likelihood (soft weights, branch-length aware), or "
             "hal-ancestor (read the ancestor Cactus already reconstructed)",
    )
    hal.add_argument(
        "--min-sister-divergence", type=float, default=0.0, metavar="D",
        help="skip comparators closer than D substitutions/site; guards "
             "against pairing near-identical assemblies",
    )
    hal.add_argument(
        "--species-table", default=None, metavar="TSV",
        help="TSV mapping assembly to species so same-species comparators "
             "are skipped; needs columns assembly and species (or the "
             "focal_assembly/SpeciesName pair used by Selected_trio_sets.tsv)",
    )
    hal.add_argument(
        "--flank-threshold", type=float, default=0.95,
        help="ml only: minimum posterior for a flanking base to be hard-called",
    )
    hal.add_argument("--min-block", type=int, default=50,
                     help="minimum focal block length in bp (default 50)")
    hal.add_argument("--ref-sequence", default=None,
                     help="restrict to one focal sequence/chromosome")
    hal.add_argument("--maf", default=None, help="where to put the MAF")
    hal.add_argument("--reuse-maf", action="store_true",
                     help="reuse an existing MAF instead of re-extracting")
    hal.add_argument("--list-species", action="store_true",
                     help="print the alignment's genomes and exit")
    hal.add_argument("-o", "--output", default="counts.json")
    hal.set_defaults(func=cmd_hal_counts)

    trios = sub.add_parser(
        "select-trios", help="choose comparators for every genome in a tree"
    )
    trios.add_argument("--hal", default=None, help="path to the .hal alignment")
    trios.add_argument("--tree", default=None,
                       help="Newick file, instead of reading it from the HAL")
    trios.add_argument("--outgroups", type=int, default=1)
    trios.add_argument("--min-sister-divergence", type=float, default=0.0)
    trios.add_argument("--species-table", default=None)
    trios.add_argument("-o", "--output", default="trios.tsv")
    trios.set_defaults(func=cmd_select_trios)

    vcf = sub.add_parser(
        "vcf-counts", help="count contexts and mutations from FASTA + VCF"
    )
    vcf.add_argument("--fasta", required=True,
                     help="feature FASTA, plain or gzipped; headers may be "
                          "bedtools-style name:start-end or plain")
    vcf.add_argument("--vcf", required=True, help="VCF, plain or bgzipped")
    vcf.add_argument("--max-mer", type=int, default=5,
                     help="context length (default 5)")
    vcf.add_argument("--growth", default="alternating",
                     choices=["alternating", "centred"],
                     help="alternating adds one flanking base per layer "
                          "(legacy geometry); centred adds one on each side "
                          "and requires odd max-mer (default alternating)")
    vcf.add_argument("--alternation", default="right", choices=["right", "left"],
                     help="alternating growth only (default right)")
    vcf.add_argument(
        "--dataset", default="ALL", choices=list(DATASETS),
        help="parity class: ALL, or EVEN/ODD for interleaved halves. The "
             "class is selected from numerator and denominator together "
             "(default ALL)",
    )
    vcf.add_argument("--min-ac", type=int, default=1)
    vcf.add_argument("--max-ac", type=int, default=None)
    vcf.add_argument("--min-af", type=float, default=0.0)
    vcf.add_argument("--max-af", type=float, default=0.85)
    vcf.add_argument("--min-quality", type=float, default=None)
    vcf.add_argument("--ac-key", default="AC", help="INFO key for allele count")
    vcf.add_argument("--an-key", default="AN", help="INFO key for allele number")
    vcf.add_argument("--quality-key", default=None,
                     help="INFO key to threshold with --min-quality")
    vcf.add_argument(
        "--keep-multiallelic", action="store_true",
        help="retain multiallelic sites instead of discarding them; required "
             "by --likelihood recurrent",
    )
    vcf.add_argument(
        "--no-high-confidence", action="store_true",
        help="also count soft-masked lowercase bases; by default only "
             "uppercase ACGT contribute",
    )
    vcf.add_argument("--name", default=None, help="label carried into outputs")
    vcf.add_argument("-o", "--output", default="counts.json")
    vcf.set_defaults(func=cmd_vcf_counts)

    fitp = sub.add_parser("fit", help="fit the hierarchical model to counts")
    fitp.add_argument("--counts", required=True)
    fitp.add_argument("--likelihood", default="binomial",
                      choices=["binomial", "ctmc", "recurrent"])
    fitp.add_argument(
        "--linkage", default="cut", choices=["cut", "joint"],
        help="cut runs each layer against pooled counts (legacy structure, "
             "memory bounded to two layers); joint samples all layers in one "
             "chain with the data entering once at the leaves, practical to "
             "about max-mer 7 (default cut)",
    )
    fitp.add_argument(
        "--parameterization", default="alr", choices=["alr", "multiplicative"],
        help="alr removes the simplex constraint structurally; multiplicative "
             "keeps legacy's phi = log(p_v / p_parent) for comparability of "
             "phi values (default alr)",
    )
    fitp.add_argument("--branch-length", type=float, default=1.0,
                      help="ctmc only; only Q*t is identified (default 1)")
    fitp.add_argument("--iterations", type=int, default=40000)
    fitp.add_argument("--burnin", type=int, default=None,
                      help="default: half of --iterations")
    fitp.add_argument("--thin", type=int, default=10)
    fitp.add_argument("--set-sigma", type=float, default=0.8540486,
                      help="fix the slab scale; omit to estimate it")
    fitp.add_argument("--c", type=float, default=0.04472136,
                      help="spike/slab scale ratio")
    fitp.add_argument("--proposal-sd", type=float, default=0.05)
    fitp.add_argument("--seed", type=int, default=14)
    fitp.add_argument("--progress", action="store_true")
    fitp.add_argument("--rates", default=None,
                      help="write posterior mean rates as JSON")
    fitp.add_argument("-o", "--output", default=None,
                      help="directory for memory-mapped traces")
    fitp.set_defaults(func=cmd_fit)

    summ = sub.add_parser(
        "summarize",
        help="per-layer rate and inclusion-probability JSONs from traces",
        description=(
            "Summarise finished fits without refitting. Every layer is "
            "emitted per seed and pooled across seeds, with posterior mean "
            "rates, spike-and-slab inclusion probabilities, posterior SDs "
            "and mean phi. Rates are slotted by destination base, [A,C,G,T], "
            "matching legacy Baymer's rate_dict rather than the "
            "reference-first order of `fit --rates`."
        ),
    )
    summ.add_argument(
        "--traces", required=True, nargs="+", metavar="DIR",
        help="one trace directory per seed; provenance and the seed label "
             "are read from each directory's manifest.json",
    )
    summ.add_argument("--prefix", default="rates",
                      help="output filename stem (default rates)")
    summ.add_argument("--final-label", default="final", dest="final_label",
                      help="label for the pooled file (default final)")
    summ.add_argument(
        "--digits", type=int, default=8,
        help="significant digits retained; traces are float32, so 8 "
             "discards nothing real (default 8)",
    )
    summ.add_argument("--progress", action="store_true")
    summ.add_argument("-o", "--output", required=True,
                      help="directory for the JSON files")
    summ.set_defaults(func=cmd_summarize)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

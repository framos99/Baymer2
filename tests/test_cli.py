"""The command-line surface.

Batch pipelines drive everything through this interface, so a flag that
silently stops being accepted breaks a whole sweep rather than one
function call.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from baymer2.cli import _read_count_json, build_parser, main

from conftest import LEGACY_ROOT_ENV, LEGACY_TUTORIAL

FASTA = LEGACY_TUTORIAL / "testingname.chr1.cpg_islands.ancestral.fa"
VCF = LEGACY_TUTORIAL / "chr1.AFR.cpg_islands.vcf.bgz"

# Totals the original implementation reports for this dataset.
LEGACY_TOTAL_CONTEXTS = 799_381
LEGACY_VARIANTS_KEPT = 12_000


@pytest.fixture(scope="module")
def tutorial_inputs():
    if not FASTA.exists() or not VCF.exists():
        pytest.skip(f"legacy tutorial data not available; set ${LEGACY_ROOT_ENV}")
    return FASTA, VCF


# --------------------------------------------------------------------------
# Parser surface
# --------------------------------------------------------------------------


def test_every_subcommand_is_present():
    parser = build_parser()
    for argv in (
        ["hal-counts", "--hal", "x.hal"],
        ["vcf-counts", "--fasta", "x.fa", "--vcf", "x.vcf"],
        ["select-trios"],
        ["fit", "--counts", "x.json"],
        ["summarize", "--traces", "traces.rs14", "-o", "out"],
    ):
        assert parser.parse_args(argv).command == argv[0]


def test_tree_depth_is_configurable_on_both_counting_paths():
    """Both senses of depth: k-mer context and phylogenetic."""
    parser = build_parser()
    hal = parser.parse_args(
        ["hal-counts", "--hal", "x.hal", "--focal", "S",
         "--max-mer", "9", "--outgroups", "3"]
    )
    assert hal.max_mer == 9 and hal.outgroups == 3

    vcf = parser.parse_args(
        ["vcf-counts", "--fasta", "x.fa", "--vcf", "x.vcf", "--max-mer", "7"]
    )
    assert vcf.max_mer == 7

    trios = parser.parse_args(["select-trios", "--outgroups", "2"])
    assert trios.outgroups == 2


def test_fit_takes_no_max_mer():
    """Geometry comes from the counts file, so the two cannot disagree."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["fit", "--counts", "x.json", "--max-mer", "9"])


def test_fit_exposes_linkage_and_parameterization():
    parser = build_parser()
    args = parser.parse_args(
        ["fit", "--counts", "x.json", "--linkage", "joint",
         "--parameterization", "multiplicative"]
    )
    assert args.linkage == "joint"
    assert args.parameterization == "multiplicative"
    assert parser.parse_args(["fit", "--counts", "x.json"]).linkage == "cut"


def test_invalid_choices_are_rejected():
    parser = build_parser()
    for argv in (
        ["fit", "--counts", "x.json", "--linkage", "sideways"],
        ["fit", "--counts", "x.json", "--parameterization", "polar"],
        ["fit", "--counts", "x.json", "--likelihood", "poisson"],
        ["vcf-counts", "--fasta", "f", "--vcf", "v", "--dataset", "MIDDLE"],
        ["vcf-counts", "--fasta", "f", "--vcf", "v", "--growth", "spiral"],
        ["hal-counts", "--hal", "h", "--method", "guessing"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_vcf_counts_filter_flags_are_wired():
    args = build_parser().parse_args([
        "vcf-counts", "--fasta", "f", "--vcf", "v",
        "--min-ac", "2", "--max-ac", "50", "--min-af", "0.01", "--max-af", "0.9",
        "--min-quality", "30", "--ac-key", "AC_raw", "--an-key", "AN_raw",
        "--quality-key", "QD", "--keep-multiallelic", "--no-high-confidence",
    ])
    assert (args.min_ac, args.max_ac) == (2, 50)
    assert (args.min_af, args.max_af) == (0.01, 0.9)
    assert args.min_quality == 30
    assert (args.ac_key, args.an_key, args.quality_key) == ("AC_raw", "AN_raw", "QD")
    assert args.keep_multiallelic and args.no_high_confidence


# --------------------------------------------------------------------------
# vcf-counts against the original implementation's totals
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_vcf_counts_reproduces_legacy_totals(tutorial_inputs, tmp_path):
    fasta, vcf = tutorial_inputs
    out = tmp_path / "counts.json"
    assert main([
        "vcf-counts", "--fasta", str(fasta), "--vcf", str(vcf),
        "--max-mer", "5", "--dataset", "ALL", "--name", "AFR", "-o", str(out),
    ]) == 0

    payload = json.loads(out.read_text())
    assert payload["qc"]["kept"] == LEGACY_VARIANTS_KEPT
    assert payload["dataset"] == "ALL"
    assert payload["growth"] == "alternating"

    table = _read_count_json(out)
    assert table.total_sites == LEGACY_TOTAL_CONTEXTS
    assert table.leaf_counts[:, 1:].sum() == LEGACY_VARIANTS_KEPT


@pytest.mark.slow
def test_parity_datasets_partition_the_data(tutorial_inputs, tmp_path):
    """The parity fix, exercised through the CLI: EVEN + ODD == ALL."""
    fasta, vcf = tutorial_inputs
    counts = {}
    for dataset in ("ALL", "EVEN", "ODD"):
        out = tmp_path / f"{dataset}.json"
        main(["vcf-counts", "--fasta", str(fasta), "--vcf", str(vcf),
              "--max-mer", "5", "--dataset", dataset, "-o", str(out)])
        counts[dataset] = _read_count_json(out).leaf_counts
    np.testing.assert_array_equal(
        counts["EVEN"] + counts["ODD"], counts["ALL"]
    )


@pytest.mark.slow
def test_emitted_counts_round_trip_and_fit(tutorial_inputs, tmp_path):
    fasta, vcf = tutorial_inputs
    counts = tmp_path / "counts.json"
    rates = tmp_path / "rates.json"
    main(["vcf-counts", "--fasta", str(fasta), "--vcf", str(vcf),
          "--max-mer", "3", "-o", str(counts)])
    assert main([
        "fit", "--counts", str(counts), "--likelihood", "binomial",
        "--iterations", "600", "--thin", "3", "--rates", str(rates),
    ]) == 0

    emitted = json.loads(rates.read_text())
    assert len(emitted) == 32          # deepest layer of a 3-mer model
    for probs in emitted.values():
        assert len(probs) == 4
        assert sum(probs) == pytest.approx(1.0)


@pytest.mark.slow
def test_centred_growth_requires_odd_max_mer(tutorial_inputs, tmp_path):
    fasta, vcf = tutorial_inputs
    with pytest.raises(ValueError, match="max_mer must be odd"):
        main(["vcf-counts", "--fasta", str(fasta), "--vcf", str(vcf),
              "--max-mer", "4", "--growth", "centred",
              "-o", str(tmp_path / "x.json")])


# --------------------------------------------------------------------------
# Empty-count guards
#
# A production run produced zero usable blocks, wrote an all-zero count table,
# then sampled 40,000 iterations against it and emitted rate files in which
# every context held the initialisation floor. Nothing failed. These pin the
# guards that now stop that.
# --------------------------------------------------------------------------


def test_fit_refuses_an_all_empty_count_table():
    from baymer2.backends.numba_mh import LayerConfig
    from baymer2.contexts import ContextTree
    from baymer2.counts import CountTable
    from baymer2.model import ModelConfig, fit

    tree = ContextTree(max_mer=3, growth="centred")
    empty = CountTable(tree=tree, leaf_counts=np.zeros((tree.n_leaves, 4)))
    with pytest.raises(ValueError, match="every context .* is empty"):
        fit(empty, ModelConfig(
            max_mer=3, growth="centred",
            default_layer=LayerConfig(n_iterations=200, burnin=100, thin=2,
                                      set_sigma=0.85),
        ))


def test_vcf_counts_fails_when_nothing_is_counted(tmp_path, capsys):
    """A FASTA shorter than the context length yields no opportunity."""
    fasta = tmp_path / "tiny.fa"
    fasta.write_text(">chr1:0-2\nAC\n")
    vcf = tmp_path / "v.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    assert main([
        "vcf-counts", "--fasta", str(fasta), "--vcf", str(vcf),
        "--max-mer", "5", "-o", str(tmp_path / "counts.json"),
    ]) == 1
    assert "no contexts were counted" in capsys.readouterr().err
    assert not (tmp_path / "counts.json").exists()


def test_vcf_counts_writes_nothing_on_failure(tmp_path):
    """A zero table must not be left behind to be mistaken for a result."""
    fasta = tmp_path / "tiny.fa"
    fasta.write_text(">chr1:0-2\nAC\n")
    vcf = tmp_path / "v.vcf"
    vcf.write_text("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    out = tmp_path / "counts.json"
    main(["vcf-counts", "--fasta", str(fasta), "--vcf", str(vcf),
          "--max-mer", "5", "-o", str(out)])
    assert not out.exists()

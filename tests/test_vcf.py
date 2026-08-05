"""FASTA/VCF counting, including the EVEN/ODD parity fix.

The counters are checked against legacy's own reported totals on the
tutorial data (799,381 context occurrences, 12,000 variants kept), and
the parity mismatch is reproduced explicitly so a regression cannot
reintroduce it quietly.
"""

from __future__ import annotations

import gzip

import numpy as np
import pytest

from baymer2.contexts import ContextTree
from baymer2.io.vcf import (
    VariantFilter,
    build_count_table,
    count_contexts,
    count_mutations,
    iter_fasta,
)

from conftest import LEGACY_ROOT_ENV, LEGACY_TUTORIAL

# Totals legacy printed for this dataset.
LEGACY_TOTAL_CONTEXTS = 799_381
LEGACY_VARIANTS_KEPT = 12_000

FASTA = LEGACY_TUTORIAL / "testingname.chr1.cpg_islands.ancestral.fa"
VCF = LEGACY_TUTORIAL / "chr1.AFR.cpg_islands.vcf.bgz"


@pytest.fixture(scope="module")
def tree() -> ContextTree:
    return ContextTree(max_mer=5, alternation="right")


@pytest.fixture(scope="module")
def tutorial(tree):
    if not FASTA.exists() or not VCF.exists():
        pytest.skip(f"legacy tutorial data not available; set ${LEGACY_ROOT_ENV}")
    contexts = count_contexts(FASTA, tree, high_confidence=True)
    mutations = count_mutations(
        VCF, FASTA, tree, filters=VariantFilter(min_ac=1, max_af=0.85)
    )
    return contexts, mutations


# --------------------------------------------------------------------------
# FASTA reading
# --------------------------------------------------------------------------


def test_parses_bedtools_headers(tmp_path):
    path = tmp_path / "r.fa"
    path.write_text(">chr1:100-104\nACGT\n>chr1:200-204\nTTTT\n")
    regions = list(iter_fasta(path))
    assert [(r.name, r.start, r.sequence) for r in regions] == [
        ("chr1", 100, "ACGT"),
        ("chr1", 200, "TTTT"),
    ]
    assert regions[0].end == 104


def test_plain_headers_start_at_zero(tmp_path):
    """Legacy raised IndexError on these; whole-chromosome FASTA is valid."""
    path = tmp_path / "p.fa"
    path.write_text(">chr1\nACGT\n")
    region = next(iter_fasta(path))
    assert (region.name, region.start) == ("chr1", 0)


def test_reads_gzipped_fasta(tmp_path):
    """context_counter accepted .gz, mutation_counter did not; both do now."""
    path = tmp_path / "r.fa.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(">chr1:0-4\nACGT\n")
    assert next(iter_fasta(path)).sequence == "ACGT"


def test_multiline_records_are_joined(tmp_path):
    path = tmp_path / "m.fa"
    path.write_text(">chr1:0-8\nACGT\nTTTT\n")
    assert next(iter_fasta(path)).sequence == "ACGTTTTT"


# --------------------------------------------------------------------------
# Context counting
# --------------------------------------------------------------------------


def test_context_totals_match_legacy(tutorial):
    contexts, _ = tutorial
    assert contexts.sum() == LEGACY_TOTAL_CONTEXTS


def test_context_parities_are_balanced(tutorial):
    """Interleaved halves should be nearly equal in size."""
    contexts, _ = tutorial
    even, odd = contexts[:, 0].sum(), contexts[:, 1].sum()
    assert abs(even - odd) / (even + odd) < 0.01


def test_high_confidence_rejects_soft_masked(tmp_path, tree):
    path = tmp_path / "s.fa"
    path.write_text(">chr1:0-10\nAAAaaAAAAA\n")
    strict = count_contexts(path, tree, high_confidence=True)
    loose = count_contexts(path, tree, high_confidence=False)
    assert strict.sum() < loose.sum()


def test_windows_with_n_are_skipped(tmp_path, tree):
    path = tmp_path / "n.fa"
    path.write_text(">chr1:0-10\nAAAANAAAAA\n")
    counts = count_contexts(path, tree)
    # 6 windows of length 5; those touching the N are dropped
    assert counts.sum() == 1


def test_counts_land_in_the_folded_context(tmp_path, tree):
    path = tmp_path / "f.fa"
    path.write_text(">chr1:0-5\nAAGAA\n")  # focal G at index 2 -> folds
    counts = count_contexts(path, tree)
    contexts = tree.contexts(tree.n_layers - 1)
    assert counts.sum() == 1
    assert contexts[int(np.argmax(counts.sum(axis=1)))] == "TTCTT"


# --------------------------------------------------------------------------
# Mutation counting
# --------------------------------------------------------------------------


def test_variants_kept_matches_legacy(tutorial):
    _, mutations = tutorial
    assert mutations.n_kept == LEGACY_VARIANTS_KEPT


def test_no_reference_mismatches_on_the_tutorial(tutorial):
    """The FASTA is the ancestral sequence the VCF was polarised against."""
    _, mutations = tutorial
    assert mutations.n_ref_mismatch == 0


def test_summary_reports_every_tally(tutorial):
    _, mutations = tutorial
    summary = mutations.summary()
    assert summary["kept"] == LEGACY_VARIANTS_KEPT
    assert set(summary) >= {"records", "kept", "failed_filters", "multiallelic"}


def _write_vcf(path, rows, header="##fileformat=VCFv4.2\n#CHROM POS ID REF ALT QUAL FILTER INFO\n"):
    lines = [header.replace(" ", "\t")]
    for row in rows:
        lines.append("\t".join(str(v) for v in row) + "\n")
    path.write_text("".join(lines))


def test_multiallelic_sites_are_dropped_by_default(tmp_path, tree):
    fasta = tmp_path / "f.fa"
    fasta.write_text(">chr1:0-5\nAAAAA\n")
    vcf = tmp_path / "v.vcf"
    _write_vcf(vcf, [("chr1", 3, ".", "A", "C,G", ".", "PASS", "AC=5,5;AN=100")])

    dropped = count_mutations(vcf, fasta, tree)
    assert dropped.n_multiallelic == 1
    assert dropped.n_kept == 0


def test_multiallelic_sites_can_be_retained(tmp_path, tree):
    """The clearest signature of recurrence; legacy discards it always."""
    fasta = tmp_path / "f.fa"
    fasta.write_text(">chr1:0-5\nAAAAA\n")
    vcf = tmp_path / "v.vcf"
    _write_vcf(vcf, [("chr1", 3, ".", "A", "C,G", ".", "PASS", "AC=5,5;AN=100")])

    kept = count_mutations(
        vcf, fasta, tree, filters=VariantFilter(keep_multiallelic=True)
    )
    assert kept.n_kept == 2


def test_allele_frequency_filter_is_applied(tmp_path, tree):
    fasta = tmp_path / "f.fa"
    fasta.write_text(">chr1:0-5\nAAAAA\n")
    vcf = tmp_path / "v.vcf"
    _write_vcf(vcf, [("chr1", 3, ".", "A", "C", ".", "PASS", "AC=95;AN=100")])
    assert count_mutations(vcf, fasta, tree,
                           filters=VariantFilter(max_af=0.85)).n_kept == 0
    assert count_mutations(vcf, fasta, tree,
                           filters=VariantFilter(max_af=0.99)).n_kept == 1


def test_reference_mismatch_is_reported(tmp_path, tree):
    fasta = tmp_path / "f.fa"
    fasta.write_text(">chr1:0-5\nAAAAA\n")
    vcf = tmp_path / "v.vcf"
    _write_vcf(vcf, [("chr1", 3, ".", "T", "C", ".", "PASS", "AC=5;AN=100")])
    result = count_mutations(vcf, fasta, tree)
    assert result.n_ref_mismatch == 1
    assert result.n_kept == 0


def test_indels_are_ignored(tmp_path, tree):
    fasta = tmp_path / "f.fa"
    fasta.write_text(">chr1:0-5\nAAAAA\n")
    vcf = tmp_path / "v.vcf"
    _write_vcf(vcf, [("chr1", 3, ".", "A", "AT", ".", "PASS", "AC=5;AN=100")])
    assert count_mutations(vcf, fasta, tree).n_kept == 0


# --------------------------------------------------------------------------
# The parity fix
# --------------------------------------------------------------------------


def test_even_and_odd_agree_with_each_other(tutorial, tree):
    """Two interleaved halves of the same data must give the same answer."""
    contexts, mutations = tutorial
    rates = {}
    for dataset in ("EVEN", "ODD"):
        table = build_count_table(
            contexts, mutations.counts, tree, dataset=dataset
        )
        rates[dataset] = table.leaf_counts[:, 1:].sum() / table.total_sites
    assert rates["EVEN"] == pytest.approx(rates["ODD"], rel=0.02)


def test_all_is_the_sum_of_even_and_odd(tutorial, tree):
    contexts, mutations = tutorial
    combined = build_count_table(contexts, mutations.counts, tree, dataset="ALL")
    even = build_count_table(contexts, mutations.counts, tree, dataset="EVEN")
    odd = build_count_table(contexts, mutations.counts, tree, dataset="ODD")
    np.testing.assert_allclose(
        combined.leaf_counts, even.leaf_counts + odd.leaf_counts
    )


def test_legacy_parity_mismatch_distorts_context_rates(tutorial, tree):
    """Reproduce the bug and show what it costs.

    Legacy draws the numerator from one parity class and the denominator
    from the other. Aggregate rates barely move -- the two classes hold
    almost identical *total* opportunity -- but individual contexts
    split unevenly between parities, and per-context rates are exactly
    what the model estimates.
    """
    contexts, mutations = tutorial
    numerators = mutations.counts

    for parity in (0, 1):
        correct_den = contexts[:, parity]
        swapped_den = contexts[:, 1 - parity]
        numer = numerators[:, :, parity][:, 1:].sum(axis=1)

        usable = (correct_den > 0) & (swapped_den > 0)
        correct = numer[usable] / correct_den[usable]
        swapped = numer[usable] / swapped_den[usable]
        error = np.abs(swapped - correct) / np.maximum(correct, 1e-12)

        # Aggregate barely moves...
        assert abs(numer.sum() / swapped_den.sum()
                   - numer.sum() / correct_den.sum()) < 1e-4
        # ...but per-context it is material.
        assert np.median(error) > 0.01
        assert np.percentile(error, 95) > 0.05


def test_inconsistent_inputs_are_refused(tree):
    """More alternates than opportunities means the two sides disagree."""
    n = tree.n_leaves
    contexts = np.zeros((n, 2))
    contexts[:, 0] = 5
    mutations = np.zeros((n, 4, 2))
    mutations[:, 1, 0] = 100
    with pytest.raises(ValueError, match="disagree"):
        build_count_table(contexts, mutations, tree, dataset="EVEN")


def test_unknown_dataset_is_rejected(tutorial, tree):
    contexts, mutations = tutorial
    with pytest.raises(ValueError, match="dataset must be one of"):
        build_count_table(contexts, mutations.counts, tree, dataset="MIDDLE")


def test_built_table_is_internally_consistent(tutorial, tree):
    contexts, mutations = tutorial
    table = build_count_table(contexts, mutations.counts, tree, dataset="ALL")
    assert table.total_sites == LEGACY_TOTAL_CONTEXTS
    assert table.leaf_counts[:, 1:].sum() == LEGACY_VARIANTS_KEPT
    assert np.all(table.leaf_counts >= 0)
    totals = [table.layer_stats(l).sum() for l in range(tree.n_layers)]
    assert len(set(np.round(totals, 6))) == 1

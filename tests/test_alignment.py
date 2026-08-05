"""HAL reading, Fitch parsimony, and count assembly from an alignment.

Tests that need the HAL toolkit are marked ``hal`` and skip when the
cactus binaries are absent. The parsimony tests do not need it.
"""

from __future__ import annotations

import numpy as np
import pytest

from baymer2.ancestral.parsimony import (
    CATEGORIES,
    AlignmentCounter,
    build_kmer_lookup,
    fitch_ancestral,
)
from baymer2.contexts import BASES, ContextTree, reverse_complement
from baymer2.io.hal import (
    BlockTally,
    HalRuntime,
    MafRecord,
    order_by_specificity,
    sample_src_names,
    ancestral_genomes,
    extract_maf,
    hal_genomes,
    hal_tree,
    iter_maf_blocks,
    parse_maf,
)

from conftest import CACTUS_BIN_ENV, EVOLVER_HAL, EVOLVER_HAL_ENV

FOCAL = "simHuman_chr6"


def _legacy_trio_call(f: str, s: str, o: str):
    """Verbatim three-taxon rules from ``call_block`` in the trio caller."""
    if f == s == o:
        return "invariant", f
    if f != s and s == o:
        return "derived_on_focal", s
    if f == o and f != s:
        return "derived_on_sister", f
    if f == s and f != o:
        return "derived_on_outgroup", f
    return "unresolved", None


@pytest.fixture(scope="module")
def runtime() -> HalRuntime:
    rt = HalRuntime.detect()
    if not rt.available:
        pytest.skip(f"no HAL toolkit available; set ${CACTUS_BIN_ENV} "
                    f"or put halStats on PATH")
    return rt


@pytest.fixture(scope="module")
def hal(runtime):
    if not EVOLVER_HAL.exists():
        pytest.skip(f"evolverMammals.hal not available at {EVOLVER_HAL}; "
                    f"set ${EVOLVER_HAL_ENV}")
    return EVOLVER_HAL


@pytest.fixture(scope="module")
def trio_maf(hal, runtime, tmp_path_factory):
    tree = hal_tree(hal, runtime)
    sel = tree.select(FOCAL, n_outgroups=1)
    out = tmp_path_factory.mktemp("maf") / "trio.maf"
    extract_maf(hal, sel.focal, sel.targets, out, runtime=runtime)
    return out, sel


# --------------------------------------------------------------------------
# Fitch parsimony
# --------------------------------------------------------------------------


def test_trio_parsimony_reproduces_the_legacy_rules():
    """All 64 three-taxon configurations must agree exactly."""
    for f in BASES:
        for s in BASES:
            for o in BASES:
                _, expected = _legacy_trio_call(f, s, o)
                assert fitch_ancestral(f, s, [o]) == expected, (f, s, o)


def test_parsimony_resolves_with_extra_outgroups():
    """A second outgroup can break a tie the trio leaves unresolved."""
    assert fitch_ancestral("A", "C", ["G"]) is None
    assert fitch_ancestral("A", "C", ["A", "A"]) == "A"
    assert fitch_ancestral("A", "C", ["C", "C"]) == "C"


def test_parsimony_rejects_gaps_and_ambiguity():
    assert fitch_ancestral("-", "A", ["A"]) is None
    assert fitch_ancestral("A", "N", ["A"]) is None


def test_parsimony_is_case_insensitive():
    assert fitch_ancestral("a", "a", ["a"]) == "A"


# --------------------------------------------------------------------------
# k-mer lookup
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 5])
def test_kmer_lookup_folds_every_kmer(k):
    tree = ContextTree(max_mer=k, growth="centred")
    node_of, complemented, folded_focal = build_kmer_lookup(tree)
    contexts = tree.contexts(tree.n_layers - 1)
    focal_idx = tree.focal_index(k)

    assert node_of.shape == (4**k,)
    assert np.all(node_of >= 0)

    for code in range(4**k):
        digits = []
        rest = code
        for _ in range(k):
            digits.append(rest & 3)
            rest >>= 2
        kmer = "".join(BASES[d] for d in reversed(digits))
        expected = kmer if kmer[focal_idx] in "AC" else reverse_complement(kmer)
        assert contexts[node_of[code]] == expected
        assert complemented[code] == (kmer[focal_idx] not in "AC")
        assert folded_focal[code] == (0 if expected[focal_idx] == "A" else 1)


# --------------------------------------------------------------------------
# MAF parsing
# --------------------------------------------------------------------------


def test_parse_maf_reads_blocks(tmp_path):
    maf = tmp_path / "tiny.maf"
    maf.write_text(
        "##maf version=1\n"
        "a score=0\n"
        "s sp1.chr1 10 4 + 1000 ACGT\n"
        "s sp2.chr1 20 4 + 1000 ACGA\n"
        "\n"
        "a score=0\n"
        "s sp1.chr1 30 3 + 1000 TT-A\n"
        "s sp2.chr1 40 4 + 1000 TTGA\n"
    )
    blocks = list(parse_maf(maf))
    assert len(blocks) == 2
    known = order_by_specificity({"sp1", "sp2"})
    assert blocks[0][0].genome_in(known) == "sp1"
    assert blocks[0][0].sequence_in("sp1") == "chr1"
    assert blocks[0][0].start == 10
    assert blocks[0][1].seq == "ACGA"
    assert blocks[1][0].seq == "TT-A"


def test_parse_maf_rejects_malformed_lines(tmp_path):
    maf = tmp_path / "bad.maf"
    maf.write_text("a\ns sp1.chr1 10 4 + ACGT\n")
    with pytest.raises(ValueError, match="malformed MAF"):
        list(parse_maf(maf))


def test_iter_blocks_filters_on_membership_and_length(tmp_path):
    maf = tmp_path / "f.maf"
    maf.write_text(
        "a\ns F.c1 0 4 + 100 ACGT\ns S.c1 0 4 + 100 ACGT\ns O.c1 0 4 + 100 ACGT\n"
        "\n"
        "a\ns F.c1 4 4 + 100 ACGT\ns S.c1 4 4 + 100 ACGT\n"          # no outgroup
        "\n"
        "a\ns S.c1 8 4 + 100 ACGT\ns O.c1 8 4 + 100 ACGT\n"          # no focal
    )
    kept = list(iter_maf_blocks(maf, "F", ["S", "O"], min_block=1))
    assert len(kept) == 1
    assert set(kept[0]) == {"F", "S", "O"}
    assert list(iter_maf_blocks(maf, "F", ["S", "O"], min_block=99)) == []


def test_iter_blocks_drops_duplicated_genomes(tmp_path):
    """A repeated genome means a paralogue slipped past --noDupes."""
    maf = tmp_path / "dup.maf"
    maf.write_text(
        "a\ns F.c1 0 4 + 100 ACGT\ns S.c1 0 4 + 100 ACGT\n"
        "s S.c2 0 4 + 100 ACGT\ns O.c1 0 4 + 100 ACGT\n"
    )
    assert list(iter_maf_blocks(maf, "F", ["S", "O"], min_block=1)) == []


def test_iter_blocks_rejects_a_reversed_focal(tmp_path):
    maf = tmp_path / "rev.maf"
    maf.write_text(
        "a\ns F.c1 0 4 - 100 ACGT\ns S.c1 0 4 + 100 ACGT\ns O.c1 0 4 + 100 ACGT\n"
    )
    with pytest.raises(ValueError, match="strand"):
        list(iter_maf_blocks(maf, "F", ["S", "O"], min_block=1))


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------


def _block(focal: str, sister: str, outgroup: str) -> dict:
    return {
        "F": MafRecord("F.c1", 0, len(focal.replace("-", "")), "+", 1000, focal),
        "S": MafRecord("S.c1", 0, len(sister.replace("-", "")), "+", 1000, sister),
        "O": MafRecord("O.c1", 0, len(outgroup.replace("-", "")), "+", 1000, outgroup),
    }


def _counter(k: int = 3) -> AlignmentCounter:
    return AlignmentCounter(
        ContextTree(max_mer=k, growth="centred"), "F", "S", ["O"]
    )


def test_invariant_sites_land_in_the_reference_slot():
    counter = _counter()
    counter.add_block(_block("AAAAA", "AAAAA", "AAAAA"))
    table = counter.to_count_table()
    # 5 columns, k=3 -> 3 countable centres, all AAA invariant
    assert table.counts_for("AAA")[0] == 3
    assert table.counts_for("AAA")[1:].sum() == 0
    assert counter.category_summary()["invariant"] == 5


def test_a_focal_substitution_is_counted_in_its_ancestral_context():
    """The bias fix: the context is the ancestral k-mer, not the focal one.

    Focal ``ACA`` against ``AGA``/``AGA`` means the site was ancestrally
    ``G`` and mutated to ``C`` on the focal branch. The opportunity
    therefore belongs to the ancestral 3-mer ``AGA`` (folded to ``TCT``),
    not to the focal 3-mer ``ACA``. Legacy's ``_focal_kmer`` would have
    charged it to ``ACA``.
    """
    counter = _counter()
    counter.add_block(_block("ACA", "AGA", "AGA"))
    table = counter.to_count_table()

    row = table.counts_for("TCT")
    assert row.sum() == 1
    # Folding complements the observed allele too: focal C reads as G.
    assert row[2] == 1, row
    # The focal-derived context must record nothing.
    assert table.counts_for("ACA").sum() == 0
    assert counter.category_summary()["derived_on_focal"] == 1


def test_unresolved_neighbour_removes_the_site():
    """An ambiguous flank must delete the site, not contaminate it."""
    counter = _counter()
    # Column 0 is A/C/G across the trio, so its ancestral state is
    # unresolvable; the only countable centre (column 1) spans it.
    counter.add_block(_block("AAA", "CAA", "GAA"))
    assert counter.counts.sum() == 0
    assert counter.category_summary()["unresolved"] == 1
    assert counter.category_summary()["invariant"] == 2


def test_a_resolved_neighbourhood_further_along_is_still_counted():
    """Only k-mers touching the ambiguity are lost, not the whole block."""
    counter = _counter()
    counter.add_block(_block("AAAAA", "CAAAA", "GAAAA"))
    # centres at ungapped 1..3; only the one spanning column 0 is dropped
    assert counter.counts.sum() == 2


def test_gaps_do_not_consume_focal_coordinates():
    counter = _counter()
    counter.add_block(_block("A-AAA", "AGAAA", "AGAAA"))
    # focal has 4 ungapped bases, so 2 countable centres
    assert counter.counts.sum() == 2


def test_columns_with_a_missing_comparator_are_skipped():
    counter = _counter()
    counter.add_block(_block("AAAAA", "AA-AA", "AAAAA"))
    # the centre column is unusable, so no k-mer spanning it is counted
    assert counter.counts.sum() == 0


def test_counter_requires_an_odd_kmer():
    with pytest.raises(ValueError, match="max_mer must be odd"):
        AlignmentCounter(ContextTree(max_mer=4), "F", "S", ["O"])


def test_counter_requires_an_outgroup():
    with pytest.raises(ValueError, match="outgroup"):
        AlignmentCounter(ContextTree(max_mer=3, growth="centred"), "F", "S", [])


def test_categories_are_exhaustive():
    counter = _counter()
    counter.add_block(_block("ACGTA", "AAGTA", "AAGTC"))
    assert sum(counter.category_totals) == 5
    assert set(counter.category_summary()) == set(CATEGORIES)


# --------------------------------------------------------------------------
# Against the real HAL
# --------------------------------------------------------------------------


@pytest.mark.hal
def test_hal_tree_and_genomes(hal, runtime):
    tree = hal_tree(hal, runtime)
    assert "simHuman_chr6" in tree.leaf_names
    genomes = hal_genomes(hal, runtime)
    assert set(tree.leaf_names).issubset(genomes)


@pytest.mark.hal
def test_hal_carries_reconstructed_ancestors(hal, runtime):
    """Cactus stores ancestral genomes; the hal-ancestor method needs them."""
    ancestors = ancestral_genomes(hal, runtime)
    assert "Anc1" in ancestors
    assert set(ancestors).isdisjoint(hal_tree(hal, runtime).leaf_names)


@pytest.mark.hal
def test_extracted_maf_has_the_focal_on_the_plus_strand(trio_maf):
    maf, sel = trio_maf
    blocks = list(iter_maf_blocks(maf, sel.focal, sel.targets, min_block=50))
    assert len(blocks) > 100
    assert all(b[sel.focal].strand == "+" for b in blocks)


@pytest.mark.hal
def test_counts_from_the_real_alignment_are_coherent(trio_maf):
    maf, sel = trio_maf
    tree = ContextTree(max_mer=3, growth="centred")
    counter = AlignmentCounter(tree, sel.focal, sel.sister, sel.outgroups)
    counter.add_blocks(iter_maf_blocks(maf, sel.focal, sel.targets, min_block=50))

    table = counter.to_count_table()
    assert counter.n_blocks > 100
    assert table.total_sites > 10_000
    assert 0.0 < counter.divergence < 0.5
    # every layer must account for the same sites
    totals = [table.layer_stats(l).sum() for l in range(tree.n_layers)]
    assert len(set(np.round(totals, 6))) == 1


@pytest.mark.hal
def test_cpg_contexts_are_the_most_mutable(trio_maf):
    """The signal any mutation-spectrum method must recover."""
    maf, sel = trio_maf
    tree = ContextTree(max_mer=3, growth="centred")
    counter = AlignmentCounter(tree, sel.focal, sel.sister, sel.outgroups)
    counter.add_blocks(iter_maf_blocks(maf, sel.focal, sel.targets, min_block=50))

    layer = tree.n_layers - 1
    rates = counter.to_count_table().empirical_rates(layer).sum(axis=1)
    contexts = tree.contexts(layer)

    cpg = [i for i, c in enumerate(contexts) if c[1:] == "CG"]
    other = [i for i, c in enumerate(contexts) if c[1] == "C" and c[1:] != "CG"]
    assert cpg and other
    assert rates[cpg].mean() > rates[other].mean(), (
        "CpG contexts must be more mutable than other C contexts"
    )


# --------------------------------------------------------------------------
# hal-ancestor: read the reconstruction Cactus already made
# --------------------------------------------------------------------------


def _anc_block(focal: str, ancestor: str) -> dict:
    from baymer2.io.hal import MafRecord

    return {
        "F": MafRecord("F.c1", 0, len(focal), "+", 1000, focal),
        "A": MafRecord("A.c1", 0, len(ancestor), "+", 1000, ancestor),
    }


def _anc_counter(k: int = 3):
    from baymer2.ancestral.hal_ancestor import AncestorAlignmentCounter

    return AncestorAlignmentCounter(
        ContextTree(max_mer=k, growth="centred"), "F", "A"
    )


def test_ancestor_counter_reads_the_supplied_ancestral_base():
    counter = _anc_counter()
    counter.add_block(_anc_block("ACA", "AGA"))
    table = counter.to_count_table()
    # ancestral 3-mer AGA folds to TCT; focal C reads as G after folding
    assert table.counts_for("TCT")[2] == 1
    assert table.counts_for("ACA").sum() == 0
    assert counter.category_summary()["derived_on_focal"] == 1


def test_ancestor_counter_counts_invariants():
    counter = _anc_counter()
    counter.add_block(_anc_block("AAAAA", "AAAAA"))
    assert counter.to_count_table().counts_for("AAA")[0] == 3
    assert counter.category_summary()["invariant"] == 5


def test_ancestor_gaps_remove_the_site():
    counter = _anc_counter()
    counter.add_block(_anc_block("AAAAA", "AA-AA"))
    assert counter.counts.sum() == 0
    assert counter.category_summary()["unresolved"] == 1


def test_ancestor_counter_requires_aligned_rows():
    counter = _anc_counter()
    with pytest.raises(ValueError, match="differ in length"):
        counter.add_block(_anc_block("AAAAA", "AAA"))


def test_ancestor_counter_requires_an_ancestor_name():
    from baymer2.ancestral.hal_ancestor import AncestorAlignmentCounter

    with pytest.raises(ValueError, match="ancestor genome name"):
        AncestorAlignmentCounter(
            ContextTree(max_mer=3, growth="centred"), "F", ""
        )


@pytest.mark.hal
def test_hal_ancestor_recovers_more_sites_than_a_trio(hal, runtime, tmp_path):
    """A duo needs less of each block to be present than a trio does."""
    from baymer2.ancestral.hal_ancestor import AncestorAlignmentCounter

    tree = hal_tree(hal, runtime)
    sel = tree.select(FOCAL, n_outgroups=1)
    ctx = ContextTree(max_mer=3, growth="centred")

    duo = tmp_path / "duo.maf"
    extract_maf(hal, sel.focal, [sel.ancestor], duo, runtime=runtime,
                no_ancestors=False)
    anc = AncestorAlignmentCounter.from_selection(ctx, sel)
    anc.add_blocks(iter_maf_blocks(duo, sel.focal, [sel.ancestor], min_block=50))

    trio = tmp_path / "trio.maf"
    extract_maf(hal, sel.focal, sel.targets, trio, runtime=runtime)
    par = AlignmentCounter(ctx, sel.focal, sel.sister, sel.outgroups)
    par.add_blocks(iter_maf_blocks(trio, sel.focal, sel.targets, min_block=50))

    assert anc.counts.sum() > par.counts.sum()
    # The spectra must still agree in shape even if scale differs.
    a = anc.to_count_table().empirical_rates(1).sum(axis=1)
    p = par.to_count_table().empirical_rates(1).sum(axis=1)
    assert np.corrcoef(a, p)[0, 1] > 0.85


# --------------------------------------------------------------------------
# Genome-name resolution
#
# A production run against the VGP 577-way alignment produced zero usable
# blocks for every species and wrote a table of zeros without complaint.
# `MafRecord.genome` split `src` on the first dot, but every VGP genome name
# is an assembly accession carrying a version dot, so
# `GCA_048771995.1.CM088821.1` resolved to `GCA_048771995` and matched
# nothing. The bug was invisible on evolverMammals, whose names have no dots
# -- which is exactly why these tests use accession-formatted names.
# --------------------------------------------------------------------------


VGP_FOCAL = "GCA_048771995.1"
VGP_SISTER = "GCA_046129695.1"
VGP_OUTGROUP = "GCF_026979565.1"


def _vgp_maf(tmp_path, n_blocks: int = 3):
    """A MAF with realistic dotted accessions in both name components."""
    lines = ["##maf version=1\n"]
    for i in range(n_blocks):
        start = i * 100
        lines.append("a score=0\n")
        for genome, seqname in (
            (VGP_FOCAL, "CM088821.1"),
            (VGP_SISTER, "CM077712.1"),
            (VGP_OUTGROUP, "JAGFXX010000001.1"),
        ):
            lines.append(
                f"s {genome}.{seqname} {start} 60 + 100000 "
                f"{'ACGTACGTAC' * 6}\n"
            )
        lines.append("\n")
    path = tmp_path / "vgp.maf"
    path.write_text("".join(lines))
    return path


def test_dotted_accessions_resolve_to_the_right_genome():
    known = order_by_specificity({VGP_FOCAL, VGP_SISTER, VGP_OUTGROUP})
    record = MafRecord(f"{VGP_FOCAL}.CM088821.1", 0, 60, "+", 100000, "ACGT")
    assert record.genome_in(known) == VGP_FOCAL
    assert record.sequence_in(VGP_FOCAL) == "CM088821.1"


def test_first_dot_splitting_would_have_failed():
    """Pin the exact defect, so it cannot be reintroduced quietly."""
    src = f"{VGP_FOCAL}.CM088821.1"
    assert src.split(".", 1)[0] == "GCA_048771995"
    assert src.split(".", 1)[0] != VGP_FOCAL


def test_blocks_with_dotted_accessions_are_kept(tmp_path):
    maf = _vgp_maf(tmp_path)
    blocks = list(iter_maf_blocks(
        maf, VGP_FOCAL, [VGP_SISTER, VGP_OUTGROUP], min_block=50
    ))
    assert len(blocks) == 3
    assert set(blocks[0]) == {VGP_FOCAL, VGP_SISTER, VGP_OUTGROUP}


def test_counting_dotted_accessions_produces_sites(tmp_path):
    """The end the production run never reached: non-zero counts."""
    maf = _vgp_maf(tmp_path)
    tree = ContextTree(max_mer=3, growth="centred")
    counter = AlignmentCounter(tree, VGP_FOCAL, VGP_SISTER, [VGP_OUTGROUP])
    counter.add_blocks(
        iter_maf_blocks(maf, VGP_FOCAL, [VGP_SISTER, VGP_OUTGROUP], min_block=50)
    )
    assert counter.n_blocks == 3
    assert counter.to_count_table().total_sites > 0


def test_a_genome_name_that_prefixes_another_resolves_correctly():
    """`GCA_1` must not capture an s-line belonging to `GCA_1.5`."""
    known = order_by_specificity({"GCA_1", "GCA_1.5"})
    assert MafRecord("GCA_1.5.chr1", 0, 1, "+", 1, "A").genome_in(known) == "GCA_1.5"
    assert MafRecord("GCA_1.chr1", 0, 1, "+", 1, "A").genome_in(known) == "GCA_1"


def test_unresolved_names_are_tallied_not_silently_dropped(tmp_path):
    """The tally distinguishes a name mismatch from absent coverage."""
    maf = _vgp_maf(tmp_path)
    tally = BlockTally()
    blocks = list(iter_maf_blocks(
        maf, "GCA_999999999.9", ["GCA_888888888.8"], min_block=50, tally=tally
    ))
    assert blocks == []
    assert tally.total == 3
    assert tally.unresolved_src == 3
    assert tally.kept == 0


def test_tally_separates_each_drop_reason(tmp_path):
    maf = tmp_path / "mixed.maf"
    maf.write_text(
        # kept
        "a\ns F.c1 0 60 + 100 " + "A" * 60 + "\n"
        "s S.c1 0 60 + 100 " + "A" * 60 + "\n\n"
        # focal too short
        "a\ns F.c1 60 4 + 100 ACGT\ns S.c1 60 4 + 100 ACGT\n\n"
        # comparator absent
        "a\ns F.c1 70 60 + 100 " + "A" * 60 + "\n\n"
        # focal absent
        "a\ns S.c1 80 60 + 100 " + "A" * 60 + "\n\n"
        # nothing resolves
        "a\ns Z.c1 90 60 + 100 " + "A" * 60 + "\n\n"
    )
    tally = BlockTally()
    kept = list(iter_maf_blocks(maf, "F", ["S"], min_block=50, tally=tally))
    assert len(kept) == 1
    assert tally.total == 5
    assert tally.kept == 1
    assert tally.too_short == 1
    assert tally.missing_required == 1
    assert tally.no_focal == 1
    assert tally.unresolved_src == 1
    assert "blocks read" in tally.summary()


def test_sample_src_names_reports_what_is_actually_there(tmp_path):
    """The diagnostic that makes a name mismatch obvious immediately."""
    maf = _vgp_maf(tmp_path)
    observed = sample_src_names(maf, limit=5)
    assert f"{VGP_FOCAL}.CM088821.1" in observed
    assert all("." in name for name in observed)

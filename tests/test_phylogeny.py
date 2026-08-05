"""Newick parsing and comparator selection."""

from __future__ import annotations

import pytest

from baymer2.ancestral.tree import Phylogeny, TrioSelection, parse_newick

# The evolverMammals tree, verbatim from `halStats --tree`.
EVOLVER = (
    "((simHuman_chr6:0.144018,(simMouse_chr6:0.084509,simRat_chr6:0.091589)"
    "mr:0.271974)Anc1:0.020593,(simCow_chr6:0.18908,simDog_chr6:0.16303)"
    "Anc2:0.032898)Anc0;"
)


@pytest.fixture(scope="module")
def tree() -> Phylogeny:
    return Phylogeny.from_newick(EVOLVER)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parses_the_real_hal_tree(tree):
    assert tree.leaf_names == [
        "simHuman_chr6", "simMouse_chr6", "simRat_chr6",
        "simCow_chr6", "simDog_chr6",
    ]
    assert tree.internal_names == ["Anc0", "Anc1", "mr", "Anc2"]


def test_branch_lengths_are_read(tree):
    assert tree.node("simHuman_chr6").branch_length == pytest.approx(0.144018)
    assert tree.node("mr").branch_length == pytest.approx(0.271974)


def test_parent_links_are_wired(tree):
    assert tree.node("simMouse_chr6").parent.name == "mr"
    assert tree.node("mr").parent.name == "Anc1"
    assert tree.node("Anc0").parent is None


def test_parses_minimal_forms():
    assert parse_newick("(A,B);").leaves()[0].name == "A"
    assert parse_newick("A;").name == "A"
    assert parse_newick("(A:0.1,B:0.2)R:0.3;").name == "R"


def test_handles_quoted_names():
    tree = Phylogeny.from_newick("('Homo sapiens':0.1,'Pan troglodytes':0.2)R;")
    assert "Homo sapiens" in tree.leaf_names


def test_unnamed_internal_nodes_are_allowed():
    tree = Phylogeny.from_newick("((A:0.1,B:0.1):0.2,C:0.3);")
    assert set(tree.leaf_names) == {"A", "B", "C"}


def test_rejects_malformed_newick():
    with pytest.raises(ValueError, match="unbalanced"):
        parse_newick("((A,B);")
    with pytest.raises(ValueError, match="unbalanced"):
        parse_newick("(A,B));")
    with pytest.raises(ValueError, match="empty"):
        parse_newick("   ")


def test_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        Phylogeny.from_newick("((A:0.1,A:0.1)X,B:0.2);")


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------


def test_patristic_distances_are_exact(tree):
    assert tree.distance("simHuman_chr6", "simMouse_chr6") == pytest.approx(
        0.144018 + 0.271974 + 0.084509
    )
    assert tree.distance("simHuman_chr6", "simCow_chr6") == pytest.approx(
        0.144018 + 0.020593 + 0.032898 + 0.18908
    )
    assert tree.distance("simMouse_chr6", "simRat_chr6") == pytest.approx(
        0.084509 + 0.091589
    )


def test_distance_is_symmetric_and_zero_to_self(tree):
    assert tree.distance("simCow_chr6", "simCow_chr6") == 0.0
    for a, b in [("simRat_chr6", "simDog_chr6"), ("simHuman_chr6", "simCow_chr6")]:
        assert tree.distance(a, b) == pytest.approx(tree.distance(b, a))


# --------------------------------------------------------------------------
# Comparator selection
# --------------------------------------------------------------------------


def test_selects_topological_sister_and_outgroup(tree):
    sel = tree.select("simMouse_chr6", n_outgroups=1)
    assert sel.sister == "simRat_chr6"
    assert sel.outgroups == ["simHuman_chr6"]
    assert sel.ancestor == "mr"
    assert sel.focal_branch_length == pytest.approx(0.084509)


def test_clade_is_represented_by_its_nearest_member(tree):
    """Comparator quality falls off with divergence, so pick the closest."""
    sel = tree.select("simHuman_chr6", n_outgroups=1)
    # sister clade is (mouse, rat); mouse is nearer
    assert sel.sister == "simMouse_chr6"
    assert tree.distance("simHuman_chr6", "simMouse_chr6") < tree.distance(
        "simHuman_chr6", "simRat_chr6"
    )
    # outgroup clade is (cow, dog); dog is nearer
    assert sel.outgroups == ["simDog_chr6"]


def test_targets_are_sister_then_outgroups(tree):
    sel = tree.select("simMouse_chr6", n_outgroups=2)
    assert sel.targets == [sel.sister, *sel.outgroups]
    assert len(sel.targets) == 3


def test_flags_a_sister_more_diverged_than_its_outgroup(tree):
    """Legitimate but worth surfacing: it weakens polarisation.

    Human's sister clade sits behind the long `mr` branch, so the
    topological outgroup (dog) is actually closer than the sister
    (mouse).
    """
    sel = tree.select("simHuman_chr6", n_outgroups=1)
    assert sel.distances["simMouse_chr6"] > sel.distances["simDog_chr6"]
    assert sel.sister_more_diverged_than_outgroup

    clean = tree.select("simMouse_chr6", n_outgroups=1)
    assert not clean.sister_more_diverged_than_outgroup


def test_refuses_when_the_tree_is_too_shallow(tree):
    with pytest.raises(ValueError, match="too shallow"):
        tree.select("simHuman_chr6", n_outgroups=2)


def test_focal_must_be_a_leaf(tree):
    with pytest.raises(ValueError, match="internal node"):
        tree.select("Anc1")


def test_unknown_genome_is_reported(tree):
    with pytest.raises(KeyError, match="not in this tree"):
        tree.select("Gallus_gallus")


def test_n_outgroups_must_be_positive(tree):
    with pytest.raises(ValueError, match="n_outgroups"):
        tree.select("simMouse_chr6", n_outgroups=0)


def test_select_all_skips_species_without_enough_outgroups(tree):
    every = tree.select_all(n_outgroups=1)
    assert {s.focal for s in every} == set(tree.leaf_names)
    deep = tree.select_all(n_outgroups=2)
    assert {s.focal for s in deep} == {"simMouse_chr6", "simRat_chr6"}


def test_subtree_branch_lengths_cover_every_target(tree):
    sel = tree.select("simMouse_chr6", n_outgroups=1)
    lengths = tree.subtree_branch_lengths(sel)
    assert set(lengths) == set(sel.targets)
    assert all(v > 0 for v in lengths.values())


def test_selection_row_is_serialisable(tree):
    row = tree.select("simMouse_chr6", n_outgroups=1).as_row()
    assert row["focal"] == "simMouse_chr6"
    assert row["sister"] == "simRat_chr6"
    assert row["ancestor"] == "mr"
    assert isinstance(row["sister_more_diverged_than_outgroup"], int)


def test_selection_is_a_plain_dataclass():
    sel = TrioSelection(focal="f", sister="s", outgroups=["o1", "o2"])
    assert sel.targets == ["s", "o1", "o2"]
    assert not sel.sister_more_diverged_than_outgroup


# --------------------------------------------------------------------------
# Degenerate-comparator guards
# --------------------------------------------------------------------------


def test_binomial_collapses_subspecies_but_not_congeners():
    """The distinction the VGP 577-way actually requires.

    Three *Canis lupus* assemblies must not be each other's sister; two
    distinct *Larus* species must remain usable.
    """
    from baymer2.ancestral.tree import binomial

    assert binomial("Canis_lupus_orion") == "Canis_lupus"
    assert binomial("Canis_lupus_familiaris") == "Canis_lupus"
    assert binomial("Larus_argentatus") != binomial("Larus_michahellis")


def test_binomial_resolves_assembly_nicknames():
    from baymer2.ancestral.tree import binomial

    assert binomial("hg38") == binomial("Homo_sapiens") == "Homo_sapiens"
    assert binomial("mm39") == binomial("Mus_musculus") == "Mus_musculus"
    assert binomial("GRCh38") == "Homo_sapiens"


def test_binomial_passes_through_unknown_labels():
    from baymer2.ancestral.tree import binomial

    assert binomial("Weirdname") == "Weirdname"
    assert binomial("") == ""


def _duplicate_assembly_tree() -> Phylogeny:
    """Two near-identical assemblies of one species, plus real relatives."""
    return Phylogeny.from_newick(
        "(((asmA:0.0001,asmB:0.0001)SpAnc:0.05,cousin:0.05)GenusAnc:0.1,"
        "outgroup:0.2)Root;"
    )


def test_min_divergence_skips_a_near_identical_sister():
    tree = _duplicate_assembly_tree()
    assert tree.select("asmA", 1).sister == "asmB"
    skipped = tree.select("asmA", 1, min_divergence=0.001)
    assert skipped.sister == "cousin"


def test_species_labels_skip_a_same_species_sister():
    """The criterion distance cannot express."""
    tree = _duplicate_assembly_tree()
    species = {
        "asmA": "Genus_species",
        "asmB": "Genus_species",
        "cousin": "Genus_other",
        "outgroup": "Far_away",
    }
    assert tree.select("asmA", 1, species_of=species).sister == "cousin"
    # Without labels the near-identical assembly is chosen.
    assert tree.select("asmA", 1).sister == "asmB"


def test_species_labels_keep_a_close_but_distinct_species():
    """Distance alone would wrongly exclude these."""
    tree = _duplicate_assembly_tree()
    species = {
        "asmA": "Genus_alpha",
        "asmB": "Genus_beta",  # genuinely different species
        "cousin": "Genus_other",
        "outgroup": "Far_away",
    }
    assert tree.select("asmA", 1, species_of=species).sister == "asmB"
    # A blanket distance guard would have thrown it away.
    assert tree.select("asmA", 1, min_divergence=0.001).sister == "cousin"


def test_guards_fall_through_to_the_next_clade():
    """Excluding a whole clade must skip outward, not fail."""
    tree = Phylogeny.from_newick(
        "((((asmA:0.0001,asmB:0.0001)SpAnc:0.05,cousin:0.05)GenusAnc:0.1,"
        "outgroup:0.2)Family:0.3,farout:0.5)Root;"
    )
    species = {n: "One_species" for n in ("asmA", "asmB", "cousin")}
    species["outgroup"] = "Other_species"
    species["farout"] = "Distant_species"
    selection = tree.select("asmA", 1, species_of=species)
    assert selection.sister == "outgroup"
    assert selection.outgroups == ["farout"]


def test_over_aggressive_guards_are_reported_not_silently_relaxed():
    """If nothing survives, say so rather than returning a bad comparator."""
    tree = _duplicate_assembly_tree()
    species = {n: "One_species" for n in tree.leaf_names}
    with pytest.raises(ValueError, match="too shallow"):
        tree.select("asmA", 1, species_of=species)


def test_negative_min_divergence_rejected(tree):
    with pytest.raises(ValueError, match="min_divergence"):
        tree.select("simMouse_chr6", 1, min_divergence=-0.1)


def test_select_all_accepts_the_guards():
    tree = _duplicate_assembly_tree()
    rows = tree.select_all(1, min_divergence=0.001)
    assert {r.focal for r in rows} <= set(tree.leaf_names)
    assert all(r.distances[r.sister] >= 0.001 for r in rows)

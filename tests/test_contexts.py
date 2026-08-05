"""Validate the context tree against legacy Baymer's own output.

``index_dict.layer_N.json`` is written by legacy ``run_baymer.py`` and
records the exact array-index-to-context mapping the sampler used, so it
pins down enumeration order, alternation, and folding simultaneously.
``_legacy_leaf_contexts`` is a transcription of
``run_baymer.get_leaf_contexts`` kept deliberately independent of the
new implementation.
"""

from __future__ import annotations

from itertools import product

import pytest

from baymer2.contexts import (
    BASES,
    ContextTree,
    alt_bases,
    build_index_map,
    reverse_complement,
)

from conftest import GOLDEN_ALTERNATION, GOLDEN_MAX_MER


def _legacy_leaf_contexts(context: str, max_mer: int, oppo_asymmetry: bool) -> list[str]:
    """Verbatim transcription of legacy ``run_baymer.get_leaf_contexts``."""
    leaf_contexts = []
    context_length = len(context)
    odd_max_bool = max_mer % 2
    nucleotides_to_add = int(max_mer - context_length)
    odd_bool = 0
    if odd_max_bool and nucleotides_to_add % 2 and not oppo_asymmetry:
        odd_bool = 1
    elif not odd_max_bool and nucleotides_to_add % 2 and oppo_asymmetry:
        odd_bool = 1
    left_flank_length = int(nucleotides_to_add / 2) + odd_bool

    for combination in product("ACGT", repeat=nucleotides_to_add):
        bases = "".join(combination)
        if left_flank_length == 0:
            leaf_context = context + bases
        else:
            leaf_context = bases[0:left_flank_length] + context + bases[left_flank_length:]
        leaf_contexts.append(leaf_context)
    return leaf_contexts


def _legacy_focal_index(k: int, oppo_asymmetry: bool) -> int:
    """Verbatim transcription of the central-nucleotide logic."""
    odd_bool = k % 2
    central_nuc_index = int(k / 2)
    if not odd_bool and not oppo_asymmetry:
        central_nuc_index -= 1
    return central_nuc_index


# --------------------------------------------------------------------------
# Enumeration order, against real legacy output
# --------------------------------------------------------------------------


def test_contexts_match_legacy_index_dicts(legacy_index_dicts):
    tree = ContextTree(max_mer=GOLDEN_MAX_MER, alternation=GOLDEN_ALTERNATION)
    for layer, expected in legacy_index_dicts.items():
        assert list(tree.contexts(layer)) == expected, f"layer {layer} order differs"


def test_index_dicts_cover_every_layer(legacy_index_dicts):
    assert sorted(legacy_index_dicts) == list(range(GOLDEN_MAX_MER))


def test_layer_sizes(legacy_index_dicts):
    for layer, expected in legacy_index_dicts.items():
        assert ContextTree(max_mer=5).layer_size(layer) == len(expected)


# --------------------------------------------------------------------------
# Tree structure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alternation", ["right", "left"])
@pytest.mark.parametrize("max_mer", [1, 2, 3, 5, 6])
def test_children_are_contiguous_heap_slice(max_mer, alternation):
    """Children of node i must be exactly [4i, 4i+4) in the next layer."""
    tree = ContextTree(max_mer=max_mer, alternation=alternation)
    for layer in range(max_mer - 1):
        parents = tree.contexts(layer)
        children = tree.contexts(layer + 1)
        for i, parent in enumerate(parents):
            assert list(children[4 * i : 4 * i + 4]) == tree.children(parent)


@pytest.mark.parametrize("alternation", ["right", "left"])
@pytest.mark.parametrize("max_mer", [1, 3, 4, 5])
def test_every_context_is_folded(max_mer, alternation):
    """Focal base is always A or C, at every depth."""
    tree = ContextTree(max_mer=max_mer, alternation=alternation)
    for layer in range(max_mer):
        for ctx in tree.contexts(layer):
            assert tree.focal_base(ctx) in ("A", "C"), (layer, ctx)


@pytest.mark.parametrize("alternation", ["right", "left"])
@pytest.mark.parametrize("max_mer", [1, 2, 3, 4, 5, 6, 7])
def test_contexts_are_unique_and_complete(max_mer, alternation):
    tree = ContextTree(max_mer=max_mer, alternation=alternation)
    for layer in range(max_mer):
        ctxs = tree.contexts(layer)
        assert len(set(ctxs)) == len(ctxs) == tree.layer_size(layer)
        assert {len(c) for c in ctxs} == {layer + 1}


# --------------------------------------------------------------------------
# Focal index and leaf expansion, against transcribed legacy logic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alternation", ["right", "left"])
@pytest.mark.parametrize("max_mer", [1, 2, 3, 4, 5, 6, 7, 9])
def test_focal_index_matches_legacy(max_mer, alternation):
    tree = ContextTree(max_mer=max_mer, alternation=alternation)
    oppo = alternation == "left"
    for k in range(1, max_mer + 1):
        assert tree.focal_index(k) == _legacy_focal_index(k, oppo)


def test_focal_index_documented_schedule():
    """The table in the module docstring, pinned."""
    tree = ContextTree(max_mer=5, alternation="right")
    assert [tree.focal_index(k) for k in range(1, 6)] == [0, 0, 1, 1, 2]


@pytest.mark.parametrize("alternation", ["right", "left"])
@pytest.mark.parametrize("max_mer", [3, 4, 5, 6])
def test_leaf_contexts_match_legacy(max_mer, alternation):
    tree = ContextTree(max_mer=max_mer, alternation=alternation)
    oppo = alternation == "left"
    for layer in range(max_mer):
        for ctx in tree.contexts(layer):
            assert tree.leaf_contexts(ctx) == _legacy_leaf_contexts(ctx, max_mer, oppo)


@pytest.mark.parametrize("alternation", ["right", "left"])
@pytest.mark.parametrize("max_mer", [3, 4, 5])
def test_leaves_partition_the_deepest_layer(max_mer, alternation):
    """Each layer's leaf sets must tile the deepest layer exactly once."""
    tree = ContextTree(max_mer=max_mer, alternation=alternation)
    deepest = set(tree.contexts(max_mer - 1))
    for layer in range(max_mer):
        seen: list[str] = []
        for ctx in tree.contexts(layer):
            seen.extend(tree.leaf_contexts(ctx))
        assert len(seen) == len(deepest)
        assert set(seen) == deepest


@pytest.mark.parametrize("alternation", ["right", "left"])
@pytest.mark.parametrize("max_mer", [3, 4, 5])
def test_leaf_contexts_of_a_node_are_its_children_leaves(max_mer, alternation):
    """Leaf sets must nest, which is what makes the upward reduction valid.

    Note this is a *multiset* identity, not a sequence identity: legacy
    ``get_leaf_contexts`` lays flanking bases out per-context, so a
    parent's leaf order is not the concatenation of its children's. Only
    order-independent reductions (i.e. sums, which is all the likelihood
    ever takes) may be carried across layers.
    """
    tree = ContextTree(max_mer=max_mer, alternation=alternation)
    for layer in range(max_mer - 1):
        for parent in tree.contexts(layer):
            merged: list[str] = []
            for child in tree.children(parent):
                merged.extend(tree.leaf_contexts(child))
            assert sorted(merged) == sorted(tree.leaf_contexts(parent))


@pytest.mark.parametrize("alternation", ["right", "left"])
@pytest.mark.parametrize("max_mer", [4, 5])
def test_leaf_order_differs_from_child_concatenation(max_mer, alternation):
    """Pin the ordering caveat above, so a regression cannot pass silently."""
    tree = ContextTree(max_mer=max_mer, alternation=alternation)
    parent = tree.contexts(0)[0]
    merged = [leaf for child in tree.children(parent) for leaf in tree.leaf_contexts(child)]
    assert merged != tree.leaf_contexts(parent)
    assert sorted(merged) == sorted(tree.leaf_contexts(parent))


# --------------------------------------------------------------------------
# Folding helpers
# --------------------------------------------------------------------------


def test_reverse_complement_roundtrip():
    for combo in product(BASES, repeat=4):
        s = "".join(combo)
        assert reverse_complement(reverse_complement(s)) == s


def test_reverse_complement_known_values():
    assert reverse_complement("ACGT") == "ACGT"
    assert reverse_complement("AACGT") == "ACGTT"
    assert reverse_complement("CG") == "CG"


def test_alt_bases_order():
    assert alt_bases("A") == ("C", "G", "T")
    assert alt_bases("C") == ("A", "G", "T")
    assert alt_bases("G") == ("A", "C", "T")


@pytest.mark.parametrize("max_mer", [3, 5])
def test_fold_maps_every_kmer_into_the_folded_set(max_mer):
    tree = ContextTree(max_mer=max_mer)
    folded = set(tree.contexts(max_mer - 1))
    for combo in product(BASES, repeat=max_mer):
        ctx = "".join(combo)
        assert tree.fold(ctx) in folded


@pytest.mark.parametrize("max_mer", [3, 5])
def test_fold_is_identity_on_folded_contexts(max_mer):
    tree = ContextTree(max_mer=max_mer)
    for ctx in tree.contexts(max_mer - 1):
        assert tree.fold(ctx) == ctx


def test_fold_substitution_complements_the_alt():
    tree = ContextTree(max_mer=3)
    # TGA is G-focal, so it folds; the alt must complement with it.
    ctx, alt = tree.fold_substitution("TGA", "A")
    assert ctx == "TCA"
    assert alt == "T"
    # An already-folded context passes through untouched.
    assert tree.fold_substitution("TCA", "T") == ("TCA", "T")


def test_build_index_map_round_trips():
    tree = ContextTree(max_mer=4)
    for layer in range(4):
        imap = build_index_map(tree, layer)
        ctxs = tree.contexts(layer)
        assert len(imap) == len(ctxs)
        assert all(ctxs[i] == c for c, i in imap.items())


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_rejects_bad_alternation():
    with pytest.raises(ValueError, match="alternation"):
        ContextTree(max_mer=5, alternation="sideways")


def test_rejects_bad_max_mer():
    with pytest.raises(ValueError, match="max_mer"):
        ContextTree(max_mer=0)


def test_rejects_out_of_range_layer():
    tree = ContextTree(max_mer=3)
    with pytest.raises(ValueError, match="layer"):
        tree.contexts(3)


def test_rejects_overlong_context_for_leaves():
    tree = ContextTree(max_mer=3)
    with pytest.raises(ValueError, match="longer than max_mer"):
        tree.leaf_contexts("AAAAA")


# --------------------------------------------------------------------------
# Centred growth (substitution mode)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("max_mer", [1, 3, 5, 7])
def test_centred_tree_geometry(max_mer):
    tree = ContextTree(max_mer=max_mer, growth="centred")
    assert tree.branching == 16
    assert tree.n_layers == (max_mer + 1) // 2
    for layer in range(tree.n_layers):
        k = tree.layer_k(layer)
        assert k == 2 * layer + 1
        ctxs = tree.contexts(layer)
        assert len(ctxs) == tree.layer_size(layer) == 2 * 16**layer
        assert {len(c) for c in ctxs} == {k}
        assert tree.focal_index(k) == layer


def test_centred_tree_reaches_the_same_leaf_set_as_alternating():
    """Same 9-mer leaves, 5 layers instead of 9 -- the whole point."""
    centred = ContextTree(max_mer=9, growth="centred")
    alternating = ContextTree(max_mer=9)
    assert centred.n_leaves == alternating.n_leaves == 2 * 4**8
    assert set(centred.contexts(centred.n_layers - 1)) == set(
        alternating.contexts(alternating.n_layers - 1)
    )
    assert centred.n_layers == 5
    assert alternating.n_layers == 9


@pytest.mark.parametrize("max_mer", [3, 5])
def test_centred_children_are_contiguous_heap_slice(max_mer):
    tree = ContextTree(max_mer=max_mer, growth="centred")
    for layer in range(tree.n_layers - 1):
        parents = tree.contexts(layer)
        children = tree.contexts(layer + 1)
        for i, parent in enumerate(parents):
            assert list(children[16 * i : 16 * i + 16]) == tree.children(parent)


@pytest.mark.parametrize("max_mer", [3, 5])
def test_centred_leaves_partition_the_deepest_layer(max_mer):
    tree = ContextTree(max_mer=max_mer, growth="centred")
    deepest = set(tree.contexts(tree.n_layers - 1))
    for layer in range(tree.n_layers):
        seen = [leaf for c in tree.contexts(layer) for leaf in tree.leaf_contexts(c)]
        assert len(seen) == len(deepest)
        assert set(seen) == deepest


@pytest.mark.parametrize("max_mer", [3, 5])
def test_centred_leaf_sets_nest(max_mer):
    tree = ContextTree(max_mer=max_mer, growth="centred")
    for layer in range(tree.n_layers - 1):
        for parent in tree.contexts(layer):
            merged = [
                leaf for child in tree.children(parent)
                for leaf in tree.leaf_contexts(child)
            ]
            assert sorted(merged) == sorted(tree.leaf_contexts(parent))


@pytest.mark.parametrize("max_mer", [1, 3, 5, 7, 9])
def test_centred_tree_is_reverse_complement_closed(max_mer):
    """The property the CTMC likelihood requires, at every layer."""
    tree = ContextTree(max_mer=max_mer, growth="centred")
    assert tree.reverse_complement_closed
    for layer in range(tree.n_layers):
        k = tree.layer_k(layer)
        fi = tree.focal_index(k)
        members = set(tree.contexts(layer))
        for ctx in members:
            left, right = ctx[:fi], ctx[fi + 1 :]
            for base in BASES:
                sub = left + base + right
                folded = sub if base in ("A", "C") else reverse_complement(sub)
                assert folded in members, (ctx, base)


def test_alternating_tree_is_not_reverse_complement_closed():
    assert not ContextTree(max_mer=5).reverse_complement_closed
    assert not ContextTree(max_mer=9).reverse_complement_closed


def test_centred_growth_requires_odd_max_mer():
    with pytest.raises(ValueError, match="max_mer must be odd"):
        ContextTree(max_mer=4, growth="centred")


def test_unknown_growth_rejected():
    with pytest.raises(ValueError, match="growth must be one of"):
        ContextTree(max_mer=5, growth="spiral")

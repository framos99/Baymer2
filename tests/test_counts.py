"""Validate sufficient statistics against legacy Baymer.

Two independent oracles are used:

1. **Legacy leaf enumeration.** ``_legacy_leaf_stats`` rebuilds ``S`` the
   way ``run_baymer.initialize_layer_tree_edge_list`` did -- expand each
   node to its k-mers, adjust ``C`` contexts by a 0<->1 swap, flatten,
   and take strided sums. The fast reduction must agree exactly.

2. **Legacy down-tree marginalisation.** ``generate_count_json.py``
   builds the 1..k-1 count dicts from the k-mer table by an entirely
   separate code path (``extrapolate_context_counts_down_tree``). Those
   shipped JSONs must equal the reduction of the 5-mer table. This is the
   strongest available check that both the tree geometry and the count
   pooling are right, because nothing in baymer2 was involved in
   producing them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from baymer2.contexts import ContextTree
from baymer2.counts import (
    CountTable,
    adjust_to_reference_first,
    infer_alternation,
)

from conftest import GOLDEN_ALTERNATION, GOLDEN_MAX_MER


def _legacy_adjust(counts, ref: str) -> list[float]:
    """Legacy ``get_adjusted_leaf_counts``: swap slots 0/1 for C contexts."""
    counts = list(counts[:4])
    if ref == "C":
        return [counts[1], counts[0], counts[2], counts[3]]
    return counts


def _legacy_leaf_stats(count_dict, tree: ContextTree, layer: int) -> np.ndarray:
    """Rebuild S[layer] by legacy leaf enumeration + strided summation."""
    focal_idx = tree.focal_index(tree.max_mer)
    out = np.zeros((tree.layer_size(layer), 4))
    for i, ctx in enumerate(tree.contexts(layer)):
        leaves = tree.leaf_contexts(ctx)
        flat = np.concatenate(
            [
                np.asarray(_legacy_adjust(count_dict[leaf], leaf[focal_idx]), dtype=float)
                for leaf in leaves
            ]
        )
        # legacy likelihood took leaf_counts[a::4]
        out[i] = [flat[a::4].sum() for a in range(4)]
    return out


@pytest.fixture(scope="module")
def tree() -> ContextTree:
    return ContextTree(max_mer=GOLDEN_MAX_MER, alternation=GOLDEN_ALTERNATION)


@pytest.fixture(scope="module")
def table(golden_count_dicts, tree) -> CountTable:
    return CountTable.from_count_dict(golden_count_dicts[GOLDEN_MAX_MER], tree, name="AFR")


# --------------------------------------------------------------------------
# Oracle 1: legacy leaf enumeration
# --------------------------------------------------------------------------


def test_stats_match_legacy_leaf_enumeration(table, golden_count_dicts, tree):
    for layer in range(tree.n_layers):
        expected = _legacy_leaf_stats(golden_count_dicts[GOLDEN_MAX_MER], tree, layer)
        np.testing.assert_array_equal(table.layer_stats(layer), expected)


def test_reference_adjustment_matches_legacy_swap(golden_count_dicts, tree):
    focal_idx = tree.focal_index(GOLDEN_MAX_MER)
    for ctx, entry in golden_count_dicts[GOLDEN_MAX_MER].items():
        ref = ctx[focal_idx]
        assert adjust_to_reference_first(entry[:4], ref) == _legacy_adjust(entry, ref)


# --------------------------------------------------------------------------
# Oracle 2: legacy down-tree marginalisation (the shipped 1..4mer JSONs)
# --------------------------------------------------------------------------


def test_stats_match_legacy_lower_k_count_dicts(table, golden_count_dicts, tree):
    """Reduction of the 5-mer table must equal legacy's own 1..4-mer dicts."""
    for k, cd in sorted(golden_count_dicts.items()):
        layer = k - 1
        focal_idx = tree.focal_index(k)
        got = table.layer_stats(layer)
        for i, ctx in enumerate(tree.contexts(layer)):
            expected = adjust_to_reference_first(cd[ctx][:4], ctx[focal_idx])
            np.testing.assert_array_equal(
                got[i], expected, err_msg=f"k={k} context={ctx}"
            )


def test_lower_k_dicts_declare_the_focal_index_we_compute(golden_count_dicts, tree):
    """Legacy stores ref_context_index per entry; it must match the tree."""
    for k, cd in golden_count_dicts.items():
        for ctx, entry in cd.items():
            assert int(entry[6]) == tree.focal_index(k), (k, ctx)


def test_totals_are_conserved_up_the_tree(table, tree):
    """Every layer must account for exactly the same number of sites."""
    totals = [table.layer_stats(l).sum() for l in range(tree.n_layers)]
    assert len(set(np.round(totals, 6))) == 1
    assert totals[0] == pytest.approx(table.total_sites)


def test_parent_equals_sum_of_children(table, tree):
    for layer in range(tree.n_layers - 1):
        parent = table.layer_stats(layer)
        child = table.layer_stats(layer + 1)
        np.testing.assert_array_equal(parent, child.reshape(-1, 4, 4).sum(axis=1))


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------


def test_zero_mask_flags_only_empty_nodes(table, tree):
    for layer in range(tree.n_layers):
        mask = table.zero_mask(layer)
        totals = table.layer_stats(layer).sum(axis=1)
        np.testing.assert_array_equal(mask, totals == 0)


def test_empirical_rates_are_probabilities(table, tree):
    for layer in range(tree.n_layers):
        rates = table.empirical_rates(layer)
        assert rates.shape == (tree.layer_size(layer), 3)
        assert np.all(rates >= 0)
        assert np.all(rates.sum(axis=1) <= 1.0 + 1e-12)


def test_empirical_rates_zero_for_empty_nodes():
    tree = ContextTree(max_mer=2)
    counts = np.zeros((tree.n_leaves, 4))
    counts[0] = [10, 1, 2, 3]
    table = CountTable(tree=tree, leaf_counts=counts)
    rates = table.empirical_rates(1)
    assert rates[0] == pytest.approx([1 / 16, 2 / 16, 3 / 16])
    assert np.all(rates[1:] == 0)


def test_counts_for_context_round_trips(table, tree):
    for layer in range(tree.n_layers):
        for i, ctx in enumerate(tree.contexts(layer)):
            np.testing.assert_array_equal(
                table.counts_for(ctx), table.layer_stats(layer)[i]
            )


def test_stats_are_cached(table):
    assert table.stats is table.stats


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------


def test_from_json_matches_from_count_dict(tmp_path, golden_count_dicts, tree):
    cd = golden_count_dicts[GOLDEN_MAX_MER]
    path = tmp_path / "counts.json"
    path.write_text(json.dumps(cd))
    np.testing.assert_array_equal(
        CountTable.from_json(path, tree).leaf_counts,
        CountTable.from_count_dict(cd, tree).leaf_counts,
    )


def test_missing_contexts_are_rejected(golden_count_dicts, tree):
    cd = dict(golden_count_dicts[GOLDEN_MAX_MER])
    cd.pop("AAAAA")
    with pytest.raises(KeyError, match="missing 1 of 512"):
        CountTable.from_count_dict(cd, tree)


def test_inconsistent_total_is_rejected(golden_count_dicts, tree):
    cd = {k: list(v) for k, v in golden_count_dicts[GOLDEN_MAX_MER].items()}
    cd["AAAAA"][4] += 7
    with pytest.raises(ValueError, match="inconsistent"):
        CountTable.from_count_dict(cd, tree)


def test_alternation_is_ambiguous_from_an_odd_deepest_layer(golden_count_dicts):
    """At odd max_mer the deepest dict alone cannot pin the pattern.

    focal_index(5) is 2 under both schedules, so a lone 5-mer dict is
    genuinely consistent with either. Documented rather than hidden: the
    single-dict loader therefore cannot reject a wrong ``alternation``,
    and callers who care must load the whole family.
    """
    assert infer_alternation({5: golden_count_dicts[5]}) == {"right", "left"}
    left_tree = ContextTree(max_mer=GOLDEN_MAX_MER, alternation="left")
    CountTable.from_count_dict(golden_count_dicts[GOLDEN_MAX_MER], left_tree)


def test_alternation_is_pinned_by_even_k_dicts(golden_count_dicts):
    """Even-k layers do distinguish the two schedules."""
    assert infer_alternation({2: golden_count_dicts[2]}) == {"right"}
    assert infer_alternation({4: golden_count_dicts[4]}) == {"right"}
    assert infer_alternation(golden_count_dicts) == {"right"}


def test_wrong_alternation_is_rejected_by_the_full_family(golden_count_dicts):
    left_tree = ContextTree(max_mer=GOLDEN_MAX_MER, alternation="left")
    with pytest.raises(ValueError, match="inconsistent with alternation"):
        CountTable.from_count_dict_set(golden_count_dicts, left_tree)


def test_from_count_dict_set_accepts_the_real_family(golden_count_dicts, tree, table):
    """The 1..4mer dicts must reconcile with the reduction of the 5mer."""
    built = CountTable.from_count_dict_set(golden_count_dicts, tree, name="AFR")
    np.testing.assert_array_equal(built.leaf_counts, table.leaf_counts)


def test_from_count_dict_set_catches_a_corrupted_lower_k_dict(golden_count_dicts, tree):
    cd = {k: {c: list(v) for c, v in d.items()} for k, d in golden_count_dicts.items()}
    cd[3]["AAA"][1] += 100
    with pytest.raises(ValueError, match="disagrees with the reduction"):
        CountTable.from_count_dict_set(cd, tree)


def test_negative_counts_rejected(tree):
    counts = np.zeros((tree.n_leaves, 4))
    counts[0, 1] = -1
    with pytest.raises(ValueError, match="negative"):
        CountTable(tree=tree, leaf_counts=counts)


def test_wrong_shape_rejected(tree):
    with pytest.raises(ValueError, match="shape"):
        CountTable(tree=tree, leaf_counts=np.zeros((10, 4)))


def test_fractional_counts_are_allowed(tree):
    """Soft ancestral-state weights produce non-integer counts."""
    counts = np.full((tree.n_leaves, 4), 0.25)
    table = CountTable(tree=tree, leaf_counts=counts)
    assert table.total_sites == pytest.approx(tree.n_leaves)
    assert table.layer_stats(0).sum() == pytest.approx(tree.n_leaves)

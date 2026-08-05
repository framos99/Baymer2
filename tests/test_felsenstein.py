"""Maximum-likelihood ancestral reconstruction.

The properties that matter: pruning must agree with parsimony where
parsimony is confident, must stay usable where parsimony gives up, and
must weight comparators by branch length rather than treating them
alike.
"""

from __future__ import annotations

import numpy as np
import pytest

from baymer2.ancestral.felsenstein import (
    LadderTopology,
    MLAlignmentCounter,
    SubstitutionModel,
    ancestral_posteriors,
)
from baymer2.ancestral.parsimony import AlignmentCounter, fitch_ancestral
from baymer2.ancestral.tree import Phylogeny
from baymer2.contexts import BASES, ContextTree
from baymer2.io.hal import MafRecord

EVOLVER = (
    "((simHuman_chr6:0.144018,(simMouse_chr6:0.084509,simRat_chr6:0.091589)"
    "mr:0.271974)Anc1:0.020593,(simCow_chr6:0.18908,simDog_chr6:0.16303)"
    "Anc2:0.032898)Anc0;"
)


@pytest.fixture(scope="module")
def phylo() -> Phylogeny:
    return Phylogeny.from_newick(EVOLVER)


@pytest.fixture(scope="module")
def selection(phylo):
    return phylo.select("simHuman_chr6", n_outgroups=1)


@pytest.fixture(scope="module")
def topology(phylo, selection) -> LadderTopology:
    return LadderTopology.build(phylo, selection)


# --------------------------------------------------------------------------
# Substitution model
# --------------------------------------------------------------------------


def test_jukes_cantor_is_a_valid_generator():
    model = SubstitutionModel.jukes_cantor()
    assert np.allclose(model.rate_matrix.sum(axis=1), 0.0)
    assert np.allclose(model.stationary, 0.25)


@pytest.mark.parametrize("t", [0.0, 1e-4, 0.1, 1.0, 10.0])
def test_transition_matrices_are_stochastic(t):
    p = SubstitutionModel.jukes_cantor().transition(t)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-12)
    assert np.all(p >= 0.0)


def test_zero_branch_is_the_identity():
    np.testing.assert_allclose(
        SubstitutionModel.jukes_cantor().transition(0.0), np.eye(4), atol=1e-14
    )


def test_long_branch_approaches_the_stationary_distribution():
    p = SubstitutionModel.jukes_cantor().transition(100.0)
    np.testing.assert_allclose(p, 0.25, atol=1e-6)


def test_from_rates_recovers_the_stationary_distribution():
    # A chain that is four times more likely to enter A than anything else.
    rates = np.array([
        [0.0, 0.1, 0.1, 0.1],
        [0.4, 0.0, 0.1, 0.1],
        [0.4, 0.1, 0.0, 0.1],
        [0.4, 0.1, 0.1, 0.0],
    ])
    model = SubstitutionModel.from_rates(rates)
    assert model.stationary.sum() == pytest.approx(1.0)
    assert model.stationary[0] > model.stationary[1:].max()
    # A long branch must land on exactly that distribution.
    np.testing.assert_allclose(
        model.transition(200.0)[0], model.stationary, atol=1e-6
    )


def test_invalid_models_are_rejected():
    with pytest.raises(ValueError, match="rows must sum to zero"):
        SubstitutionModel(np.ones((4, 4)), np.full(4, 0.25))
    with pytest.raises(ValueError, match="stationary must sum to one"):
        SubstitutionModel(
            SubstitutionModel.jukes_cantor().rate_matrix, np.ones(4)
        )
    with pytest.raises(ValueError, match="non-negative"):
        SubstitutionModel.jukes_cantor().transition(-1.0)


# --------------------------------------------------------------------------
# Ladder geometry
# --------------------------------------------------------------------------


def test_ladder_lengths_reconstruct_patristic_distances(phylo, selection):
    lengths = phylo.ladder(selection)
    assert lengths["focal_length"] + lengths["sister_length"] == pytest.approx(
        phylo.distance(selection.focal, selection.sister)
    )
    assert (
        lengths["focal_length"]
        + lengths["internal_lengths"][0]
        + lengths["outgroup_lengths"][0]
    ) == pytest.approx(phylo.distance(selection.focal, selection.outgroups[0]))
    assert lengths["ancestor"] == "Anc1"


def test_ladder_refuses_a_shallow_tree(phylo):
    from baymer2.ancestral.tree import TrioSelection

    bogus = TrioSelection(
        focal="simHuman_chr6", sister="simMouse_chr6", outgroups=["a", "b", "c"]
    )
    with pytest.raises(ValueError, match="too shallow"):
        phylo.ladder(bogus)


# --------------------------------------------------------------------------
# Posteriors
# --------------------------------------------------------------------------


def test_posteriors_are_normalised(topology):
    for f in BASES:
        for s in BASES:
            for o in BASES:
                post = ancestral_posteriors(f, s, [o], topology)
                assert post is not None
                assert post.sum() == pytest.approx(1.0)
                assert np.all(post >= 0.0)


def test_full_agreement_is_nearly_certain(topology):
    post = ancestral_posteriors("A", "A", ["A"], topology)
    assert "ACGT"[int(np.argmax(post))] == "A"
    assert post.max() > 0.99


def test_agrees_with_parsimony_wherever_parsimony_resolves(topology):
    """The ML mode must not contradict confident parsimony calls."""
    disagreements = []
    for f in BASES:
        for s in BASES:
            for o in BASES:
                called = fitch_ancestral(f, s, [o])
                if called is None:
                    continue
                post = ancestral_posteriors(f, s, [o], topology)
                if "ACGT"[int(np.argmax(post))] != called:
                    disagreements.append((f, s, o, called))
    assert not disagreements, disagreements


def test_stays_informative_where_parsimony_gives_up(topology):
    """The point of the ML mode: an ambiguous site still carries signal."""
    assert fitch_ancestral("A", "C", ["G"]) is None
    post = ancestral_posteriors("A", "C", ["G"], topology)
    assert post.max() > 0.3
    assert post.max() < 0.9  # genuinely uncertain, not silently confident


def test_comparators_are_weighted_by_branch_length(topology):
    """A difference on a long branch is weaker evidence than on a short one.

    Human's sister sits behind the long `mr` branch (0.36) while the
    outgroup is nearer (0.20), so a sister mismatch should dent
    confidence less than an outgroup mismatch. Parsimony treats them
    identically.
    """
    sister_differs = ancestral_posteriors("A", "C", ["A"], topology).max()
    outgroup_differs = ancestral_posteriors("A", "A", ["C"], topology).max()
    assert sister_differs > outgroup_differs


def test_missing_observations_return_none(topology):
    assert ancestral_posteriors("-", "A", ["A"], topology) is None
    assert ancestral_posteriors("A", "N", ["A"], topology) is None


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------


def _block(focal: str, sister: str, outgroup: str) -> dict:
    return {
        "simHuman_chr6": MafRecord("F.c1", 0, len(focal), "+", 1000, focal),
        "simMouse_chr6": MafRecord("S.c1", 0, len(sister), "+", 1000, sister),
        "simDog_chr6": MafRecord("O.c1", 0, len(outgroup), "+", 1000, outgroup),
    }


def _ml_counter(phylo, selection, **kw) -> MLAlignmentCounter:
    return MLAlignmentCounter.from_selection(
        ContextTree(max_mer=3, growth="centred"), phylo, selection, **kw
    )


def test_soft_weights_sum_to_the_site_count(phylo, selection):
    counter = _ml_counter(phylo, selection, flank_threshold=1e-6, min_weight=0.0)
    counter.add_block(_block("AAAAA", "AAAAA", "AAAAA"))
    # 5 columns, k=3 -> 3 countable centres, each contributing weight 1
    assert counter.counts.sum() == pytest.approx(3.0)


def test_an_ambiguous_site_is_split_across_contexts(phylo, selection):
    """Parsimony would discard this site; ML spreads it."""
    counter = _ml_counter(phylo, selection, flank_threshold=1e-6, min_weight=0.0)
    counter.add_block(_block("AAA", "ACA", "AGA"))
    assert counter.counts.sum() == pytest.approx(1.0)
    assert (counter.counts > 0).sum() > 1, "weight should reach several contexts"


def test_flank_threshold_drops_low_confidence_neighbourhoods(phylo, selection):
    strict = _ml_counter(phylo, selection, flank_threshold=0.999)
    loose = _ml_counter(phylo, selection, flank_threshold=1e-6)
    block = _block("AAAAA", "ACAAA", "AGAAA")
    strict.add_block(block)
    loose.add_block(block)
    assert loose.counts.sum() > strict.counts.sum()
    assert strict.n_dropped_flank[0] > 0


def test_min_weight_prunes_negligible_states(phylo, selection):
    everything = _ml_counter(phylo, selection, flank_threshold=1e-6, min_weight=0.0)
    pruned = _ml_counter(phylo, selection, flank_threshold=1e-6, min_weight=0.2)
    block = _block("AAA", "ACA", "AGA")
    everything.add_block(block)
    pruned.add_block(block)
    assert (pruned.counts > 0).sum() < (everything.counts > 0).sum()
    assert pruned.counts.sum() <= everything.counts.sum()


def test_ml_counter_validates_its_arguments(phylo, selection):
    with pytest.raises(ValueError, match="max_mer must be odd"):
        MLAlignmentCounter.from_selection(
            ContextTree(max_mer=4), phylo, selection
        )
    with pytest.raises(ValueError, match="flank_threshold"):
        _ml_counter(phylo, selection, flank_threshold=0.0)


def test_ml_recovers_hypermutable_contexts_parsimony_discards(phylo, selection):
    """The ascertainment bias this mode exists to remove.

    A context is only counted by parsimony when every base in it
    resolves unambiguously -- and ambiguity is caused by mutation. So
    parsimony preferentially deletes the very contexts that mutate
    fastest, shrinking their denominator and flattening the estimated
    spectrum. Soft weights keep them.
    """
    tree = ContextTree(max_mer=3, growth="centred")
    # A CG whose C is ambiguous across the trio, repeated so it is not noise.
    blocks = [_block("ACGTA", "AGGTA", "ATGTA") for _ in range(20)]

    hard = AlignmentCounter(tree, selection.focal, selection.sister,
                            selection.outgroups)
    for b in blocks:
        hard.add_block(b)
    soft = _ml_counter(phylo, selection, flank_threshold=1e-6, min_weight=0.0)
    for b in blocks:
        soft.add_block(b)

    cg = [i for i, c in enumerate(tree.contexts(1)) if c[1:] == "CG"]
    assert hard.counts[cg].sum() == 0, "parsimony should discard these"
    assert soft.counts[cg].sum() > 0, "ML should retain them"

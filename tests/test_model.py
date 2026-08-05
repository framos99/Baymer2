"""Whole-tree fits, including agreement with legacy Baymer on real data.

The legacy-agreement test uses a shorter chain than the published
tutorial run so it stays inside a test suite's time budget; the full
40000-iteration comparison lives in ``validation/compare_to_legacy.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from baymer2.backends.numba_mh import LayerConfig
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.model import ModelConfig, fit

from conftest import (GOLDEN_ALTERNATION, GOLDEN_MAX_MER, LEGACY_ROOT_ENV,
                      LEGACY_TUTORIAL)

LEGACY_C = 0.04472136
LEGACY_SET_SIGMA = 0.8540486

LEGACY_RATE_MATRIX = (
    LEGACY_TUTORIAL
    / "tutorial_outs/tutorial_baymer_out/AFR"
    / "AFR_cpg_islands_ALL_rs14_rate_matrix.burned_in.thinned.layer_{layer}.npy"
)


@pytest.fixture(scope="module")
def table(golden_count_dicts) -> CountTable:
    tree = ContextTree(max_mer=GOLDEN_MAX_MER, alternation=GOLDEN_ALTERNATION)
    return CountTable.from_count_dict_set(golden_count_dicts, tree, name="AFR")


def _config(seed: int, iterations: int = 6000) -> ModelConfig:
    return ModelConfig(
        max_mer=GOLDEN_MAX_MER,
        seed=seed,
        default_layer=LayerConfig(
            n_iterations=iterations,
            burnin=iterations // 2,
            thin=5,
            set_sigma=LEGACY_SET_SIGMA,
            c=LEGACY_C,
            phi_proposal_sd=0.05,
            alpha_init=0.8,
        ),
    )


def _median_relative(a: np.ndarray, b: np.ndarray) -> float:
    scale = np.maximum((np.abs(a) + np.abs(b)) / 2.0, 1e-12)
    return float(np.median(np.abs(a - b) / scale))


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_fit_produces_one_result_per_layer(table):
    result = fit(table, _config(seed=1, iterations=400))
    assert len(result.layers) == GOLDEN_MAX_MER
    for layer, r in enumerate(result.layers):
        assert r.layer == layer
        assert r.rates.shape[1] == table.tree.layer_size(layer)
        assert r.rates.shape[2] == 3


def test_rate_dict_rows_are_probability_vectors(table):
    result = fit(table, _config(seed=1, iterations=400))
    rd = result.rate_dict(2)
    assert set(rd) == set(table.tree.contexts(2))
    for ctx, probs in rd.items():
        assert len(probs) == 4
        assert all(p >= 0 for p in probs), ctx
        assert sum(probs) == pytest.approx(1.0)


def test_summary_mentions_every_layer(table):
    result = fit(table, _config(seed=1, iterations=400))
    text = result.summary()
    for layer in range(GOLDEN_MAX_MER):
        assert f"layer {layer}" in text


def test_mismatched_tree_is_rejected(table):
    with pytest.raises(ValueError, match="max_mer"):
        fit(table, ModelConfig(max_mer=4))
    with pytest.raises(ValueError, match="alternation"):
        fit(table, ModelConfig(max_mer=GOLDEN_MAX_MER, alternation="left"))


def test_both_linkages_are_accepted():
    """The user-facing contract: cut and joint are both selectable."""
    assert ModelConfig(max_mer=3, linkage="cut").linkage == "cut"
    assert ModelConfig(max_mer=3, linkage="joint").linkage == "joint"


def test_unknown_linkage_is_rejected():
    with pytest.raises(ValueError, match="linkage must be one of"):
        ModelConfig(max_mer=3, linkage="sideways")


def test_unknown_likelihood_is_rejected():
    with pytest.raises(ValueError, match="likelihood must be one of"):
        ModelConfig(max_mer=3, likelihood="poisson")


# --------------------------------------------------------------------------
# Agreement with legacy Baymer
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.legacy
def test_agrees_with_legacy_within_monte_carlo_error(table):
    """The cross-implementation gap must not exceed the within-seed spread.

    baymer2 and legacy do not target a bit-identical posterior: the ALR
    reparameterisation changes the proposal geometry and the corrected
    alpha step changes the chain. So the meaningful question is whether
    the two implementations differ by more than one implementation
    differs from itself under a different seed.
    """
    if not LEGACY_RATE_MATRIX.parent.exists():
        pytest.skip(f"legacy tutorial output not available; set ${LEGACY_ROOT_ENV}")

    chains = [
        [fit(table, _config(seed=s)).rates(l).astype(np.float64)
         for l in range(GOLDEN_MAX_MER)]
        for s in (14, 15)
    ]

    for layer in range(GOLDEN_MAX_MER):
        legacy = np.load(
            str(LEGACY_RATE_MATRIX).format(layer=layer)
        ).mean(axis=0).astype(np.float64)

        within = _median_relative(chains[0][layer], chains[1][layer])
        consensus = (chains[0][layer] + chains[1][layer]) / 2.0
        across = _median_relative(consensus, legacy)

        assert across <= max(4.0 * within, 0.05), (
            f"layer {layer}: baymer2 differs from legacy by {across:.4f} "
            f"but from itself by only {within:.4f}"
        )
        assert np.corrcoef(consensus.ravel(), legacy.ravel())[0, 1] > 0.99


@pytest.mark.slow
@pytest.mark.legacy
def test_root_layer_matches_the_empirical_mle(table):
    """Layer 0 has a flat prior on p, so its posterior mean is the MLE."""
    result = fit(table, _config(seed=14, iterations=8000))
    np.testing.assert_allclose(
        result.rates(0), table.empirical_rates(0), rtol=0.01
    )


# --------------------------------------------------------------------------
# Joint linkage
# --------------------------------------------------------------------------


def _joint_config(seed: int, iterations: int = 3000) -> ModelConfig:
    cfg = _config(seed, iterations)
    cfg.linkage = "joint"
    return cfg


def test_joint_linkage_runs_and_covers_every_layer(table):
    result = fit(table, _joint_config(seed=1, iterations=600))
    assert result.config.linkage == "joint"
    assert len(result.layers) == GOLDEN_MAX_MER
    for layer in range(GOLDEN_MAX_MER):
        rates = result.rates(layer)
        assert rates.shape == (table.tree.layer_size(layer), 3)
        assert np.all(np.isfinite(rates))
        assert np.all(rates > 0)
        assert np.all(rates.sum(axis=1) < 1.0)


def test_joint_is_reproducible(table):
    a = fit(table, _joint_config(seed=5, iterations=400))
    b = fit(table, _joint_config(seed=5, iterations=400))
    for layer in range(GOLDEN_MAX_MER):
        np.testing.assert_array_equal(a.rates(layer), b.rates(layer))


def test_joint_root_tracks_the_pooled_empirical_rate(table):
    """Data enter once at the leaves, so the root is the pooled rate."""
    result = fit(table, _joint_config(seed=14, iterations=4000))
    np.testing.assert_allclose(
        result.rates(0), table.empirical_rates(0), rtol=0.02
    )


@pytest.mark.slow
def test_joint_and_cut_agree_broadly_but_are_not_identical(table):
    """They target different posteriors; the difference should be modest."""
    cut = fit(table, _config(seed=14, iterations=4000))
    joint = fit(table, _joint_config(seed=14, iterations=4000))
    for layer in range(GOLDEN_MAX_MER):
        c, j = cut.rates(layer), joint.rates(layer)
        assert np.corrcoef(c.ravel(), j.ravel())[0, 1] > 0.95
    deepest = GOLDEN_MAX_MER - 1
    difference = _median_relative(cut.rates(deepest), joint.rates(deepest))
    assert 0.0 < difference < 0.25, (
        f"cut and joint differ by {difference:.3f}; they should neither "
        f"coincide nor diverge wildly"
    )


def test_joint_refuses_the_ctmc_likelihood():
    """The ctmc mode is quartet-local and has no joint kernel."""
    with pytest.raises(ValueError, match="joint linkage currently supports"):
        tree = ContextTree(max_mer=3, growth="centred")
        counts = np.ones((tree.n_leaves, 4)) * 100
        fit(
            CountTable(tree=tree, leaf_counts=counts),
            ModelConfig(max_mer=3, growth="centred", linkage="joint",
                        likelihood="ctmc",
                        default_layer=LayerConfig(n_iterations=200, burnin=100,
                                                  thin=2, set_sigma=0.85)),
        )


@pytest.mark.slow
def test_multiplicative_and_alr_agree_at_mutation_rates(table):
    """The two parameterisations differ by O(p), which is ~1e-3 here.

    phi is a log fold change in rate under ``multiplicative`` and in odds
    under ``alr``; the reference probability is within 1e-3 of one at
    mutation rates, so the fitted rates should barely move.
    """
    base = dict(n_iterations=6000, burnin=3000, thin=5,
                set_sigma=LEGACY_SET_SIGMA, c=LEGACY_C,
                phi_proposal_sd=0.05, alpha_init=0.8)
    fits = {}
    for parameterization in ("alr", "multiplicative"):
        fits[parameterization] = fit(table, ModelConfig(
            max_mer=GOLDEN_MAX_MER, seed=14,
            default_layer=LayerConfig(parameterization=parameterization, **base),
        ))
    for layer in range(GOLDEN_MAX_MER):
        a = fits["alr"].rates(layer)
        m = fits["multiplicative"].rates(layer)
        assert np.corrcoef(a.ravel(), m.ravel())[0, 1] > 0.99
        assert _median_relative(a, m) < 0.05

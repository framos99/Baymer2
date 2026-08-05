"""Correctness of the Metropolis-within-Gibbs backend.

The emphasis is on the things legacy Baymer had no way to check: that
each Gibbs step draws from its actual full conditional, that the sampler
recovers parameters it was simulated from, and that results do not
depend on thread count.
"""

from __future__ import annotations

import numba
import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

from baymer2.backends.numba_mh import (
    LayerConfig,
    NumbaMetropolisBackend,
    eta_from_rates,
    rates_from_eta,
    sample_alpha,
    sample_indicators,
    sample_sigma,
)
from baymer2.rng import make_key


def simulate_stats(rates: np.ndarray, n_per_node: int, rng) -> np.ndarray:
    """Multinomial counts from known alternate-allele rates."""
    rates = np.atleast_2d(rates)
    out = np.zeros((rates.shape[0], 4))
    for i, r in enumerate(rates):
        out[i] = rng.multinomial(n_per_node, [1.0 - r.sum(), *r])
    return out


@pytest.fixture
def backend() -> NumbaMetropolisBackend:
    return NumbaMetropolisBackend()


# --------------------------------------------------------------------------
# ALR transform
# --------------------------------------------------------------------------


def test_alr_round_trip_across_magnitudes():
    rng = np.random.default_rng(0)
    for scale in (1e-8, 1e-6, 1e-4, 1e-2, 1e-1):
        rates = rng.uniform(0.1 * scale, scale, size=(50, 3))
        np.testing.assert_allclose(
            rates_from_eta(eta_from_rates(rates)), rates, rtol=1e-10
        )


def test_alr_always_lands_inside_the_simplex():
    """The property that removes legacy's rejection region entirely.

    Held over the range mutation rates actually occupy: rates of 1e-8 to
    1e-3 correspond to eta between roughly -20 and -5, so +/-30 is
    generous headroom.
    """
    rng = np.random.default_rng(1)
    eta = rng.uniform(-30, 30, size=(50000, 3))
    p = rates_from_eta(eta)
    assert np.all(p > 0.0)
    assert np.all(p.sum(axis=1) < 1.0)
    assert np.all(np.isfinite(p))


def test_alr_saturation_is_safe_rather_than_corrupting():
    """Beyond float64's resolution the map saturates; that must not corrupt.

    Once one coordinate dominates by more than ~37, exp(-delta)
    underflows against 1 and the leading rate rounds to exactly 1.0.
    The result must stay finite and non-negative, and the likelihood
    must reject it rather than producing a NaN that would poison the
    chain.
    """
    from baymer2.likelihoods.binomial import node_loglik

    eta = np.array([[200.0, -5.0, -5.0], [1e4, 0.0, -1e4]])
    p = rates_from_eta(eta)
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0.0)
    assert np.all(p.sum(axis=1) <= 1.0)
    for row in p:
        ll = node_loglik(100.0, 1.0, 1.0, 1.0, row[0], row[1], row[2])
        assert ll == -np.inf, "saturated states must be rejected, not NaN"


def test_alr_rejects_invalid_rates():
    with pytest.raises(ValueError, match="strictly less than 1"):
        eta_from_rates(np.array([[0.5, 0.4, 0.3]]))
    with pytest.raises(ValueError, match="strictly positive"):
        eta_from_rates(np.array([[0.0, 0.1, 0.1]]))


# --------------------------------------------------------------------------
# Gibbs steps draw from their actual full conditionals
# --------------------------------------------------------------------------


def test_indicator_gibbs_matches_analytic_conditional():
    """P(z=1) must equal the spike-and-slab posterior odds."""
    alpha, sigma, c = 0.4, 0.8, 0.04472136
    phi_values = np.array([0.0, 0.01, 0.05, 0.2, 1.0, 2.0])
    n = len(phi_values)

    phi = np.tile(phi_values[:, None], (1, 3))
    zero_mask = np.zeros(n, dtype=np.bool_)

    draws = np.zeros((n, 3))
    reps = 4000
    for r in range(reps):
        ind = np.zeros((n, 3), dtype=np.int8)
        key = np.uint64(make_key(np.uint64(99), np.uint64(0), np.uint64(r)))
        sample_indicators(phi, ind, zero_mask, alpha, sigma, c, key)
        draws += ind
    freq = draws / reps

    slab = np.exp(-(phi_values**2) / (2 * sigma**2)) / sigma
    spike = np.exp(-(phi_values**2) / (2 * (c * sigma) ** 2)) / (c * sigma)
    expected = alpha * slab / (alpha * slab + (1 - alpha) * spike)

    for j in range(3):
        np.testing.assert_allclose(freq[:, j], expected, atol=0.025)


def test_indicator_gibbs_favours_spike_at_zero_and_slab_when_large():
    alpha, sigma, c = 0.5, 0.8, 0.04472136
    phi = np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
    zero_mask = np.zeros(2, dtype=np.bool_)
    tally = np.zeros((2, 3))
    for r in range(2000):
        ind = np.zeros((2, 3), dtype=np.int8)
        key = np.uint64(make_key(np.uint64(5), np.uint64(0), np.uint64(r)))
        sample_indicators(phi, ind, zero_mask, alpha, sigma, c, key)
        tally += ind
    assert np.all(tally[0] / 2000 < 0.1), "phi=0 should sit in the spike"
    assert np.all(tally[1] / 2000 > 0.9), "large phi should sit in the slab"


def test_alpha_gibbs_has_correct_beta_moments():
    """Bug (c): legacy filtered this exact Gibbs draw through an MH ratio.

    An unfiltered draw reproduces Beta(1+S, 1+n-S) moments; a filtered
    one is over-concentrated and biased toward the mode.
    """
    n_nodes = 200
    rng = np.random.default_rng(3)
    indicator = (rng.random((n_nodes, 3)) < 0.3).astype(np.int8)
    zero_mask = np.zeros(n_nodes, dtype=np.bool_)

    total = int(indicator.sum())
    active = n_nodes * 3
    a, b = 1.0 + total, 1.0 + active - total

    draws = np.array([
        sample_alpha(indicator, zero_mask,
                     np.uint64(make_key(np.uint64(7), np.uint64(0), np.uint64(r))))
        for r in range(3000)
    ])

    exp_mean = a / (a + b)
    exp_var = a * b / ((a + b) ** 2 * (a + b + 1))
    assert draws.mean() == pytest.approx(exp_mean, abs=4 * np.sqrt(exp_var / 3000))
    # Variance is the discriminating statistic: an MH filter shrinks it.
    assert draws.var() == pytest.approx(exp_var, rel=0.20)


def test_alpha_gibbs_ignores_zero_masked_nodes():
    n = 100
    indicator = np.ones((n, 3), dtype=np.int8)
    mask = np.zeros(n, dtype=np.bool_)
    mask[50:] = True
    indicator[50:] = 0  # masked nodes carry stale values
    key = np.uint64(make_key(np.uint64(1), np.uint64(0), np.uint64(0)))
    # 50 active nodes, all indicators 1 -> alpha should be near 1
    assert sample_alpha(indicator, mask, key) > 0.95


# --------------------------------------------------------------------------
# Sigma sampling -- legacy could not run this at all (bug a)
# --------------------------------------------------------------------------


def test_sigma_sampler_runs_and_concentrates_near_truth():
    sigma_true, c = 0.7, 0.04472136
    n = 3000
    rng = np.random.default_rng(11)
    phi = rng.normal(0.0, sigma_true, size=(n, 3))
    indicator = np.ones((n, 3), dtype=np.int8)
    zero_mask = np.zeros(n, dtype=np.bool_)

    sigma = 0.2  # start far away
    for r in range(3000):
        key = np.uint64(make_key(np.uint64(21), np.uint64(0), np.uint64(r)))
        sigma, _ = sample_sigma(
            phi, indicator, zero_mask, sigma, c, np.log(0.05), 1.0 / c, key
        )
    assert sigma == pytest.approx(sigma_true, rel=0.05)


def test_sigma_respects_its_upper_bound():
    c = 0.04472136
    n = 10
    phi = np.zeros((n, 3))
    indicator = np.ones((n, 3), dtype=np.int8)
    zero_mask = np.zeros(n, dtype=np.bool_)
    sigma = 1.0
    for r in range(500):
        key = np.uint64(make_key(np.uint64(2), np.uint64(0), np.uint64(r)))
        sigma, _ = sample_sigma(
            phi, indicator, zero_mask, sigma, c, np.log(0.3), 2.0, key
        )
        assert 0.0 < sigma < 2.0


# --------------------------------------------------------------------------
# Parameter recovery
# --------------------------------------------------------------------------


def test_layer0_recovers_known_rates(backend):
    true = np.array([[1e-3, 5e-3, 2e-3], [2e-3, 1e-2, 3e-3]])
    rng = np.random.default_rng(0)
    stats = simulate_stats(true, 5_000_000, rng)
    zero_mask = np.zeros(2, dtype=np.bool_)

    cfg = LayerConfig(n_iterations=4000, burnin=2000, thin=5, seed=14,
                      phi_proposal_sd=0.02, set_sigma=0.85)
    res = backend.run_layer(0, stats, zero_mask, cfg)

    np.testing.assert_allclose(res.mean_rates, true, rtol=0.02)
    assert 0.15 < res.accept_rate_phi < 0.75


def test_layer0_posterior_tracks_the_mle(backend):
    """With a flat prior on p the posterior mean must sit on the MLE."""
    true = np.array([[2e-3, 4e-3, 1e-3]])
    rng = np.random.default_rng(5)
    stats = simulate_stats(true, 2_000_000, rng)
    zero_mask = np.zeros(1, dtype=np.bool_)
    cfg = LayerConfig(n_iterations=4000, burnin=2000, thin=5, seed=3,
                      phi_proposal_sd=0.02, set_sigma=0.85)
    res = backend.run_layer(0, stats, zero_mask, cfg)
    mle = stats[:, 1:] / stats.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(res.mean_rates, mle, rtol=0.01)


def test_two_layer_cut_linkage_recovers_child_rates(backend):
    """Layer 1 fitted against layer 0's posterior must recover truth."""
    rng = np.random.default_rng(7)
    parent_true = np.array([[1e-3, 3e-3, 2e-3], [2e-3, 5e-3, 1e-3]])
    # Children deviate from their parent by a real, recoverable amount.
    child_true = np.repeat(parent_true, 4, axis=0) * np.exp(
        rng.normal(0.0, 0.3, size=(8, 3))
    )

    parent_stats = simulate_stats(parent_true, 4_000_000, rng)
    child_stats = simulate_stats(child_true, 1_000_000, rng)

    cfg0 = LayerConfig(n_iterations=3000, burnin=1500, thin=5, seed=1,
                       phi_proposal_sd=0.02, set_sigma=0.85)
    res0 = backend.run_layer(
        0, parent_stats, np.zeros(2, dtype=np.bool_), cfg0
    )

    cfg1 = LayerConfig(n_iterations=6000, burnin=3000, thin=5, seed=2,
                       phi_proposal_sd=0.05, set_sigma=0.85, alpha_init=0.9)
    res1 = backend.run_layer(
        1, child_stats, np.zeros(8, dtype=np.bool_), cfg1,
        parent_samples=res0.eta.astype(np.float64),
    )

    np.testing.assert_allclose(res1.mean_rates, child_true, rtol=0.10)


def test_zero_masked_nodes_are_left_alone(backend):
    true = np.array([[1e-3, 2e-3, 1e-3], [1e-3, 2e-3, 1e-3]])
    rng = np.random.default_rng(9)
    stats = simulate_stats(true, 1_000_000, rng)
    stats[1] = 0.0  # node 1 has no data at all
    zero_mask = np.array([False, True])

    cfg = LayerConfig(n_iterations=1500, burnin=700, thin=5, seed=4,
                      phi_proposal_sd=0.02, set_sigma=0.85)
    res = backend.run_layer(0, stats, zero_mask, cfg)

    assert np.all(res.rates[:, 1, :] == res.rates[0, 1, :]), (
        "a node with no data must never move"
    )
    np.testing.assert_allclose(res.mean_rates[0], true[0], rtol=0.05)


# --------------------------------------------------------------------------
# Reproducibility (bug h)
# --------------------------------------------------------------------------


def test_identical_results_across_thread_counts(backend):
    true = np.array([[1e-3, 3e-3, 2e-3]] * 8)
    rng = np.random.default_rng(2)
    stats = simulate_stats(true, 500_000, rng)
    zero_mask = np.zeros(8, dtype=np.bool_)
    cfg = LayerConfig(n_iterations=800, burnin=400, thin=4, seed=77,
                      phi_proposal_sd=0.02, set_sigma=0.85)

    original = get_num_threads()
    max_threads = numba.config.NUMBA_NUM_THREADS
    counts = sorted({1, 2, max_threads})
    if len(counts) < 2:
        pytest.skip(f"only {max_threads} numba thread(s) available")
    results = []
    try:
        for t in counts:
            set_num_threads(t)
            results.append(backend.run_layer(0, stats, zero_mask, cfg).rates.copy())
    finally:
        set_num_threads(original)
    for r in results[1:]:
        np.testing.assert_array_equal(results[0], r)


def test_same_seed_reproduces_exactly(backend):
    true = np.array([[1e-3, 3e-3, 2e-3]] * 4)
    rng = np.random.default_rng(2)
    stats = simulate_stats(true, 200_000, rng)
    zero_mask = np.zeros(4, dtype=np.bool_)
    cfg = LayerConfig(n_iterations=600, burnin=300, thin=3, seed=5,
                      phi_proposal_sd=0.02, set_sigma=0.85)
    a = backend.run_layer(0, stats, zero_mask, cfg)
    b = backend.run_layer(0, stats, zero_mask, cfg)
    np.testing.assert_array_equal(a.rates, b.rates)


def test_different_seeds_give_different_chains(backend):
    true = np.array([[1e-3, 3e-3, 2e-3]] * 4)
    rng = np.random.default_rng(2)
    stats = simulate_stats(true, 200_000, rng)
    zero_mask = np.zeros(4, dtype=np.bool_)
    base = dict(n_iterations=600, burnin=300, thin=3,
                phi_proposal_sd=0.02, set_sigma=0.85)
    a = backend.run_layer(0, stats, zero_mask, LayerConfig(seed=1, **base))
    b = backend.run_layer(0, stats, zero_mask, LayerConfig(seed=2, **base))
    assert not np.array_equal(a.rates, b.rates)


# --------------------------------------------------------------------------
# Configuration validation (bug i)
# --------------------------------------------------------------------------


def test_set_sigma_zero_is_rejected_not_silently_reinterpreted():
    """Legacy's ``if set_sigma:`` read 0 as 'do not fix', then crashed."""
    with pytest.raises(ValueError, match="set_sigma must be positive or None"):
        LayerConfig(set_sigma=0.0)


def test_burnin_must_be_less_than_iterations():
    with pytest.raises(ValueError, match="burnin"):
        LayerConfig(n_iterations=100, burnin=100)


def test_c_must_be_a_proper_fraction():
    with pytest.raises(ValueError, match="c must be in"):
        LayerConfig(c=1.5)


def test_sample_count_is_consistent():
    cfg = LayerConfig(n_iterations=1000, burnin=400, thin=6)
    assert cfg.n_samples == 100


def test_parent_sample_shape_is_validated(backend):
    stats = np.array([[100.0, 1.0, 1.0, 1.0]] * 8)
    zero_mask = np.zeros(8, dtype=np.bool_)
    cfg = LayerConfig(n_iterations=200, burnin=100, thin=2, set_sigma=0.85)
    bad_parents = np.zeros((10, 3, 3))  # 3 parents cannot yield 8 nodes
    with pytest.raises(ValueError, match="cannot produce"):
        backend.run_layer(1, stats, zero_mask, cfg, parent_samples=bad_parents)


# --------------------------------------------------------------------------
# Legacy multiplicative parameterisation
# --------------------------------------------------------------------------


def test_parameterization_is_validated():
    with pytest.raises(ValueError, match="parameterization must be"):
        LayerConfig(parameterization="polar")
    assert LayerConfig().parameterization == "alr"


def test_multiplicative_recovers_known_rates(backend):
    true = np.array([[1e-3, 5e-3, 2e-3], [2e-3, 1e-2, 3e-3]])
    rng = np.random.default_rng(0)
    stats = simulate_stats(true, 5_000_000, rng)
    cfg = LayerConfig(n_iterations=4000, burnin=2000, thin=5, seed=14,
                      phi_proposal_sd=0.02, set_sigma=0.85,
                      parameterization="multiplicative")
    res = backend.run_layer(0, stats, np.zeros(2, dtype=np.bool_), cfg)
    np.testing.assert_allclose(res.mean_rates, true, rtol=0.02)
    assert 0.15 < res.accept_rate_phi < 0.75


def test_multiplicative_never_leaves_the_simplex(backend):
    """Rejection replaces legacy's redraw, so the state stays valid."""
    true = np.array([[0.05, 0.08, 0.06]] * 4)
    rng = np.random.default_rng(3)
    stats = simulate_stats(true, 50_000, rng)
    cfg = LayerConfig(n_iterations=2000, burnin=1000, thin=4, seed=2,
                      phi_proposal_sd=0.3, set_sigma=0.85,
                      parameterization="multiplicative")
    res = backend.run_layer(0, stats, np.zeros(4, dtype=np.bool_), cfg)
    rates = np.asarray(res.rates, dtype=np.float64)
    assert np.all(rates > 0.0)
    assert np.all(rates.sum(axis=2) < 1.0)


def test_multiplicative_is_reproducible(backend):
    true = np.array([[1e-3, 3e-3, 2e-3]] * 4)
    rng = np.random.default_rng(2)
    stats = simulate_stats(true, 200_000, rng)
    zero = np.zeros(4, dtype=np.bool_)
    cfg = LayerConfig(n_iterations=600, burnin=300, thin=3, seed=5,
                      phi_proposal_sd=0.02, set_sigma=0.85,
                      parameterization="multiplicative")
    np.testing.assert_array_equal(
        backend.run_layer(0, stats, zero, cfg).rates,
        backend.run_layer(0, stats, zero, cfg).rates,
    )

"""Does the posterior have the width it claims?

Every other test checks where the posterior *sits*. A sampler can
recover means perfectly while reporting intervals twice too narrow, and
no recovery test would notice -- the result would be overconfident
credible intervals on every rate the model reports.

These run a real fit per replicate, so they are marked ``slow``. The
full sweep lives in ``validation/check_calibration.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from baymer2.backends.numba_mh import LayerConfig, rates_from_eta
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.model import ModelConfig, fit

ALPHA, SIGMA, C = 0.6, 0.3, 0.04472136


def simulate_from_prior(tree, rng, *, alpha=ALPHA, sigma=SIGMA, c=C):
    """Draw true ALR coordinates for every layer, top down."""
    etas = [rng.uniform(-9.0, -6.0, size=(tree.layer_size(0), 3))]
    for layer in range(1, tree.n_layers):
        repeated = np.repeat(etas[-1], tree.branching, axis=0)
        indicator = rng.random((tree.layer_size(layer), 3)) < alpha
        phi = rng.normal(0.0, np.where(indicator, sigma, c * sigma))
        etas.append(repeated + phi)
    return etas


def simulate_counts(tree, eta_leaf, depth, rng):
    rates = rates_from_eta(eta_leaf)
    counts = np.zeros((rates.shape[0], 4))
    for i, p in enumerate(rates):
        counts[i] = rng.multinomial(depth, [1.0 - p.sum(), *p])
    return CountTable(tree=tree, leaf_counts=counts, name="sim"), rates


def coverage(samples, truth, level=0.95):
    tail = (1.0 - level) / 2.0
    lower = np.quantile(samples, tail, axis=0)
    upper = np.quantile(samples, 1.0 - tail, axis=0)
    return float(((truth >= lower) & (truth <= upper)).mean())


def _run(max_mer=5, replicates=3, depth=200_000, iterations=8000):
    tree = ContextTree(max_mer=max_mer, alternation="right")
    pooled_cov = {l: [] for l in range(tree.n_layers)}
    cascade_cov = {l: [] for l in range(tree.n_layers)}
    inflation = {l: [] for l in range(tree.n_layers)}

    for replicate in range(replicates):
        rng = np.random.default_rng(1000 + replicate)
        etas = simulate_from_prior(tree, rng)
        table, leaf_rates = simulate_counts(tree, etas[-1], depth, rng)

        result = fit(table, ModelConfig(
            max_mer=max_mer, seed=replicate,
            default_layer=LayerConfig(
                n_iterations=iterations, burnin=iterations // 2, thin=5,
                set_sigma=SIGMA, c=C, alpha_init=ALPHA, phi_proposal_sd=0.05,
            ),
        ))

        for layer in range(tree.n_layers):
            samples = np.asarray(result.layers[layer].rates, dtype=np.float64)
            cascade = rates_from_eta(etas[layer])
            pooled = leaf_rates.reshape(tree.layer_size(layer), -1, 3).mean(axis=1)
            pooled_cov[layer].append(coverage(samples, pooled))
            cascade_cov[layer].append(coverage(samples, cascade))
            inflation[layer].append(float(np.mean(pooled / cascade)))
    return tree, pooled_cov, cascade_cov, inflation


@pytest.mark.slow
def test_credible_intervals_have_nominal_coverage():
    """95% intervals must contain the truth ~95% of the time.

    Judged against the estimand the model actually fits: an internal
    node's likelihood is a multinomial on its *pooled* descendant
    counts, so the parameter it identifies is the aggregate rate over
    those descendants.
    """
    tree, pooled_cov, _, _ = _run()
    for layer in range(tree.n_layers):
        observed = float(np.mean(pooled_cov[layer]))
        assert 0.90 <= observed <= 1.0, (
            f"layer {layer}: 95% intervals covered {observed:.3f} of true "
            f"pooled rates; the posterior width is wrong"
        )


@pytest.mark.slow
def test_intervals_are_not_absurdly_wide():
    """Guard the other direction: coverage bought by useless intervals."""
    tree = ContextTree(max_mer=3, alternation="right")
    rng = np.random.default_rng(7)
    etas = simulate_from_prior(tree, rng)
    table, leaf_rates = simulate_counts(tree, etas[-1], 500_000, rng)
    result = fit(table, ModelConfig(
        max_mer=3, seed=1,
        default_layer=LayerConfig(
            n_iterations=8000, burnin=4000, thin=5,
            set_sigma=SIGMA, c=C, alpha_init=ALPHA, phi_proposal_sd=0.05,
        ),
    ))
    samples = np.asarray(result.layers[0].rates, dtype=np.float64)
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
    truth = leaf_rates.reshape(2, -1, 3).mean(axis=1)
    relative_width = np.mean((upper - lower) / truth)
    assert relative_width < 0.5, (
        f"root intervals span {relative_width:.2f} of the rate; that is not "
        f"informative at this depth"
    )


@pytest.mark.slow
def test_internal_nodes_estimate_the_pooled_rate_not_the_cascade_value():
    """A real property of Baymer's formulation, pinned so it stays known.

    ``phi`` is Gaussian on the log scale, so ``E[exp(phi)] > 1`` by
    Jensen and each layer of descent inflates a node's aggregate rate
    above the multiplicative cascade value sitting at it. An internal
    node's fitted rate is therefore the arithmetic mean of its
    descendants' rates, which is what its pooled multinomial identifies
    -- not the value a top-down simulation placed there. Interpreting
    shallow-layer rates as cascade values overstates nothing, but
    conflating the two does.
    """
    tree, pooled_cov, cascade_cov, inflation = _run(max_mer=5, replicates=3)

    per_layer = ALPHA * np.exp(SIGMA**2 / 2) + (1 - ALPHA) * np.exp((C * SIGMA) ** 2 / 2)
    deepest = tree.n_layers - 1

    # Leaves: the two definitions coincide exactly.
    assert np.mean(inflation[deepest]) == pytest.approx(1.0, abs=1e-9)

    # Shallower: inflation compounds with the number of descent layers.
    for layer in range(tree.n_layers - 1):
        depth = deepest - layer
        predicted = per_layer**depth
        assert np.mean(inflation[layer]) == pytest.approx(predicted, rel=0.05), (
            f"layer {layer}: expected ~{predicted:.4f} inflation over {depth} "
            f"layers of descent"
        )

    # And the mismatch is what makes cascade coverage fail at the root
    # while pooled coverage holds -- the discriminating observation.
    assert np.mean(cascade_cov[0]) < 0.5
    assert np.mean(pooled_cov[0]) > 0.9

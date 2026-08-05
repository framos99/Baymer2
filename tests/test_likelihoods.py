"""Validate the binomial likelihood against legacy Baymer's kernel.

``_legacy_edge_likelihood`` is a transcription of
``run_baymer.get_edge_likelihood_numba``, operating on the flat
``leaf_counts`` layout legacy used. It is the definition of correctness
for this mode; the O(1) kernel must reproduce it to floating-point
tolerance on real data.
"""

from __future__ import annotations

import numpy as np
import pytest

from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.likelihoods import MODES, BinomialLikelihood
from baymer2.likelihoods.binomial import layer_loglik, node_loglik

from conftest import GOLDEN_ALTERNATION, GOLDEN_MAX_MER


def _legacy_edge_likelihood(leaf_counts: np.ndarray, p_vec: np.ndarray) -> float:
    """Verbatim transcription of ``get_edge_likelihood_numba``."""
    return (
        np.sum(leaf_counts[0::4]) * np.log(1.0 - np.sum(p_vec))
        + np.sum(leaf_counts[1::4]) * np.log(p_vec[0])
        + np.sum(leaf_counts[2::4]) * np.log(p_vec[1])
        + np.sum(leaf_counts[3::4]) * np.log(p_vec[2])
    )


def _legacy_flat_leaf_counts(count_dict, tree: ContextTree, layer: int) -> np.ndarray:
    """Rebuild legacy's flat (layer_size, 4*n_leaves) leaf count array."""
    focal_idx = tree.focal_index(tree.max_mer)
    rows = []
    for ctx in tree.contexts(layer):
        flat = []
        for leaf in tree.leaf_contexts(ctx):
            c = list(count_dict[leaf][:4])
            if leaf[focal_idx] == "C":
                c = [c[1], c[0], c[2], c[3]]
            flat.extend(c)
        rows.append(flat)
    return np.asarray(rows, dtype=np.float64)


@pytest.fixture(scope="module")
def tree() -> ContextTree:
    return ContextTree(max_mer=GOLDEN_MAX_MER, alternation=GOLDEN_ALTERNATION)


@pytest.fixture(scope="module")
def table(golden_count_dicts, tree) -> CountTable:
    return CountTable.from_count_dict(golden_count_dicts[GOLDEN_MAX_MER], tree)


@pytest.fixture(scope="module")
def lik() -> BinomialLikelihood:
    return BinomialLikelihood()


def _plausible_rates(rng, n: int) -> np.ndarray:
    """Rate vectors on the scale real mutation probabilities occupy."""
    return rng.uniform(1e-8, 1e-3, size=(n, 3))


# --------------------------------------------------------------------------
# Agreement with the legacy kernel, on real data
# --------------------------------------------------------------------------


def test_matches_legacy_kernel_on_golden_counts(table, golden_count_dicts, tree, lik):
    rng = np.random.default_rng(0)
    for layer in range(tree.n_layers):
        flat = _legacy_flat_leaf_counts(golden_count_dicts[GOLDEN_MAX_MER], tree, layer)
        rates = _plausible_rates(rng, tree.layer_size(layer))

        expected = np.array(
            [_legacy_edge_likelihood(flat[i], rates[i]) for i in range(flat.shape[0])]
        )
        got = lik.log_likelihood(table.layer_stats(layer), rates)

        # Empty nodes: legacy produced 0*log(...) == 0 or nan; we define 0.
        empty = table.zero_mask(layer)
        np.testing.assert_allclose(got[~empty], expected[~empty], rtol=1e-12)
        assert np.all(got[empty] == 0.0)


def test_compiled_kernel_matches_numpy_reference(table, tree, lik):
    rng = np.random.default_rng(1)
    for layer in range(tree.n_layers):
        stats = table.layer_stats(layer)
        rates = _plausible_rates(rng, stats.shape[0])
        out = np.zeros(stats.shape[0])
        layer_loglik(stats, rates, out)
        np.testing.assert_allclose(out, lik.log_likelihood(stats, rates), rtol=1e-12)


def test_node_kernel_matches_layer_kernel(table, tree):
    rng = np.random.default_rng(2)
    stats = table.layer_stats(2)
    rates = _plausible_rates(rng, stats.shape[0])
    out = np.zeros(stats.shape[0])
    layer_loglik(stats, rates, out)
    for i in range(stats.shape[0]):
        assert out[i] == pytest.approx(
            node_loglik(*stats[i], *rates[i]), rel=1e-12
        )


# --------------------------------------------------------------------------
# Numerical behaviour
# --------------------------------------------------------------------------


def test_log1p_beats_legacy_at_small_rates():
    """The reference term is where legacy lost precision."""
    s = np.array([[1e12, 1.0, 1.0, 1.0]])
    p = np.array([[1e-12, 1e-12, 1e-12]])
    exact = 1e12 * np.log1p(-3e-12) + 3 * np.log(1e-12)
    got = BinomialLikelihood().log_likelihood(s, p)[0]
    legacy = _legacy_edge_likelihood(
        np.array([1e12, 1.0, 1.0, 1.0]), np.array([1e-12, 1e-12, 1e-12])
    )
    assert abs(got - exact) < abs(legacy - exact)


def test_empty_nodes_contribute_nothing(lik):
    stats = np.zeros((3, 4))
    rates = np.full((3, 3), 1e-4)
    assert np.all(lik.log_likelihood(stats, rates) == 0.0)


def test_out_of_simplex_is_negative_infinity(lik):
    stats = np.array([[100.0, 1.0, 1.0, 1.0]])
    assert lik.log_likelihood(stats, np.array([[0.5, 0.4, 0.3]]))[0] == -np.inf
    assert lik.log_likelihood(stats, np.array([[0.0, 0.1, 0.1]]))[0] == -np.inf
    assert lik.log_likelihood(stats, np.array([[-0.1, 0.1, 0.1]]))[0] == -np.inf


def test_zero_alt_counts_do_not_produce_nan(lik):
    """A context with opportunities but no observed mutations is common."""
    stats = np.array([[1000.0, 0.0, 0.0, 0.0]])
    out = lik.log_likelihood(stats, np.array([[1e-6, 1e-6, 1e-6]]))
    assert np.isfinite(out[0])
    assert out[0] == pytest.approx(1000.0 * np.log1p(-3e-6))


def test_likelihood_is_maximised_at_the_empirical_rate(lik):
    """The multinomial MLE is the observed fraction."""
    stats = np.array([[9700.0, 100.0, 150.0, 50.0]])
    mle = stats[0, 1:] / stats[0].sum()
    best = lik.log_likelihood(stats, mle[None, :])[0]
    rng = np.random.default_rng(3)
    for _ in range(200):
        perturbed = mle * rng.uniform(0.5, 1.5, size=3)
        assert lik.log_likelihood(stats, perturbed[None, :])[0] <= best + 1e-9


def test_scales_linearly_with_duplicated_counts(lik):
    stats = np.array([[1000.0, 10.0, 20.0, 5.0]])
    rates = np.array([[1e-2, 2e-2, 5e-3]])
    single = lik.log_likelihood(stats, rates)[0]
    assert lik.log_likelihood(stats * 3, rates)[0] == pytest.approx(3 * single)


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------


def test_spec_reports_binomial_mode(lik):
    spec = lik.spec(layer=0)
    assert spec.mode == MODES["binomial"]
    assert spec.name == "binomial"
    assert spec.per_node.shape == (0, 0)


def test_rejects_mismatched_shapes(lik):
    with pytest.raises(ValueError, match=r"stats must be"):
        lik.log_likelihood(np.zeros((4, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match=r"rates must be"):
        lik.log_likelihood(np.zeros((4, 4)), np.zeros((5, 3)))

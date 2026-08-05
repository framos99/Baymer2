"""The recurrent likelihood must reduce to the binomial and beat it on truth.

The claim being tested is specific: when sites can be hit more than
once, the observed mutated fraction is ``1 - exp(-L)``, not ``L``, so a
binomial fit understates the rate. The understatement must be
negligible at genome-average rates and large where recurrence is real.
"""

from __future__ import annotations

import numpy as np
import pytest

from baymer2.likelihoods.binomial import BinomialLikelihood
from baymer2.likelihoods.recurrent import (
    RecurrentLikelihood,
    recurrent_layer_loglik,
    recurrent_node_loglik,
)


@pytest.fixture
def lik() -> RecurrentLikelihood:
    return RecurrentLikelihood()


def _simulate(rates: np.ndarray, n_sites: int, rng) -> np.ndarray:
    """Counts from competing Poisson processes: what the data really is."""
    rates = np.atleast_2d(rates)
    out = np.zeros((rates.shape[0], 4))
    for i, lam in enumerate(rates):
        total = lam.sum()
        n_hit = rng.binomial(n_sites, -np.expm1(-total))
        which = rng.multinomial(n_hit, lam / total)
        out[i, 0] = n_sites - n_hit
        out[i, 1:] = which
    return out


# --------------------------------------------------------------------------
# Reduction to the binomial
# --------------------------------------------------------------------------


def test_converges_to_binomial_as_rates_shrink(lik):
    rng = np.random.default_rng(0)
    n = 64
    stats = np.zeros((n, 4))
    stats[:, 0] = 1_000_000
    stats[:, 1:] = rng.integers(10, 200, size=(n, 3))

    previous = np.inf
    for scale in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        rates = rng.uniform(0.5, 1.5, size=(n, 3)) * scale
        gap = np.max(
            np.abs(
                lik.log_likelihood(stats, rates)
                - BinomialLikelihood().log_likelihood(stats, rates)
            )
        )
        relative = gap / abs(lik.log_likelihood(stats, rates).sum())
        assert relative < previous
        previous = relative
    assert previous < 1e-6


def test_differs_from_binomial_at_high_rates(lik):
    """If they agreed everywhere the mode would be pointless."""
    stats = np.array([[500_000.0, 200_000.0, 200_000.0, 100_000.0]])
    rates = np.array([[0.4, 0.4, 0.3]])
    assert abs(
        lik.log_likelihood(stats, rates)[0]
        - BinomialLikelihood().log_likelihood(stats, rates)[0]
    ) > 100.0


def test_observed_fraction_saturates():
    """The ceiling the binomial cannot represent."""
    frac = RecurrentLikelihood.observed_fraction(np.array([[1.0, 1.0, 1.0]]))
    assert frac[0] == pytest.approx(1.0 - np.exp(-3.0))
    assert frac[0] < 1.0
    tiny = RecurrentLikelihood.observed_fraction(np.array([[1e-9, 1e-9, 1e-9]]))
    assert tiny[0] == pytest.approx(3e-9, rel=1e-6)


def test_rates_are_not_capped_at_one(lik):
    """Unlike the binomial, total rate may exceed 1 -- it is a count."""
    stats = np.array([[10.0, 100.0, 100.0, 100.0]])
    out = lik.log_likelihood(stats, np.array([[2.0, 3.0, 4.0]]))
    assert np.isfinite(out[0])
    assert BinomialLikelihood().log_likelihood(
        stats, np.array([[2.0, 3.0, 4.0]])
    )[0] == -np.inf


# --------------------------------------------------------------------------
# Kernels agree with the reference
# --------------------------------------------------------------------------


def test_compiled_kernel_matches_numpy_reference(lik):
    rng = np.random.default_rng(1)
    n = 200
    stats = rng.integers(0, 10000, size=(n, 4)).astype(float)
    rates = rng.uniform(1e-6, 0.5, size=(n, 3))
    out = np.zeros(n)
    recurrent_layer_loglik(stats, rates, out)
    np.testing.assert_allclose(out, lik.log_likelihood(stats, rates), rtol=1e-11)


def test_node_kernel_matches_layer_kernel():
    rng = np.random.default_rng(2)
    n = 50
    stats = rng.integers(1, 5000, size=(n, 4)).astype(float)
    rates = rng.uniform(1e-4, 0.3, size=(n, 3))
    out = np.zeros(n)
    recurrent_layer_loglik(stats, rates, out)
    for i in range(n):
        assert out[i] == pytest.approx(
            recurrent_node_loglik(*stats[i], *rates[i]), rel=1e-12
        )


def test_log1mexp_is_accurate_at_tiny_rates():
    """The naive log(1 - exp(-L)) loses all precision here."""
    lam = 1e-12
    got = recurrent_node_loglik(0.0, 1.0, 0.0, 0.0, lam / 3, lam / 3, lam / 3)
    expected = np.log(lam) - np.log(3.0) + np.log(lam) - np.log(lam)
    assert np.isfinite(got)
    assert got == pytest.approx(np.log(lam / 3), rel=1e-9)


# --------------------------------------------------------------------------
# The scientific claim
# --------------------------------------------------------------------------


def test_binomial_understates_recurrent_rates_and_recurrent_does_not():
    """Fit both to data generated with real recurrence; compare recovery."""
    rng = np.random.default_rng(7)
    true = np.array([[0.30, 0.45, 0.25]])  # total 1.0 expected events per site
    n_sites = 2_000_000
    stats = _simulate(true, n_sites, rng)

    # Binomial MLE is the observed fraction per allele.
    binom_mle = stats[0, 1:] / n_sites
    # Recurrent MLE inverts the saturation.
    observed_any = stats[0, 1:].sum() / n_sites
    total_lambda = -np.log1p(-observed_any)
    recurrent_mle = total_lambda * stats[0, 1:] / stats[0, 1:].sum()

    assert binom_mle.sum() < true.sum() * 0.85, "binomial must understate"
    np.testing.assert_allclose(recurrent_mle, true[0], rtol=0.02)

    # And the recurrent likelihood must prefer the truth over the binomial MLE.
    lik = RecurrentLikelihood()
    assert (
        lik.log_likelihood(stats, true)[0]
        > lik.log_likelihood(stats, binom_mle[None, :])[0]
    )


def test_recurrent_likelihood_is_maximised_at_its_mle():
    rng = np.random.default_rng(9)
    true = np.array([[0.20, 0.30, 0.10]])
    stats = _simulate(true, 1_000_000, rng)
    lik = RecurrentLikelihood()
    best = lik.log_likelihood(stats, true)[0]
    for factor in (0.8, 0.9, 1.1, 1.25):
        assert lik.log_likelihood(stats, true * factor)[0] < best


def test_empty_nodes_contribute_nothing(lik):
    assert np.all(lik.log_likelihood(np.zeros((3, 4)), np.full((3, 3), 0.1)) == 0.0)


def test_non_positive_rates_rejected(lik):
    stats = np.array([[100.0, 1.0, 1.0, 1.0]])
    assert lik.log_likelihood(stats, np.array([[0.0, 0.1, 0.1]]))[0] == -np.inf
    assert lik.log_likelihood(stats, np.array([[-0.1, 0.1, 0.1]]))[0] == -np.inf


def test_rejects_mismatched_shapes(lik):
    with pytest.raises(ValueError, match="stats must be"):
        lik.log_likelihood(np.zeros((4, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="rates must be"):
        lik.log_likelihood(np.zeros((4, 4)), np.zeros((5, 3)))

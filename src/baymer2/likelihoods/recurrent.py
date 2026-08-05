"""Recurrent mutation: a site may be hit more than once.

The binomial likelihood treats each site as one Bernoulli trial, so it
reads the *fraction of sites that carry a mutation* as the mutation
rate. Those coincide only while the per-site expected number of events
is far below one. At hypermutable contexts in a large cohort -- CpG
transitions above all -- it is not, and the binomial saturates: no
matter how high the true rate goes, the observed fraction cannot exceed
one.

Model
-----
Treat the three alternate alleles as competing Poisson processes with
expected event counts ``lambda_j`` over the sampled history. Then::

    P(site unmutated)     = exp(-L)                     where L = sum_j lambda_j
    P(site carries j)     = (lambda_j / L) * (1 - exp(-L))

so the probability of seeing *any* mutation is ``1 - exp(-L)`` rather
than ``L``. The parameters keep their shape -- three per context -- but
are now expected event counts rather than probabilities, which removes
the ``sum < 1`` ceiling entirely.

As ``L -> 0``, ``1 - exp(-L) -> L`` and this reduces exactly to
:class:`~baymer2.likelihoods.binomial.BinomialLikelihood`. The
difference is second order in ``L``, so it is invisible at genome-average
rates and material precisely where recurrence is suspected.

The correction is node-local: unlike the CTMC mode it needs no rates
from neighbouring contexts, so it drops into the existing sampler
without a quartet structure.

Not modelled here
-----------------
Explicit per-site multiplicities. Those need data the current pipeline
discards -- legacy ``mutation_counter.py`` rejects multiallelic sites
outright at ``len(REF) + len(ALT) > 2`` -- and are a separate mode.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from baymer2.likelihoods.base import MODES, LikelihoodSpec

__all__ = ["RecurrentLikelihood", "recurrent_node_loglik", "recurrent_layer_loglik"]


@njit(inline="always", cache=True)
def recurrent_node_loglik(s0, s1, s2, s3, l0, l1, l2):
    """Competing-Poisson log-likelihood for one node.

    ``log(1 - exp(-L))`` is evaluated as ``log(-expm1(-L))``, which stays
    accurate as ``L`` approaches zero where the naive form loses every
    significant digit.
    """
    total = s0 + s1 + s2 + s3
    if total == 0.0:
        return 0.0
    if l0 <= 0.0 or l1 <= 0.0 or l2 <= 0.0:
        return -np.inf

    lam = l0 + l1 + l2
    # P(any mutation) = 1 - exp(-lam)
    log_any = np.log(-np.expm1(-lam))

    ll = -s0 * lam
    n_mut = s1 + s2 + s3
    if n_mut != 0.0:
        ll += n_mut * (log_any - np.log(lam))
        if s1 != 0.0:
            ll += s1 * np.log(l0)
        if s2 != 0.0:
            ll += s2 * np.log(l1)
        if s3 != 0.0:
            ll += s3 * np.log(l2)
    return ll


@njit(parallel=True, cache=True)
def recurrent_layer_loglik(stats, rates, out):
    n = stats.shape[0]
    for i in prange(n):
        out[i] = recurrent_node_loglik(
            stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3],
            rates[i, 0], rates[i, 1], rates[i, 2],
        )
    return out


class RecurrentLikelihood:
    """Poisson competing-risks likelihood with an at-least-one-hit term."""

    name = "recurrent"

    def spec(self, layer: int) -> LikelihoodSpec:
        return LikelihoodSpec(mode=MODES[self.name])

    def log_likelihood(self, stats: np.ndarray, rates: np.ndarray) -> np.ndarray:
        """Readable numpy reference; the kernels are checked against this."""
        stats = np.asarray(stats, dtype=np.float64)
        rates = np.asarray(rates, dtype=np.float64)
        if stats.ndim != 2 or stats.shape[1] != 4:
            raise ValueError(f"stats must be (n, 4), got {stats.shape}")
        if rates.shape != (stats.shape[0], 3):
            raise ValueError(f"rates must be {(stats.shape[0], 3)}, got {rates.shape}")

        totals = stats.sum(axis=1)
        lam = rates.sum(axis=1)
        valid = (totals > 0) & np.all(rates > 0.0, axis=1)

        out = np.zeros(stats.shape[0], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_any = np.log(-np.expm1(-lam))
            n_mut = stats[:, 1:].sum(axis=1)
            terms = -stats[:, 0] * lam + n_mut * (log_any - np.log(lam))
            for j in range(3):
                contrib = stats[:, j + 1] * np.log(rates[:, j])
                terms = terms + np.where(stats[:, j + 1] == 0.0, 0.0, contrib)
        out[valid] = terms[valid]
        out[(~valid) & (totals > 0)] = -np.inf
        return out

    @staticmethod
    def observed_fraction(rates: np.ndarray) -> np.ndarray:
        """``1 - exp(-L)``: the fraction of sites that look mutated.

        The quantity the binomial mistakes for the rate itself.
        """
        lam = np.asarray(rates, dtype=np.float64).sum(axis=-1)
        return -np.expm1(-lam)

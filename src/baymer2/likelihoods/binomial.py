"""Multinomial likelihood on pooled counts -- the legacy Baymer model.

Every leaf under a node shares that node's rate vector, so the product
of per-leaf multinomials collapses onto the node's four pooled totals::

    log L_v = S[v,0] * log1p(-sum_j p_j) + sum_{j=1..3} S[v,j] * log(p_j)

Legacy Baymer computed the same quantity as
``np.sum(leaf_counts[a::4])`` for each ``a``, on every proposal, which is
O(4**(max_mer - layer)) per evaluation. Here ``S`` arrives precomputed
from :mod:`baymer2.counts`, so evaluation is O(1).

One deliberate numerical difference: the reference term uses
``log1p(-sum(p))`` where legacy used ``log(1.0 - np.sum(p_vec))``.
Mutation probabilities are ~1e-3 to 1e-8, and ``1.0 - x`` for small ``x``
discards the low bits of ``x`` before the log ever sees them. ``log1p``
is accurate across the whole range. The two agree to ~1e-16 relative,
far below Monte Carlo error, but they are not bit-identical -- so tests
compare with a tolerance and pin the improvement rather than the legacy
rounding.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from baymer2.likelihoods.base import MODES, LikelihoodSpec

__all__ = ["BinomialLikelihood", "node_loglik", "layer_loglik"]


@njit(cache=True, inline="always")
def node_loglik(s0: float, s1: float, s2: float, s3: float,
                p0: float, p1: float, p2: float) -> float:
    """Log-likelihood of one node. O(1) in the k-mer depth.

    Returns 0.0 for nodes with no observed opportunities, which
    contribute nothing and whose rates are unidentified.
    """
    total = s0 + s1 + s2 + s3
    if total == 0.0:
        return 0.0

    psum = p0 + p1 + p2
    # Outside the simplex the state is invalid, not merely unlikely.
    if psum >= 1.0 or p0 <= 0.0 or p1 <= 0.0 or p2 <= 0.0:
        return -np.inf

    ll = s0 * np.log1p(-psum)
    if s1 != 0.0:
        ll += s1 * np.log(p0)
    if s2 != 0.0:
        ll += s2 * np.log(p1)
    if s3 != 0.0:
        ll += s3 * np.log(p2)
    return ll


@njit(cache=True, parallel=True)
def layer_loglik(stats: np.ndarray, rates: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Per-node log-likelihoods for a whole layer.

    ``stats`` is ``(n, 4)``, ``rates`` is ``(n, 3)``, ``out`` is ``(n,)``.
    """
    n = stats.shape[0]
    for i in prange(n):
        out[i] = node_loglik(
            stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3],
            rates[i, 0], rates[i, 1], rates[i, 2],
        )
    return out


class BinomialLikelihood:
    """One multinomial trial per site; no multiplicity, no branch model.

    Appropriate for polymorphism data where the per-site mutation
    probability is small enough that recurrence is negligible. For
    substitution data across a non-trivial branch, or for hypermutable
    contexts in large cohorts, prefer ``ctmc`` or ``recurrent``.
    """

    name = "binomial"

    def spec(self, layer: int) -> LikelihoodSpec:
        return LikelihoodSpec(mode=MODES[self.name])

    def log_likelihood(self, stats: np.ndarray, rates: np.ndarray) -> np.ndarray:
        """Readable numpy reference; the kernels are checked against this."""
        stats = np.asarray(stats, dtype=np.float64)
        rates = np.asarray(rates, dtype=np.float64)
        if stats.ndim != 2 or stats.shape[1] != 4:
            raise ValueError(f"stats must be (n, 4), got {stats.shape}")
        if rates.shape != (stats.shape[0], 3):
            raise ValueError(
                f"rates must be {(stats.shape[0], 3)}, got {rates.shape}"
            )

        totals = stats.sum(axis=1)
        psum = rates.sum(axis=1)
        valid = (totals > 0) & (psum < 1.0) & np.all(rates > 0.0, axis=1)

        out = np.zeros(stats.shape[0], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = stats[:, 0] * np.log1p(-psum)
            for j in range(3):
                contrib = stats[:, j + 1] * np.log(rates[:, j])
                terms = terms + np.where(stats[:, j + 1] == 0.0, 0.0, contrib)
        out[valid] = terms[valid]
        # Empty nodes contribute nothing; out-of-simplex states are invalid.
        out[(~valid) & (totals > 0)] = -np.inf
        return out

    def total_log_likelihood(self, stats: np.ndarray, rates: np.ndarray) -> float:
        return float(self.log_likelihood(stats, rates).sum())

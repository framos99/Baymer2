"""Metropolis-within-Gibbs sampler, numba-compiled.

This is a reimplementation of legacy Baymer's sampler, not a port. The
step structure is the same -- resample parents, Gibbs the indicators,
random-walk the log-rate increments, Gibbs alpha, random-walk sigma --
but the parameterisation and several of the steps are corrected.

Parameterisation
----------------
Legacy wrote ``p_v = p_parent * exp(phi_v)`` and enforced
``sum_j p_vj < 1`` by redrawing an offending proposal up to five times
and then giving up. That makes the effective proposal a truncated normal
whose normalising constant depends on the current state and never
appears in the acceptance ratio, so the chain does not target the stated
posterior. The same pattern in ``init_set_edges`` silently kept the
*previous* parent whenever a redrawn parent fell outside the simplex,
biasing the plug-in prior toward small rates.

Here the state is the additive log-ratio (ALR) coordinate::

    eta_vj  = log(p_vj / p_vref)
    p_vj    = exp(eta_vj) / (1 + sum_k exp(eta_vk))
    p_vref  = 1          / (1 + sum_k exp(eta_vk))
    phi_vj  = eta_vj - eta_parent,j

Every real ``eta`` maps to a valid interior simplex point, so the
rejection region disappears *structurally* rather than being patched.
Because mutation probabilities are ~1e-3 to 1e-8, ``p_ref`` is within
1e-3 of one and ``phi`` here is numerically almost identical to legacy's
``phi``; the interpretation ("log fold change in rate contributed by
adding one flanking base") is unchanged.

At layer 0 there is no parent. Legacy used a flat prior on ``p`` over
the valid region; that is preserved by adding the log-Jacobian of the
ALR map, ``log p_ref + sum_j log p_j``, so the target is flat in ``p``
rather than flat in ``eta``.

Corrections relative to legacy
-----------------------------
=====  ==========================================================
Bug    Resolution
=====  ==========================================================
(a)    ``sample_new_sigmas`` arity mismatch: rewritten, exercised
       by tests whenever sigma is estimated rather than fixed.
(b)    Layer-0 out-of-bounds write into a length-3 acceptance
       array: acceptance is counted per (node, allele).
(c)    Alpha's exact Beta-Gibbs draw was filtered through a
       Metropolis ratio, wrongly rejecting ~21% of valid draws.
       The draw is now accepted unconditionally, as a Gibbs step
       must be.
(d)    Truncated proposals: removed by the ALR reparameterisation.
(e)    ``zero_contexts_bool`` leaking across siblings: replaced by
       a per-node boolean mask derived from the count table.
(f)    ``accepted_array[i + sub_index]`` index error and prange
       write race: each node owns its own counters.
(g)    Dead ``phi_alpha = 0.5`` overwrite: gone.
(h)    Thread-count-dependent results: see :mod:`baymer2.rng`.
(i)    ``set_sigma: 0`` misread as "do not fix": ``None`` is the
       only sentinel for "estimate it".
=====  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from baymer2.likelihoods.base import MODES
from baymer2.likelihoods.binomial import node_loglik
from baymer2.likelihoods.ctmc import ctmc_node_loglik
from baymer2.likelihoods.recurrent import recurrent_node_loglik
if TYPE_CHECKING:  # pragma: no cover
    from baymer2.io.traces import TraceStore

from baymer2.rng import (
    PURPOSE_ALPHA,
    PURPOSE_INDICATOR,
    PURPOSE_PARENT_PICK,
    PURPOSE_PHI_ACCEPT,
    PURPOSE_PHI_PROPOSAL,
    PURPOSE_SIGMA_ACCEPT,
    PURPOSE_SIGMA_PROPOSAL,
    bernoulli,
    make_key,
    randint,
    standard_normal,
    uniform,
)

__all__ = [
    "JointResult",
    "LayerConfig",
    "LayerResult",
    "NumbaJointBackend",
    "NumbaMetropolisBackend",
    "rates_from_eta",
    "eta_from_rates",
]

#: Roberts & Rosenthal adaptive-scaling batch length.
BATCH = 50
#: Target acceptance rate for univariate random-walk Metropolis.
TARGET_ACCEPT = 0.44
#: How often (in thinned samples) to write back and evict trace pages.
#: Small enough to bound resident memory, large enough that the
#: writeback cost stays negligible against a sampling sweep.
TRACE_RELEASE_EVERY = 50


# ==========================================================================
# ALR transform
# ==========================================================================


@njit(inline="always", cache=True)
def _rates_from_eta_row(e0, e1, e2):
    """ALR coordinates to (p1, p2, p3). Overflow-safe.

    Exact in real arithmetic: every real ``eta`` maps strictly inside the
    simplex. In float64 the map *saturates* once one coordinate exceeds
    the others by more than ~37, because ``exp(-delta)`` then underflows
    against 1 and the dominant rate rounds to exactly 1.0.

    That boundary is unreachable in practice -- mutation rates of 1e-8
    to 1e-3 put ``eta`` between about -20 and -5 -- and it is safe when
    reached, because :func:`~baymer2.likelihoods.binomial.node_loglik`
    returns ``-inf`` for ``sum(p) >= 1`` and the proposal is rejected.
    Clamping here instead would silently distort the target for no
    practical gain.
    """
    m = 0.0
    if e0 > m:
        m = e0
    if e1 > m:
        m = e1
    if e2 > m:
        m = e2
    # exp(-m) factored out so the largest exponent is exp(0) = 1.
    r = np.exp(-m)
    x0 = np.exp(e0 - m)
    x1 = np.exp(e1 - m)
    x2 = np.exp(e2 - m)
    denom = r + x0 + x1 + x2
    return x0 / denom, x1 / denom, x2 / denom


def rates_from_eta(eta: np.ndarray) -> np.ndarray:
    """ALR coordinates to alternate-allele rates, ``(n, 3)`` -> ``(n, 3)``."""
    eta = np.asarray(eta, dtype=np.float64)
    m = np.maximum(eta.max(axis=1, keepdims=True), 0.0)
    x = np.exp(eta - m)
    denom = np.exp(-m)[:, 0] + x.sum(axis=1)
    return x / denom[:, None]


def eta_from_rates(rates: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rates_from_eta`."""
    rates = np.asarray(rates, dtype=np.float64)
    ref = 1.0 - rates.sum(axis=1, keepdims=True)
    if np.any(ref <= 0):
        raise ValueError("rates must sum to strictly less than 1")
    if np.any(rates <= 0):
        raise ValueError("rates must be strictly positive")
    return np.log(rates) - np.log(ref)


@njit(inline="always", cache=True)
def _log_jacobian(p0, p1, p2):
    """log|d p / d eta| for the ALR map: log p_ref + sum_j log p_j.

    Added at layer 0 so a flat prior on the simplex stays flat in ``p``
    after reparameterisation, matching legacy's layer-0 target.
    """
    pref = 1.0 - p0 - p1 - p2
    if pref <= 0.0:
        return -np.inf
    return np.log(pref) + np.log(p0) + np.log(p1) + np.log(p2)


# ==========================================================================
# Priors
# ==========================================================================


@njit(inline="always", cache=True)
def _log_phi_prior(phi, indicator, sigma, c):
    """Spike-and-slab Gaussian log-density, up to an additive constant."""
    s = sigma if indicator == 1 else c * sigma
    return -np.log(s) - (phi * phi) / (2.0 * s * s)


@njit(inline="always", cache=True)
def _node_target(mode, s0, s1, s2, s3, p0, p1, p2, is_root):
    """Likelihood plus any parameterisation Jacobian for one node.

    Only the node-local modes route through here. ``ctmc`` is
    quartet-local and has its own sampling path.
    """
    if mode == 2:  # recurrent
        ll = recurrent_node_loglik(s0, s1, s2, s3, p0, p1, p2)
    else:
        ll = node_loglik(s0, s1, s2, s3, p0, p1, p2)
    if is_root and ll != -np.inf:
        ll += _log_jacobian(p0, p1, p2)
    return ll


@njit(inline="always", cache=True)
def _observed_loglik(mode, s0, s1, s2, s3, p0, p1, p2):
    """Reported likelihood, without any reparameterisation Jacobian."""
    if mode == 2:
        return recurrent_node_loglik(s0, s1, s2, s3, p0, p1, p2)
    return node_loglik(s0, s1, s2, s3, p0, p1, p2)


# ==========================================================================
# Gibbs: indicators
# ==========================================================================


@njit(parallel=True, cache=True)
def sample_indicators(phi, indicator, zero_mask, alpha, sigma, c, key):
    """Exact Gibbs draw of the spike-and-slab indicators.

    ``P(z=1 | phi, alpha, sigma)`` in logit form::

        logit = log(alpha/(1-alpha)) + log(c)
                + phi**2 / 2 * (1/(c*sigma)**2 - 1/sigma**2)

    The ``log c`` term pulls toward the spike at ``phi = 0``; the
    quadratic term pulls toward the slab as ``|phi|`` grows.
    """
    n = phi.shape[0]
    if alpha <= 0.0:
        logit_alpha = -np.inf
    elif alpha >= 1.0:
        logit_alpha = np.inf
    else:
        logit_alpha = np.log(alpha) - np.log1p(-alpha)

    inv_slab = 1.0 / (sigma * sigma)
    inv_spike = 1.0 / (c * c * sigma * sigma)
    log_c = np.log(c)

    for i in prange(n):
        if zero_mask[i]:
            continue
        for j in range(3):
            ph = phi[i, j]
            logit = logit_alpha + log_c + 0.5 * ph * ph * (inv_spike - inv_slab)
            if logit >= 0.0:
                prob = 1.0 / (1.0 + np.exp(-logit))
            else:
                e = np.exp(logit)
                prob = e / (1.0 + e)
            indicator[i, j] = bernoulli(prob, key, i, j, PURPOSE_INDICATOR)


# ==========================================================================
# Metropolis: phi
# ==========================================================================


@njit(parallel=True, cache=True)
def sample_phi(
    mode, stats, phi, eta, parent_eta, rates, indicator, zero_mask,
    log_scale, accepted, node_loglik_out, sigma, c, is_root, key,
):
    """Univariate random-walk Metropolis on every phi, in ALR space.

    Each node is updated independently and owns its own accumulators, so
    the ``prange`` carries no cross-node dependency and no write race --
    unlike legacy, where the acceptance counter was indexed
    ``[i + sub_index]`` and written concurrently.
    """
    n = phi.shape[0]
    for i in prange(n):
        if zero_mask[i]:
            continue

        s0 = stats[i, 0]
        s1 = stats[i, 1]
        s2 = stats[i, 2]
        s3 = stats[i, 3]

        e0 = eta[i, 0]
        e1 = eta[i, 1]
        e2 = eta[i, 2]
        p0, p1, p2 = _rates_from_eta_row(e0, e1, e2)
        cur_ll = _node_target(mode, s0, s1, s2, s3, p0, p1, p2, is_root)

        for j in range(3):
            step = np.exp(log_scale[i, j]) * standard_normal(
                key, i, j, PURPOSE_PHI_PROPOSAL
            )
            prop_phi = phi[i, j] + step
            prop_eta = parent_eta[i, j] + prop_phi

            if j == 0:
                q0, q1, q2 = _rates_from_eta_row(prop_eta, e1, e2)
            elif j == 1:
                q0, q1, q2 = _rates_from_eta_row(e0, prop_eta, e2)
            else:
                q0, q1, q2 = _rates_from_eta_row(e0, e1, prop_eta)

            prop_ll = _node_target(mode, s0, s1, s2, s3, q0, q1, q2, is_root)

            delta = prop_ll - cur_ll
            if not is_root:
                delta += _log_phi_prior(prop_phi, indicator[i, j], sigma, c)
                delta -= _log_phi_prior(phi[i, j], indicator[i, j], sigma, c)

            accept = False
            if delta >= 0.0:
                accept = True
            elif delta > -700.0:
                if uniform(key, i, j, PURPOSE_PHI_ACCEPT) < np.exp(delta):
                    accept = True

            if accept:
                phi[i, j] = prop_phi
                if j == 0:
                    e0 = prop_eta
                elif j == 1:
                    e1 = prop_eta
                else:
                    e2 = prop_eta
                p0, p1, p2 = q0, q1, q2
                cur_ll = prop_ll
                accepted[i, j] += 1

        eta[i, 0] = e0
        eta[i, 1] = e1
        eta[i, 2] = e2
        rates[i, 0] = p0
        rates[i, 1] = p1
        rates[i, 2] = p2
        node_loglik_out[i] = _observed_loglik(mode, s0, s1, s2, s3, p0, p1, p2)


# ==========================================================================
# Metropolis: phi, quartet-local (CTMC)
# ==========================================================================


@njit(parallel=True, cache=True)
def sample_phi_ctmc(
    stats, phi, eta, parent_eta, rates, indicator, zero_mask,
    log_scale, accepted, node_loglik_out, sigma, c, is_root,
    q_src, q_slot, slot_base, branch_length, members, sizes, key,
):
    """Random-walk Metropolis where the likelihood spans a whole quartet.

    Under the CTMC a node's likelihood reads rates from the three
    contexts that share its flanks, so changing one node changes the
    likelihood of all of them. The acceptance ratio must therefore be
    computed over the group, and the parallel loop must run over groups
    rather than nodes -- two members updated concurrently would race on
    the shared generator.

    Groups are closed under the source relation, so they are mutually
    independent and safe to run in parallel.
    """
    n_groups = members.shape[0]
    for g in prange(n_groups):
        q = np.empty((4, 4))
        pm = np.empty((4, 4))
        size = sizes[g]

        current = 0.0
        for m in range(size):
            current += ctmc_node_loglik(
                members[g, m], stats, rates, q_src, q_slot, slot_base,
                branch_length, q, pm,
            )

        for m in range(size):
            i = members[g, m]
            if zero_mask[i]:
                continue
            for j in range(3):
                step = np.exp(log_scale[i, j]) * standard_normal(
                    key, i, j, PURPOSE_PHI_PROPOSAL
                )
                prop_phi = phi[i, j] + step

                old_eta = eta[i, j]
                old_r0 = rates[i, 0]
                old_r1 = rates[i, 1]
                old_r2 = rates[i, 2]

                eta[i, j] = parent_eta[i, j] + prop_phi
                n0, n1, n2 = _rates_from_eta_row(eta[i, 0], eta[i, 1], eta[i, 2])
                rates[i, 0] = n0
                rates[i, 1] = n1
                rates[i, 2] = n2

                proposed = 0.0
                for mk in range(size):
                    proposed += ctmc_node_loglik(
                        members[g, mk], stats, rates, q_src, q_slot, slot_base,
                        branch_length, q, pm,
                    )

                delta = proposed - current
                if is_root:
                    delta += _log_jacobian(n0, n1, n2)
                    delta -= _log_jacobian(old_r0, old_r1, old_r2)
                else:
                    delta += _log_phi_prior(prop_phi, indicator[i, j], sigma, c)
                    delta -= _log_phi_prior(phi[i, j], indicator[i, j], sigma, c)

                accept = False
                if delta >= 0.0:
                    accept = True
                elif delta > -700.0:
                    if uniform(key, i, j, PURPOSE_PHI_ACCEPT) < np.exp(delta):
                        accept = True

                if accept:
                    phi[i, j] = prop_phi
                    current = proposed
                    accepted[i, j] += 1
                else:
                    eta[i, j] = old_eta
                    rates[i, 0] = old_r0
                    rates[i, 1] = old_r1
                    rates[i, 2] = old_r2

        for m in range(size):
            i = members[g, m]
            node_loglik_out[i] = ctmc_node_loglik(
                i, stats, rates, q_src, q_slot, slot_base, branch_length, q, pm,
            )


# ==========================================================================
# Cut linkage: resample parents
# ==========================================================================


@njit(parallel=True, cache=True)
def resample_parents(parent_samples, phi, eta, parent_eta, rates, branching, key):
    """Draw a parent state from the parent layer's stored posterior.

    ``parent_samples`` is ``(n_samples, n_parents, 3)`` of parent ALR
    coordinates. Each parent node picks its own sample index, matching
    legacy. Unlike legacy there is no validity test and no silent
    fallback to the previous parent: every ALR value is admissible.
    """
    n_samples, n_parents, _ = parent_samples.shape
    for parent in prange(n_parents):
        r = randint(n_samples, key, parent, 0, PURPOSE_PARENT_PICK)
        for child in range(branching):
            i = parent * branching + child
            for j in range(3):
                pe = parent_samples[r, parent, j]
                parent_eta[i, j] = pe
                eta[i, j] = pe + phi[i, j]
            p0, p1, p2 = _rates_from_eta_row(eta[i, 0], eta[i, 1], eta[i, 2])
            rates[i, 0] = p0
            rates[i, 1] = p1
            rates[i, 2] = p2


@njit(parallel=True, cache=True)
def recompute_loglik(stats, rates, zero_mask, out):
    n = stats.shape[0]
    for i in prange(n):
        if zero_mask[i]:
            out[i] = 0.0
        else:
            out[i] = node_loglik(
                stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3],
                rates[i, 0], rates[i, 1], rates[i, 2],
            )


# ==========================================================================
# Gibbs: alpha
# ==========================================================================


@njit(cache=True)
def _gamma_variate(shape, key, slot):
    """Marsaglia-Tsang gamma draw, shape >= 1, scale 1.

    The rejection loop varies the RNG coordinate by attempt, so the draw
    stays a pure function of ``(key, slot)``.
    """
    d = shape - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)
    for attempt in range(1, 200):
        x = standard_normal(key, slot, attempt, PURPOSE_ALPHA)
        v = 1.0 + c * x
        if v <= 0.0:
            continue
        v = v * v * v
        u = uniform(key, slot + 1000000, attempt, PURPOSE_ALPHA)
        if np.log(u) < 0.5 * x * x + d - d * v + d * np.log(v):
            return d * v
    return d  # pathological; effectively never reached


@njit(cache=True)
def sample_alpha(indicator, zero_mask, key):
    """Exact Beta-Gibbs draw for the slab inclusion probability.

    With a Beta(1, 1) prior and Bernoulli indicators the full
    conditional is ``Beta(1 + sum z, 1 + n_active - sum z)``.

    Legacy drew from exactly this distribution and then passed it
    through ``perform_metropolis_step_numba`` against the Bernoulli
    likelihood ratio. For a Gibbs step the correct acceptance
    probability is identically one; the extra filter rejected any draw
    that moved downhill, reporting 0.79 acceptance and concentrating
    alpha at its mode. No filter here.
    """
    n = indicator.shape[0]
    total = 0
    active = 0
    for i in range(n):
        if zero_mask[i]:
            continue
        active += 3
        for j in range(3):
            total += indicator[i, j]

    a = 1.0 + total
    b = 1.0 + active - total
    x = _gamma_variate(a, key, 0)
    y = _gamma_variate(b, key, 1)
    return x / (x + y)


# ==========================================================================
# Metropolis: sigma
# ==========================================================================


@njit(parallel=True, cache=True)
def _total_phi_prior(phi, indicator, zero_mask, sigma, c):
    n = phi.shape[0]
    total = 0.0
    for i in prange(n):
        if zero_mask[i]:
            continue
        acc = 0.0
        for j in range(3):
            acc += _log_phi_prior(phi[i, j], indicator[i, j], sigma, c)
        total += acc
    return total


@njit(cache=True)
def sample_sigma(phi, indicator, zero_mask, sigma, c, log_scale, sigma_max, key):
    """Random walk on ``log sigma`` for the slab scale.

    Proposing on the log scale keeps sigma positive without a rejection
    region; the ``log sigma' - log sigma`` term is the Jacobian for a
    flat prior on sigma itself, matching legacy's improper flat prior on
    ``(0, sigma_max)``.

    Returns ``(sigma, accepted)``. Legacy's equivalent could not be
    called at all -- the call site passed 14 arguments to a 13-argument
    definition, so any attempt to estimate sigma raised ``TypeError``.
    """
    step = np.exp(log_scale) * standard_normal(key, 0, 0, PURPOSE_SIGMA_PROPOSAL)
    prop = sigma * np.exp(step)
    if prop <= 0.0 or prop >= sigma_max:
        return sigma, 0

    cur_lp = _total_phi_prior(phi, indicator, zero_mask, sigma, c)
    prop_lp = _total_phi_prior(phi, indicator, zero_mask, prop, c)
    delta = prop_lp - cur_lp + (np.log(prop) - np.log(sigma))

    if delta >= 0.0:
        return prop, 1
    if delta > -700.0 and uniform(key, 0, 0, PURPOSE_SIGMA_ACCEPT) < np.exp(delta):
        return prop, 1
    return sigma, 0


# ==========================================================================
# Adaptation
# ==========================================================================


@njit(cache=True)
def adapt_log_scales(log_scale, accepted, batch_len, batch_index):
    """Roberts & Rosenthal diminishing adaptation toward 0.44 acceptance."""
    delta = min(0.01, 1.0 / np.sqrt(batch_index))
    n = log_scale.shape[0]
    for i in range(n):
        for j in range(3):
            rate = accepted[i, j] / batch_len
            if rate > TARGET_ACCEPT:
                log_scale[i, j] += delta
            else:
                log_scale[i, j] -= delta
            accepted[i, j] = 0


# ==========================================================================
# Python-level driver
# ==========================================================================


@dataclass
class LayerConfig:
    """MCMC settings for one layer."""

    n_iterations: int = 40000
    burnin: int = 20000
    thin: int = 10
    alpha_init: float = 0.5
    sigma_init: float = 0.85
    #: ``None`` means "estimate sigma". Legacy used a falsy test, so a
    #: config saying ``set_sigma: 0`` -- a plausible way to request
    #: estimation -- was read as "do not fix" and then crashed.
    set_sigma: float | None = None
    c: float = 0.04472136
    phi_proposal_sd: float = 0.05
    sigma_proposal_sd: float = 0.1
    sigma_max: float | None = None
    seed: int = 0
    #: ``"alr"`` (default) or ``"multiplicative"``. The latter keeps
    #: legacy's phi = log(p_v / p_parent); see the module notes. It is a
    #: correct sampler for that parameterisation, not a re-creation of
    #: legacy's.
    parameterization: str = "alr"

    def __post_init__(self) -> None:
        if self.burnin >= self.n_iterations:
            raise ValueError(
                f"burnin ({self.burnin}) must be less than n_iterations "
                f"({self.n_iterations})"
            )
        if self.thin < 1:
            raise ValueError(f"thin must be >= 1, got {self.thin}")
        if not 0.0 < self.c < 1.0:
            raise ValueError(f"c must be in (0, 1), got {self.c}")
        if self.set_sigma is not None and self.set_sigma <= 0.0:
            raise ValueError(
                f"set_sigma must be positive or None, got {self.set_sigma}"
            )
        if self.parameterization not in ("alr", "multiplicative"):
            raise ValueError(
                f"parameterization must be 'alr' or 'multiplicative', got "
                f"{self.parameterization!r}"
            )

    @property
    def n_samples(self) -> int:
        return (self.n_iterations - self.burnin) // self.thin

    @property
    def effective_sigma_max(self) -> float:
        return self.sigma_max if self.sigma_max is not None else 1.0 / self.c


@dataclass
class LayerResult:
    """Thinned posterior draws for one layer."""

    layer: int
    eta: np.ndarray  # (n_samples, n_nodes, 3)
    phi: np.ndarray  # (n_samples, n_nodes, 3)
    rates: np.ndarray  # (n_samples, n_nodes, 3)
    indicator: np.ndarray  # (n_samples, n_nodes, 3) int8
    alpha: np.ndarray  # (n_samples,)
    sigma: np.ndarray  # (n_samples,)
    loglik: np.ndarray  # (n_iterations,)
    accept_rate_phi: float
    accept_rate_sigma: float
    zero_mask: np.ndarray = field(repr=False, default=None)

    @property
    def mean_rates(self) -> np.ndarray:
        return self.rates.mean(axis=0)


class NumbaMetropolisBackend:
    """Metropolis-within-Gibbs over the nested context tree."""

    name = "numba_mh"

    def __init__(self, n_threads: int | None = None) -> None:
        self.n_threads = n_threads

    def run_layer(
        self,
        layer: int,
        stats: np.ndarray,
        zero_mask: np.ndarray,
        config: LayerConfig,
        *,
        parent_samples: np.ndarray | None = None,
        mode: int = MODES["binomial"],
        init_rates: np.ndarray | None = None,
        progress: bool = False,
        trace_store: "TraceStore | None" = None,
        branching: int = 4,
        ctmc_maps: tuple | None = None,
        branch_length: float = 1.0,
    ) -> LayerResult:
        """Sample one layer.

        Parameters
        ----------
        stats:
            ``(n_nodes, 4)`` sufficient statistics, reference first.
        zero_mask:
            ``(n_nodes,)`` boolean, ``True`` where a node has no data.
        parent_samples:
            ``(n_parent_samples, n_nodes // 4, 3)`` of parent ALR
            coordinates, or ``None`` at layer 0.
        trace_store:
            When given, thinned draws are allocated as memory-mapped
            ``.npy`` files rather than in RAM. Required in practice
            beyond about ``max_mer=7``, where the in-memory traces run
            to several GB per layer.
        """
        n_nodes = stats.shape[0]
        is_root = parent_samples is None
        if mode == MODES["ctmc"] and ctmc_maps is None:
            raise ValueError(
                "the ctmc likelihood is quartet-local and needs its index "
                "maps; pass ctmc_maps=(q_src, q_slot, slot_base, members, "
                "sizes) built from baymer2.likelihoods.ctmc"
            )
        if not is_root and parent_samples.shape[1] * branching != n_nodes:
            raise ValueError(
                f"parent_samples has {parent_samples.shape[1]} parents, which "
                f"with branching={branching} cannot produce {n_nodes} nodes "
                f"(expected {n_nodes // branching})"
            )

        stats = np.ascontiguousarray(stats, dtype=np.float64)
        zero_mask = np.ascontiguousarray(zero_mask, dtype=np.bool_)

        # --- state ---------------------------------------------------
        phi = np.zeros((n_nodes, 3), dtype=np.float64)
        parent_eta = np.zeros((n_nodes, 3), dtype=np.float64)
        if init_rates is None:
            init_rates = self._initial_rates(stats, zero_mask)
        if config.parameterization == "multiplicative":
            # The quantity that crosses a layer boundary here is the rate
            # itself, not an ALR coordinate, so `eta` aliases `rates` and
            # the trace records rates in both slots. Without this the next
            # layer receives zeros as its parent and every proposal is
            # rejected.
            rates = np.ascontiguousarray(init_rates, dtype=np.float64)
            eta = rates
            if is_root:
                phi[:] = rates
        else:
            eta = eta_from_rates(init_rates)
            if is_root:
                phi[:] = eta
            rates = rates_from_eta(eta)

        indicator = np.zeros((n_nodes, 3), dtype=np.int8)
        if config.parameterization == "multiplicative" and is_root:
            # At the root of the multiplicative parameterisation phi *is*
            # the rate, so an absolute proposal scale tuned for a log
            # quantity is wildly wrong: a step of 0.02 on a rate of 1e-3
            # is rejected essentially always, and diminishing adaptation
            # needs tens of thousands of iterations to climb down. Read
            # phi_proposal_sd as a fraction of the rate instead. Legacy
            # handled this by hardcoding phi_sigma: 0.00005 for layer 0
            # in every shipped config.
            log_scale = np.log(
                np.maximum(config.phi_proposal_sd * init_rates, 1e-12)
            )
        else:
            log_scale = np.full((n_nodes, 3), np.log(config.phi_proposal_sd))
        accepted = np.zeros((n_nodes, 3), dtype=np.int64)
        accepted_total = np.zeros((n_nodes, 3), dtype=np.int64)
        node_ll = np.zeros(n_nodes, dtype=np.float64)

        alpha = config.alpha_init
        sigma = config.set_sigma if config.set_sigma is not None else config.sigma_init
        estimate_sigma = config.set_sigma is None
        sigma_log_scale = np.log(config.sigma_proposal_sd)
        sigma_accepted = 0

        # --- output --------------------------------------------------
        # Either RAM or memory-mapped .npy; both behave as ndarrays for
        # the assignment and reduction this code does.
        n_samples = config.n_samples
        if trace_store is not None:
            arrays = trace_store.allocate_layer(layer, n_samples, n_nodes)
            out_eta = arrays["eta"]
            out_phi = arrays["phi"]
            out_rates = arrays["rates"]
            out_ind = arrays["indicator"]
            out_alpha = arrays["alpha"]
            out_sigma = arrays["sigma"]
            trace_ll = trace_store.allocate_loglik(layer, config.n_iterations)
        else:
            out_eta = np.zeros((n_samples, n_nodes, 3), dtype=np.float32)
            out_phi = np.zeros((n_samples, n_nodes, 3), dtype=np.float32)
            out_rates = np.zeros((n_samples, n_nodes, 3), dtype=np.float32)
            out_ind = np.zeros((n_samples, n_nodes, 3), dtype=np.int8)
            out_alpha = np.zeros(n_samples, dtype=np.float64)
            out_sigma = np.zeros(n_samples, dtype=np.float64)
            trace_ll = np.zeros(config.n_iterations, dtype=np.float64)

        seed = np.uint64(config.seed)
        sample_i = 0
        adaptation_frozen = False

        for it in range(1, config.n_iterations + 1):
            key = make_key(seed, np.uint64(layer), np.uint64(it))
            key = np.uint64(key)

            if not is_root:
                if config.parameterization == "multiplicative":
                    resample_parents_multiplicative(
                        parent_samples, phi, parent_eta, rates, branching, key
                    )
                else:
                    resample_parents(
                        parent_samples, phi, eta, parent_eta, rates, branching,
                        key,
                    )
                sample_indicators(
                    phi, indicator, zero_mask, alpha, sigma, config.c, key
                )

            if config.parameterization == "multiplicative":
                sample_phi_multiplicative(
                    mode, stats, phi, parent_eta, rates, indicator, zero_mask,
                    log_scale, accepted, node_ll, sigma, config.c, is_root, key,
                )
            elif ctmc_maps is None:
                sample_phi(
                    mode, stats, phi, eta, parent_eta, rates, indicator,
                    zero_mask, log_scale, accepted, node_ll, sigma, config.c,
                    is_root, key,
                )
            else:
                q_src, q_slot, slot_base, members, sizes = ctmc_maps
                sample_phi_ctmc(
                    stats, phi, eta, parent_eta, rates, indicator, zero_mask,
                    log_scale, accepted, node_ll, sigma, config.c, is_root,
                    q_src, q_slot, slot_base, branch_length, members, sizes, key,
                )

            if not is_root:
                alpha = sample_alpha(indicator, zero_mask, key)
                if estimate_sigma:
                    sigma, acc = sample_sigma(
                        phi, indicator, zero_mask, sigma, config.c,
                        sigma_log_scale, config.effective_sigma_max, key,
                    )
                    sigma_accepted += acc

            trace_ll[it - 1] = node_ll.sum()

            # Diminishing adaptation, frozen once burn-in ends so the
            # retained draws come from a time-homogeneous chain.
            if it % BATCH == 0:
                accepted_total += accepted
                if not adaptation_frozen:
                    adapt_log_scales(log_scale, accepted, BATCH, it // BATCH)
                else:
                    accepted[:] = 0
            if it >= config.burnin and not adaptation_frozen:
                adaptation_frozen = True

            if it > config.burnin and (it - config.burnin) % config.thin == 0:
                if sample_i < n_samples:
                    out_eta[sample_i] = eta
                    out_phi[sample_i] = phi
                    out_rates[sample_i] = rates
                    out_ind[sample_i] = indicator
                    out_alpha[sample_i] = alpha
                    out_sigma[sample_i] = sigma
                    sample_i += 1
                    # Write back and evict periodically, otherwise the
                    # whole layer's trace accumulates in resident memory
                    # and the disk backing buys nothing.
                    if (
                        trace_store is not None
                        and sample_i % TRACE_RELEASE_EVERY == 0
                    ):
                        trace_store.release_layer_pages(layer)

            if progress and it % max(1, config.n_iterations // 10) == 0:
                print(f"  layer {layer}: {it}/{config.n_iterations}")

        if trace_store is not None:
            trace_store.flush_layer(layer)
            final = trace_store.truncate_layer(layer, sample_i, n_samples)
            out_eta = final["eta"]
            out_phi = final["phi"]
            out_rates = final["rates"]
            out_ind = final["indicator"]
            out_alpha = final["alpha"]
            out_sigma = final["sigma"]
            sample_i = out_eta.shape[0]

        n_active = int((~zero_mask).sum()) * 3
        denom = max(1, n_active * config.n_iterations)
        return LayerResult(
            layer=layer,
            eta=out_eta[:sample_i],
            phi=out_phi[:sample_i],
            rates=out_rates[:sample_i],
            indicator=out_ind[:sample_i],
            alpha=out_alpha[:sample_i],
            sigma=out_sigma[:sample_i],
            loglik=trace_ll,
            accept_rate_phi=float(accepted_total.sum()) / denom,
            accept_rate_sigma=(
                sigma_accepted / config.n_iterations if estimate_sigma else float("nan")
            ),
            zero_mask=zero_mask,
        )

    @staticmethod
    def _initial_rates(stats: np.ndarray, zero_mask: np.ndarray) -> np.ndarray:
        """Empirical rates, floored away from zero so ALR stays finite."""
        totals = stats.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rates = np.where(totals > 0, stats[:, 1:] / np.maximum(totals, 1.0), 0.0)
        # A context with opportunities but no observed mutations is common;
        # start it at half a pseudo-count rather than at the boundary.
        floor = np.where(totals > 0, 0.5 / np.maximum(totals, 1.0), 1e-6)
        rates = np.maximum(rates, floor)
        rates[zero_mask] = 1e-6
        # Keep well inside the simplex.
        excess = rates.sum(axis=1, keepdims=True)
        too_big = excess >= 0.5
        rates = np.where(too_big, rates * (0.5 / np.maximum(excess, 1e-12)), rates)
        return rates


# ==========================================================================
# Joint linkage: all layers in one chain
# ==========================================================================
#
# The cut posterior gives every layer its own likelihood on pooled counts,
# which means the same observations enter the target once per layer. That
# is deliberate -- it is what makes the layers separable and bounds memory
# to two at a time -- but it is not a joint posterior over the tree.
#
# Under joint linkage the data enter exactly once, at the leaves:
#
#     log pi = sum_leaves log Multinomial(n_u | p_u)
#              + sum_{v, j} log p(phi_vj | z_vj, sigma_l)
#              + sum log p(z | alpha_l)
#
# with eta_u = eta_root + sum of phi along the path to u. Changing one phi
# therefore moves every leaf beneath it, so an acceptance ratio has to be
# computed over that whole subtree rather than one node.
#
# What makes this affordable is the heap layout: node v at layer l owns
# exactly the contiguous leaf range [v*span, (v+1)*span) where
# span = branching**(n_layers-1-l). Nodes within a layer own disjoint
# ranges, so a layer can still be swept in parallel.
#
# Cost is O(branching**(n_layers-1)) per layer sweep regardless of layer --
# shallow layers have few nodes but wide subtrees. Practical to about
# max_mer 7; beyond that prefer cut.


@njit(parallel=True, cache=True)
def sample_phi_joint(
    span, stats_leaf, eta_leaf, leaf_loglik, phi, parent_eta, indicator,
    zero_mask_layer, log_scale, accepted, sigma, c, is_root, key,
):
    """Random-walk Metropolis on one layer, scored over each node's subtree."""
    n_nodes = phi.shape[0]
    for v in prange(n_nodes):
        if zero_mask_layer[v]:
            continue
        start = v * span
        stop = start + span

        current = 0.0
        for leaf in range(start, stop):
            current += leaf_loglik[leaf]

        for j in range(3):
            step = np.exp(log_scale[v, j]) * standard_normal(
                key, v, j, PURPOSE_PHI_PROPOSAL
            )
            proposed = 0.0
            for leaf in range(start, stop):
                e0 = eta_leaf[leaf, 0]
                e1 = eta_leaf[leaf, 1]
                e2 = eta_leaf[leaf, 2]
                if j == 0:
                    e0 += step
                elif j == 1:
                    e1 += step
                else:
                    e2 += step
                p0, p1, p2 = _rates_from_eta_row(e0, e1, e2)
                ll = node_loglik(
                    stats_leaf[leaf, 0], stats_leaf[leaf, 1],
                    stats_leaf[leaf, 2], stats_leaf[leaf, 3], p0, p1, p2,
                )
                if is_root and ll != -np.inf:
                    ll += _log_jacobian(p0, p1, p2)
                proposed += ll

            delta = proposed - current
            if not is_root:
                new_phi = phi[v, j] + step
                delta += _log_phi_prior(new_phi, indicator[v, j], sigma, c)
                delta -= _log_phi_prior(phi[v, j], indicator[v, j], sigma, c)

            accept = False
            if delta >= 0.0:
                accept = True
            elif delta > -700.0:
                if uniform(key, v, j, PURPOSE_PHI_ACCEPT) < np.exp(delta):
                    accept = True

            if accept:
                phi[v, j] += step
                for leaf in range(start, stop):
                    eta_leaf[leaf, j] += step
                    p0, p1, p2 = _rates_from_eta_row(
                        eta_leaf[leaf, 0], eta_leaf[leaf, 1], eta_leaf[leaf, 2]
                    )
                    ll = node_loglik(
                        stats_leaf[leaf, 0], stats_leaf[leaf, 1],
                        stats_leaf[leaf, 2], stats_leaf[leaf, 3], p0, p1, p2,
                    )
                    if is_root and ll != -np.inf:
                        ll += _log_jacobian(p0, p1, p2)
                    leaf_loglik[leaf] = ll
                current = proposed
                accepted[v, j] += 1


@dataclass
class JointResult:
    """Posterior draws from a single chain over the whole tree."""

    layers: list[LayerResult]
    loglik: np.ndarray
    elapsed_seconds: float = 0.0


class NumbaJointBackend:
    """Sample every layer in one chain, with the data entering only once.

    Contrast with :class:`NumbaMetropolisBackend`, which runs the cut
    posterior: there each layer carries its own likelihood on pooled
    counts, so observations are reused across layers and information
    never flows from deep layers back to shallow ones. Here the
    likelihood lives only at the leaves and every layer is a parameter of
    one target, so a deep context can inform its ancestors.

    The price is cost and memory: a proposal at layer ``l`` rescores that
    node's entire leaf subtree, and all layers stay resident. Practical
    to roughly ``max_mer=7``.
    """

    name = "numba_joint"

    def run(
        self,
        table,
        configs: "list[LayerConfig]",
        *,
        progress: bool = False,
    ) -> JointResult:
        import time

        tree = table.tree
        n_layers = tree.n_layers
        branching = tree.branching
        stats_leaf = np.ascontiguousarray(
            table.layer_stats(n_layers - 1), dtype=np.float64
        )
        zero_leaf = np.ascontiguousarray(table.zero_mask(n_layers - 1), dtype=np.bool_)

        base = configs[0]
        n_iterations = base.n_iterations
        burnin = base.burnin
        thin = base.thin

        # --- state, one entry per layer -----------------------------
        phi = [np.zeros((tree.layer_size(l), 3)) for l in range(n_layers)]
        indicator = [
            np.zeros((tree.layer_size(l), 3), dtype=np.int8) for l in range(n_layers)
        ]
        log_scale = [
            np.full((tree.layer_size(l), 3), np.log(configs[l].phi_proposal_sd))
            for l in range(n_layers)
        ]
        accepted = [
            np.zeros((tree.layer_size(l), 3), dtype=np.int64) for l in range(n_layers)
        ]
        accepted_total = [
            np.zeros((tree.layer_size(l), 3), dtype=np.int64) for l in range(n_layers)
        ]
        zero_layer = [
            np.ascontiguousarray(table.zero_mask(l), dtype=np.bool_)
            for l in range(n_layers)
        ]
        alpha = [c.alpha_init for c in configs]
        sigma = [
            c.set_sigma if c.set_sigma is not None else c.sigma_init for c in configs
        ]
        estimate_sigma = [c.set_sigma is None for c in configs]

        # Root initialised from the pooled empirical rate; deeper layers
        # start at phi = 0 so every leaf begins at the root's rate.
        root_rates = NumbaMetropolisBackend._initial_rates(
            table.layer_stats(0), table.zero_mask(0)
        )
        phi[0][:] = eta_from_rates(root_rates)
        eta_leaf = np.repeat(phi[0], branching ** (n_layers - 1), axis=0).copy()

        leaf_loglik = np.zeros(stats_leaf.shape[0])
        rates_leaf = rates_from_eta(eta_leaf)
        recompute_loglik(stats_leaf, rates_leaf, zero_leaf, leaf_loglik)

        n_samples = (n_iterations - burnin) // thin
        out_rates = [
            np.zeros((n_samples, tree.layer_size(l), 3), dtype=np.float32)
            for l in range(n_layers)
        ]
        out_phi = [
            np.zeros((n_samples, tree.layer_size(l), 3), dtype=np.float32)
            for l in range(n_layers)
        ]
        trace = np.zeros(n_iterations)
        seed = np.uint64(base.seed)
        sample_i = 0
        frozen = False

        started = time.time()
        for it in range(1, n_iterations + 1):
            for layer in range(n_layers):
                key = np.uint64(make_key(seed, np.uint64(layer), np.uint64(it)))
                span = branching ** (n_layers - 1 - layer)
                is_root = layer == 0

                if not is_root:
                    sample_indicators(
                        phi[layer], indicator[layer], zero_layer[layer],
                        alpha[layer], sigma[layer], configs[layer].c, key,
                    )
                sample_phi_joint(
                    span, stats_leaf, eta_leaf, leaf_loglik, phi[layer],
                    phi[layer], indicator[layer], zero_layer[layer],
                    log_scale[layer], accepted[layer], sigma[layer],
                    configs[layer].c, is_root, key,
                )
                if not is_root:
                    alpha[layer] = sample_alpha(
                        indicator[layer], zero_layer[layer], key
                    )
                    if estimate_sigma[layer]:
                        sigma[layer], _ = sample_sigma(
                            phi[layer], indicator[layer], zero_layer[layer],
                            sigma[layer], configs[layer].c,
                            np.log(configs[layer].sigma_proposal_sd),
                            configs[layer].effective_sigma_max, key,
                        )

            trace[it - 1] = leaf_loglik.sum()

            if it % BATCH == 0:
                for layer in range(n_layers):
                    accepted_total[layer] += accepted[layer]
                    if not frozen:
                        adapt_log_scales(
                            log_scale[layer], accepted[layer], BATCH, it // BATCH
                        )
                    else:
                        accepted[layer][:] = 0
            if it >= burnin and not frozen:
                frozen = True

            if it > burnin and (it - burnin) % thin == 0 and sample_i < n_samples:
                # Rates at every layer follow from the accumulated phi.
                running = phi[0].copy()
                for layer in range(n_layers):
                    if layer > 0:
                        running = np.repeat(running, branching, axis=0) + phi[layer]
                    out_rates[layer][sample_i] = rates_from_eta(running)
                    out_phi[layer][sample_i] = phi[layer]
                sample_i += 1

            if progress and it % max(1, n_iterations // 10) == 0:
                print(f"  joint: {it}/{n_iterations}")

        results = []
        for layer in range(n_layers):
            n_active = int((~zero_layer[layer]).sum()) * 3
            denom = max(1, n_active * n_iterations)
            results.append(LayerResult(
                layer=layer,
                eta=out_rates[layer][:sample_i],
                phi=out_phi[layer][:sample_i],
                rates=out_rates[layer][:sample_i],
                indicator=np.zeros((sample_i, tree.layer_size(layer), 3), dtype=np.int8),
                alpha=np.full(sample_i, alpha[layer]),
                sigma=np.full(sample_i, sigma[layer]),
                loglik=trace,
                accept_rate_phi=float(accepted_total[layer].sum()) / denom,
                accept_rate_sigma=float("nan"),
                zero_mask=zero_layer[layer],
            ))
        return JointResult(
            layers=results, loglik=trace, elapsed_seconds=time.time() - started
        )


# ==========================================================================
# Legacy multiplicative parameterisation
# ==========================================================================
#
# ALR is the default because it removes the simplex constraint outright.
# The legacy parameterisation, p_v = p_parent * exp(phi_v), is offered
# alongside it so that phi keeps its original meaning -- a log fold change
# in rate rather than in odds -- for anyone comparing against published
# Baymer phi values. The two agree to within ~1e-3 at mutation rates, but
# they are not the same quantity.
#
# This is NOT legacy's sampler. Legacy redrew an offending proposal up to
# five times before giving up, which makes the effective proposal a
# truncated normal whose normalising constant depends on the current
# state and never enters the acceptance ratio -- the chain does not
# target the stated posterior. Here a proposal is drawn once and simply
# rejected if it leaves the simplex, which is ordinary Metropolis and is
# valid. Reproducing legacy's numbers bit-for-bit requires running
# legacy, which remains on disk.


@njit(parallel=True, cache=True)
def sample_phi_multiplicative(
    mode, stats, phi, parent_rates, rates, indicator, zero_mask,
    log_scale, accepted, node_loglik_out, sigma, c, is_root, key,
):
    """Random-walk Metropolis with ``p_v = p_parent * exp(phi_v)``."""
    n = phi.shape[0]
    for i in prange(n):
        if zero_mask[i]:
            continue

        s0 = stats[i, 0]
        s1 = stats[i, 1]
        s2 = stats[i, 2]
        s3 = stats[i, 3]

        p0 = rates[i, 0]
        p1 = rates[i, 1]
        p2 = rates[i, 2]
        cur_ll = _node_target(mode, s0, s1, s2, s3, p0, p1, p2, is_root)

        for j in range(3):
            step = np.exp(log_scale[i, j]) * standard_normal(
                key, i, j, PURPOSE_PHI_PROPOSAL
            )
            prop_phi = phi[i, j] + step

            q0, q1, q2 = p0, p1, p2
            if is_root:
                # At the root phi *is* the rate, as in legacy.
                if j == 0:
                    q0 = prop_phi
                elif j == 1:
                    q1 = prop_phi
                else:
                    q2 = prop_phi
            else:
                value = parent_rates[i, j] * np.exp(prop_phi)
                if j == 0:
                    q0 = value
                elif j == 1:
                    q1 = value
                else:
                    q2 = value

            # Single-shot proposal: outside the simplex simply rejects.
            # Redrawing here is what broke detailed balance in legacy.
            if q0 <= 0.0 or q1 <= 0.0 or q2 <= 0.0 or (q0 + q1 + q2) >= 1.0:
                continue

            prop_ll = _node_target(mode, s0, s1, s2, s3, q0, q1, q2, is_root)
            delta = prop_ll - cur_ll
            if not is_root:
                delta += _log_phi_prior(prop_phi, indicator[i, j], sigma, c)
                delta -= _log_phi_prior(phi[i, j], indicator[i, j], sigma, c)

            accept = False
            if delta >= 0.0:
                accept = True
            elif delta > -700.0:
                if uniform(key, i, j, PURPOSE_PHI_ACCEPT) < np.exp(delta):
                    accept = True

            if accept:
                phi[i, j] = prop_phi
                p0, p1, p2 = q0, q1, q2
                cur_ll = prop_ll
                accepted[i, j] += 1

        rates[i, 0] = p0
        rates[i, 1] = p1
        rates[i, 2] = p2
        node_loglik_out[i] = _observed_loglik(mode, s0, s1, s2, s3, p0, p1, p2)


@njit(parallel=True, cache=True)
def resample_parents_multiplicative(
    parent_samples, phi, parent_rates, rates, branching, key
):
    """Cut-linkage parent draw for the multiplicative parameterisation."""
    n_samples, n_parents, _ = parent_samples.shape
    for parent in prange(n_parents):
        r = randint(n_samples, key, parent, 0, PURPOSE_PARENT_PICK)
        for child in range(branching):
            i = parent * branching + child
            for j in range(3):
                value = parent_samples[r, parent, j]
                parent_rates[i, j] = value
                rates[i, j] = value * np.exp(phi[i, j])

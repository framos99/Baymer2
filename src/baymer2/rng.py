"""Counter-based random numbers: reproducible regardless of thread count.

Legacy Baymer seeds numba's global generator with ``np.random.seed``,
which only touches the calling thread's state. Its sampling loops then
run under ``@njit(parallel=True)``, where each worker thread draws from
its own stream, so a run with ``-t 8`` and a run with ``-t 4`` diverge
despite an identical ``random_seed``. That makes results unreproducible
across machines and unreviewable across reruns.

The fix is to stop having *streams* at all. Every random number here is
a pure function of an explicit coordinate -- seed, layer, iteration,
node, allele, and a purpose tag -- hashed with splitmix64. Two runs with
the same seed produce identical draws whatever the thread count,
scheduling order, or even loop order, and any single draw can be
reproduced in isolation when debugging.

splitmix64 is Vigna's finaliser: fast (a handful of integer ops), passes
BigCrush, and needs no state. It is not cryptographic, which is not a
requirement here.

.. warning::

   Numba's ``cache=True`` does **not** invalidate a caller's on-disk
   cache when an ``inline="always"`` callee changes. Editing anything in
   this module can therefore leave stale machine code behind in other
   modules' ``__pycache__``. If a change here appears to have no effect,
   clear ``*.nbi``/``*.nbc`` before concluding anything::

       find . -name '__pycache__' -type d -exec rm -rf {} +

   Test kernels in this project deliberately omit ``cache=True`` so they
   always recompile against current source.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numba.types import float64, int64, uint64

__all__ = [
    "PURPOSE_PHI_PROPOSAL",
    "PURPOSE_PHI_ACCEPT",
    "PURPOSE_INDICATOR",
    "PURPOSE_SIGMA_PROPOSAL",
    "PURPOSE_SIGMA_ACCEPT",
    "PURPOSE_ALPHA",
    "PURPOSE_PARENT_PICK",
    "PURPOSE_INIT",
    "splitmix64",
    "make_key",
    "uniform",
    "standard_normal",
    "bernoulli",
    "randint",
]

# Purpose tags keep independent uses of the same (node, allele) coordinate
# from colliding. Append rather than renumber: changing these changes the
# draws a given seed produces.
PURPOSE_PHI_PROPOSAL = 0
PURPOSE_PHI_ACCEPT = 1
PURPOSE_INDICATOR = 2
PURPOSE_SIGMA_PROPOSAL = 3
PURPOSE_SIGMA_ACCEPT = 4
PURPOSE_ALPHA = 5
PURPOSE_PARENT_PICK = 6
PURPOSE_INIT = 7

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_SHIFT_30 = np.uint64(30)
_SHIFT_27 = np.uint64(27)
_SHIFT_31 = np.uint64(31)
_SHIFT_11 = np.uint64(11)

# 2**-53, so a 53-bit mantissa maps into the unit interval.
_TWO_POW_M53 = 1.0 / 9007199254740992.0

_TWO_PI = 2.0 * np.pi


# Unsignedness has to be enforced twice, and both are load-bearing.
#
# 1. The explicit signatures handle the *Python* boundary. `make_key`
#    returns a value that routinely exceeds 2**63; Python receives it as
#    a plain int, and an untyped kernel would infer int64 and raise
#    OverflowError.
#
# 2. The `np.uint64(...)` casts inside each body handle the *njit* call
#    sites. Under `inline="always"` numba re-specialises the body at the
#    caller's types, so a declared uint64 parameter receiving an int64
#    loop variable is NOT converted -- the mixed expression `node * 4`
#    is then promoted to float64 exactly as numpy would, silently
#    destroying the bit arithmetic. This is not hypothetical: without
#    the casts, `uniform` returns values with mean 0.25 instead of 0.5
#    when called from a `range`/`prange` loop, while remaining correct
#    when called from Python. `test_uniform_moments` pins it.


@njit(uint64(uint64), inline="always", cache=True)
def splitmix64(state):
    """Vigna's splitmix64 finaliser. Unsigned arithmetic wraps by design."""
    z = state + _GOLDEN
    z = (z ^ (z >> _SHIFT_30)) * _MIX_A
    z = (z ^ (z >> _SHIFT_27)) * _MIX_B
    return z ^ (z >> _SHIFT_31)


@njit(uint64(uint64, uint64, uint64), inline="always", cache=True)
def make_key(seed, layer, iteration):
    """Collapse the run-level coordinates into one 64-bit key.

    Held constant across a sweep so per-node keys stay cheap.
    """
    s = np.uint64(seed)
    ly = np.uint64(layer)
    it = np.uint64(iteration)
    return splitmix64(s ^ splitmix64(ly * _GOLDEN + it))


@njit(uint64(uint64, uint64, uint64, uint64), inline="always", cache=True)
def _draw(key, node, allele, purpose):
    k = np.uint64(key)
    n = np.uint64(node)
    a = np.uint64(allele)
    p = np.uint64(purpose)
    coord = (n * np.uint64(4) + a) * np.uint64(16) + p
    return splitmix64(k ^ splitmix64(coord))


@njit(float64(uint64, uint64, uint64, uint64), inline="always", cache=True)
def uniform(key, node, allele, purpose):
    """Uniform on the open interval (0, 1).

    Open rather than half-open because callers take logarithms of it.
    """
    z = _draw(key, node, allele, purpose)
    return (np.float64(z >> _SHIFT_11) + 0.5) * _TWO_POW_M53


@njit(float64(uint64, uint64, uint64, uint64), inline="always", cache=True)
def standard_normal(key, node, allele, purpose):
    """Standard normal by Box-Muller.

    The second uniform comes from a perturbed key rather than a
    neighbouring coordinate, so the draw stays a pure function of its
    coordinate without consuming a purpose tag.
    """
    k = np.uint64(key)
    u1 = uniform(k, node, allele, purpose)
    u2 = uniform(k ^ _GOLDEN, node, allele, purpose)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(_TWO_PI * u2)


@njit(int64(float64, uint64, uint64, uint64, uint64), inline="always", cache=True)
def bernoulli(p, key, node, allele, purpose):
    return 1 if uniform(key, node, allele, purpose) < p else 0


@njit(uint64(uint64, uint64, uint64, uint64, uint64), inline="always", cache=True)
def randint(n, key, node, allele, purpose):
    """Uniform integer in [0, n).

    Plain modulo. The residual bias is ~n/2**64 and n here is a posterior
    sample count (thousands), so it is unmeasurable.
    """
    return _draw(key, node, allele, purpose) % np.uint64(n)


def spawn_seed(seed: int, tag: str) -> np.uint64:
    """Derive a stable sub-seed from a run seed and a text tag.

    Used to give independent chains or stages distinct streams without
    the caller having to invent seeds.
    """
    h = np.uint64(seed)
    for ch in tag.encode():
        h = np.uint64(splitmix64(h ^ np.uint64(ch)))
    return h

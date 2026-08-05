"""The RNG must be reproducible, thread-count-independent, and well-behaved.

The thread-count property is the whole reason this module exists: legacy
Baymer's results depend on ``-t`` despite a fixed seed, because
``@njit(parallel=True)`` regions draw from per-thread state.
"""

from __future__ import annotations

import numba
import numpy as np
import pytest
from numba import njit, prange, set_num_threads, get_num_threads

from baymer2.rng import (
    PURPOSE_INDICATOR,
    PURPOSE_PHI_ACCEPT,
    PURPOSE_PHI_PROPOSAL,
    bernoulli,
    make_key,
    randint,
    spawn_seed,
    standard_normal,
    uniform,
)


# Explicit signatures matter here. `make_key` returns a uint64, which
# Python receives as a plain int that may exceed 2**63; an *untyped*
# njit kernel would then infer int64 for it and raise OverflowError, or
# silently reinterpret the sign bit. Keys must stay unsigned end to end.
_KERNEL_2D = numba.void(numba.uint64, numba.int64, numba.float64[:, :])
_KERNEL_1D = numba.void(numba.uint64, numba.int64, numba.uint64, numba.float64[:])


@njit(_KERNEL_2D, parallel=True)
def _fill_parallel(key, n, out):
    for i in prange(n):
        for j in range(3):
            out[i, j] = standard_normal(key, i, j, PURPOSE_PHI_PROPOSAL)


@njit(_KERNEL_2D)
def _fill_serial(key, n, out):
    for i in range(n):
        for j in range(3):
            out[i, j] = standard_normal(key, i, j, PURPOSE_PHI_PROPOSAL)


@njit(_KERNEL_1D)
def _fill_uniform(key, n, purpose, out):
    for i in range(n):
        out[i] = uniform(key, i, 0, purpose)


def fill_parallel(key, n):
    out = np.zeros((n, 3))
    _fill_parallel(np.uint64(key), n, out)
    return out


def fill_serial(key, n):
    out = np.zeros((n, 3))
    _fill_serial(np.uint64(key), n, out)
    return out


def fill_uniform(key, n, purpose):
    out = np.zeros(n)
    _fill_uniform(np.uint64(key), n, np.uint64(purpose), out)
    return out


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_identical_across_thread_counts():
    """The property legacy Baymer does not have."""
    key = make_key(np.uint64(14), np.uint64(3), np.uint64(1000))
    n = 4096
    original = get_num_threads()
    max_threads = numba.config.NUMBA_NUM_THREADS
    counts = sorted({1, 2, min(4, max_threads), max_threads})
    if len(counts) < 2:
        pytest.skip(f"only {max_threads} numba thread(s) available")
    results = []
    try:
        for threads in counts:
            set_num_threads(threads)
            results.append(fill_parallel(key, n))
    finally:
        set_num_threads(original)
    for r in results[1:]:
        np.testing.assert_array_equal(results[0], r)


def test_parallel_matches_serial():
    """Loop order must not matter."""
    key = make_key(np.uint64(7), np.uint64(0), np.uint64(1))
    n = 2048
    par = fill_parallel(key, n)
    ser = fill_serial(key, n)
    np.testing.assert_array_equal(par, ser)


def test_same_coordinate_gives_same_draw():
    key = make_key(np.uint64(1), np.uint64(2), np.uint64(3))
    a = standard_normal(key, 17, 2, PURPOSE_PHI_PROPOSAL)
    b = standard_normal(key, 17, 2, PURPOSE_PHI_PROPOSAL)
    assert a == b


def test_different_seeds_give_different_draws():
    k1 = make_key(np.uint64(1), np.uint64(0), np.uint64(0))
    k2 = make_key(np.uint64(2), np.uint64(0), np.uint64(0))
    assert standard_normal(k1, 0, 0, 0) != standard_normal(k2, 0, 0, 0)


def test_iteration_advances_the_stream():
    k1 = make_key(np.uint64(5), np.uint64(1), np.uint64(100))
    k2 = make_key(np.uint64(5), np.uint64(1), np.uint64(101))
    assert standard_normal(k1, 3, 1, 0) != standard_normal(k2, 3, 1, 0)


def test_purposes_are_independent():
    """Different uses of one coordinate must not be correlated."""
    key = make_key(np.uint64(11), np.uint64(0), np.uint64(0))
    n = 20000
    a = fill_uniform(key, n, PURPOSE_PHI_PROPOSAL)
    b = fill_uniform(key, n, PURPOSE_PHI_ACCEPT).copy()
    c = fill_uniform(key, n, PURPOSE_INDICATOR)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.03
    assert abs(np.corrcoef(a, c)[0, 1]) < 0.03
    assert abs(np.corrcoef(b, c)[0, 1]) < 0.03


# --------------------------------------------------------------------------
# Distributional correctness
# --------------------------------------------------------------------------


def test_uniform_is_in_the_open_unit_interval():
    key = make_key(np.uint64(3), np.uint64(0), np.uint64(0))
    vals = fill_uniform(key, 100000, PURPOSE_PHI_ACCEPT)
    assert np.all(vals > 0.0)
    assert np.all(vals < 1.0)


def test_uniform_moments():
    key = make_key(np.uint64(4), np.uint64(0), np.uint64(0))
    vals = fill_uniform(key, 200000, PURPOSE_PHI_ACCEPT)
    assert vals.mean() == pytest.approx(0.5, abs=0.005)
    assert vals.var() == pytest.approx(1 / 12, abs=0.005)


def test_uniform_is_flat_across_bins():
    key = make_key(np.uint64(9), np.uint64(0), np.uint64(0))
    n = 200000
    vals = fill_uniform(key, n, PURPOSE_PHI_ACCEPT)
    counts, _ = np.histogram(vals, bins=20, range=(0, 1))
    expected = n / 20
    # chi-square with 19 df; 99.9th percentile is ~43.8
    chi2 = ((counts - expected) ** 2 / expected).sum()
    assert chi2 < 43.8


def test_normal_moments():
    key = make_key(np.uint64(6), np.uint64(0), np.uint64(0))
    n = 100000
    vals = fill_serial(key, n).ravel()
    assert vals.mean() == pytest.approx(0.0, abs=0.02)
    assert vals.std() == pytest.approx(1.0, abs=0.02)
    # Box-Muller should give clean tails
    assert abs(((vals**3).mean())) < 0.05
    assert (vals**4).mean() == pytest.approx(3.0, abs=0.1)


def test_normal_tail_fractions():
    key = make_key(np.uint64(8), np.uint64(0), np.uint64(0))
    n = 200000
    vals = fill_serial(key, n).ravel()
    assert np.mean(np.abs(vals) < 1.0) == pytest.approx(0.6827, abs=0.01)
    assert np.mean(np.abs(vals) < 2.0) == pytest.approx(0.9545, abs=0.01)


def test_bernoulli_frequency():
    key = make_key(np.uint64(12), np.uint64(0), np.uint64(0))
    draws = [bernoulli(0.3, key, i, 0, PURPOSE_INDICATOR) for i in range(50000)]
    assert np.mean(draws) == pytest.approx(0.3, abs=0.01)
    assert set(draws) == {0, 1}


def test_bernoulli_extremes_are_deterministic():
    key = make_key(np.uint64(13), np.uint64(0), np.uint64(0))
    assert all(bernoulli(1.0, key, i, 0, 0) == 1 for i in range(500))
    assert all(bernoulli(0.0, key, i, 0, 0) == 0 for i in range(500))


def test_randint_range_and_coverage():
    key = make_key(np.uint64(15), np.uint64(0), np.uint64(0))
    n = 7
    draws = np.array([randint(n, key, i, 0, 0) for i in range(20000)])
    assert draws.min() >= 0
    assert draws.max() < n
    counts = np.bincount(draws, minlength=n)
    assert np.all(counts > 20000 / n * 0.9)


# --------------------------------------------------------------------------
# Seed derivation
# --------------------------------------------------------------------------


def test_spawn_seed_is_stable_and_tag_dependent():
    assert spawn_seed(14, "chain0") == spawn_seed(14, "chain0")
    assert spawn_seed(14, "chain0") != spawn_seed(14, "chain1")
    assert spawn_seed(14, "chain0") != spawn_seed(15, "chain0")

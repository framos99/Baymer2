"""Measure the O(1) sufficient-statistic likelihood against legacy Baymer's.

Legacy ``run_baymer.get_edge_likelihood_numba`` recomputes four strided
sums over a node's full leaf-count vector on *every* Metropolis
proposal. Those sums are constants of the data. This benchmark times one
full layer sweep -- ``3 * layer_size`` likelihood evaluations, which is
what a single MCMC iteration costs -- under both schemes.

Run with::

    PYTHONPATH=src python benchmarks/bench_likelihood.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import njit, prange

from baymer2.contexts import ContextTree
from baymer2.likelihoods.binomial import node_loglik


@njit(cache=True)
def _legacy_edge_likelihood(leaf_counts, p_vec):
    """Transcription of legacy ``get_edge_likelihood_numba``."""
    return (
        np.sum(leaf_counts[0::4]) * np.log(1.0 - np.sum(p_vec))
        + np.sum(leaf_counts[1::4]) * np.log(p_vec[0])
        + np.sum(leaf_counts[2::4]) * np.log(p_vec[1])
        + np.sum(leaf_counts[3::4]) * np.log(p_vec[2])
    )


# Each of the three proposals per node perturbs one rate component, exactly
# as the univariate random-walk sampler does. Passing identical arguments
# three times would let LLVM collapse them by common-subexpression
# elimination -- which would flatter the inlined fast path in particular --
# so the perturbation is load-bearing for the measurement, not decoration.
_NUDGE = 1.0000001


@njit(cache=True, parallel=True)
def legacy_sweep(leaf_counts, rates, out):
    """One layer sweep the legacy way: 3 proposals per node, full re-sum."""
    n = leaf_counts.shape[0]
    for i in prange(n):
        acc = 0.0
        prop = np.empty(3)
        for j in range(3):
            prop[0] = rates[i, 0]
            prop[1] = rates[i, 1]
            prop[2] = rates[i, 2]
            prop[j] = prop[j] * _NUDGE
            acc += _legacy_edge_likelihood(leaf_counts[i], prop)
        out[i] = acc
    return out


@njit(cache=True, parallel=True)
def fast_sweep(stats, rates, out):
    """The same sweep against precomputed sufficient statistics."""
    n = stats.shape[0]
    for i in prange(n):
        acc = 0.0
        for j in range(3):
            p0 = rates[i, 0]
            p1 = rates[i, 1]
            p2 = rates[i, 2]
            if j == 0:
                p0 = p0 * _NUDGE
            elif j == 1:
                p1 = p1 * _NUDGE
            else:
                p2 = p2 * _NUDGE
            acc += node_loglik(
                stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], p0, p1, p2
            )
        out[i] = acc
    return out


def build_layer(max_mer: int, layer: int, rng):
    """Synthetic leaf counts plus their reduction, for one layer."""
    tree = ContextTree(max_mer=max_mer)
    layer_size = tree.layer_size(layer)
    n_leaves_per_node = 4 ** (max_mer - layer - 1)

    per_leaf = rng.integers(0, 5000, size=(layer_size, n_leaves_per_node, 4))
    per_leaf = per_leaf.astype(np.float64)

    leaf_counts = np.ascontiguousarray(per_leaf.reshape(layer_size, -1))
    stats = np.ascontiguousarray(per_leaf.sum(axis=1))
    rates = rng.uniform(1e-8, 1e-4, size=(layer_size, 3))
    return leaf_counts, stats, rates


def time_call(fn, *args, repeats: int) -> float:
    fn(*args)  # compile / warm cache
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-mers", type=int, nargs="+", default=[5, 7, 9])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(0)

    print(f"{'max_mer':>8} {'layer':>6} {'nodes':>8} {'leaves/node':>12} "
          f"{'legacy (ms)':>12} {'fast (ms)':>10} {'speedup':>9}")
    print("-" * 72)

    for max_mer in args.max_mers:
        totals = [0.0, 0.0]
        for layer in range(max_mer):
            leaf_counts, stats, rates = build_layer(max_mer, layer, rng)
            out = np.zeros(stats.shape[0])

            t_legacy = time_call(legacy_sweep, leaf_counts, rates, out, repeats=args.repeats)
            legacy_vals = out.copy()
            t_fast = time_call(fast_sweep, stats, rates, out, repeats=args.repeats)

            np.testing.assert_allclose(out, legacy_vals, rtol=1e-9)

            totals[0] += t_legacy
            totals[1] += t_fast
            print(
                f"{max_mer:>8} {layer:>6} {stats.shape[0]:>8} "
                f"{4 ** (max_mer - layer - 1):>12} "
                f"{t_legacy * 1e3:>12.3f} {t_fast * 1e3:>10.3f} "
                f"{t_legacy / t_fast:>8.1f}x"
            )
        print(
            f"{max_mer:>8} {'ALL':>6} {'':>8} {'':>12} "
            f"{totals[0] * 1e3:>12.3f} {totals[1] * 1e3:>10.3f} "
            f"{totals[0] / totals[1]:>8.1f}x   <- one full-model sweep"
        )
        print("-" * 72)


if __name__ == "__main__":
    main()

"""Measure the cost of legacy's zero-context membership test.

Legacy Baymer stores the indices of unobserved contexts in a plain numpy
array and tests membership with ``if i in zero_contexts``. Numba lowers
that to a linear scan. In ``update_alpha_probabilities`` the test sits
*inside* the per-allele loop, so one call costs
``O(layer_size * 3 * n_zero)`` -- and it runs whenever the alpha step
accepts, which the tutorial report puts at ~79% of iterations.

A boolean mask makes the same test O(1).

Run with::

    PYTHONPATH=src python benchmarks/bench_zero_mask.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import njit


@njit(cache=True)
def legacy_alpha_probabilities(layer_size, indicator, alpha_probs, zero_contexts, out):
    """Legacy shape: linear membership scan inside the allele loop."""
    for i in range(layer_size):
        for sub_index in range(3):
            skip = False
            for z in range(zero_contexts.shape[0]):
                if zero_contexts[z] == i:
                    skip = True
                    break
            if skip:
                continue
            out[i, sub_index] = alpha_probs[indicator[i, sub_index]]
    return out


@njit(cache=True)
def masked_alpha_probabilities(layer_size, indicator, alpha_probs, zero_mask, out):
    """Same computation, O(1) membership test."""
    for i in range(layer_size):
        if zero_mask[i]:
            continue
        for sub_index in range(3):
            out[i, sub_index] = alpha_probs[indicator[i, sub_index]]
    return out


def time_call(fn, *args, repeats: int) -> float:
    fn(*args)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-mer", type=int, default=9)
    ap.add_argument("--zero-fracs", type=float, nargs="+",
                    default=[0.001, 0.01, 0.05, 0.10, 0.25])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    layer = args.max_mer - 1
    layer_size = 2 * 4**layer
    indicator = rng.integers(0, 2, size=(layer_size, 3)).astype(np.int64)
    alpha_probs = np.array([np.log(0.5), np.log(0.5)])

    print(f"deepest layer of a {args.max_mer}-mer model: {layer_size} nodes")
    print(f"{'zero frac':>10} {'n_zero':>8} {'legacy (ms)':>13} "
          f"{'mask (ms)':>11} {'speedup':>9}")
    print("-" * 56)

    for frac in args.zero_fracs:
        n_zero = max(1, int(layer_size * frac))
        zero_idx = np.sort(rng.choice(layer_size, size=n_zero, replace=False))
        # legacy prepends a -1 sentinel
        zero_contexts = np.concatenate([np.array([-1]), zero_idx]).astype(np.int64)
        zero_mask = np.zeros(layer_size, dtype=np.bool_)
        zero_mask[zero_idx] = True

        out_a = np.zeros((layer_size, 3))
        out_b = np.zeros((layer_size, 3))

        t_legacy = time_call(
            legacy_alpha_probabilities, layer_size, indicator, alpha_probs,
            zero_contexts, out_a, repeats=args.repeats,
        )
        t_mask = time_call(
            masked_alpha_probabilities, layer_size, indicator, alpha_probs,
            zero_mask, out_b, repeats=args.repeats,
        )
        np.testing.assert_array_equal(out_a, out_b)

        print(f"{frac:>10.3f} {n_zero:>8} {t_legacy * 1e3:>13.2f} "
              f"{t_mask * 1e3:>11.3f} {t_legacy / t_mask:>8.0f}x")


if __name__ == "__main__":
    main()

"""Peak resident memory: in-RAM traces versus memory-mapped traces.

Legacy Baymer's trace arrays scale as ``4**layer``; at ``max_mer=9`` the
deepest layer alone needs ~12.6 GB across the four per-node quantities,
which is what drives ``rusage[mem=25GB]`` in the production wrapper.

This measures peak RSS of a whole fit under both storage modes. Run each
mode in its own process so the high-water mark is not contaminated::

    PYTHONPATH=src python benchmarks/bench_memory.py --mode memory --max-mer 8
    PYTHONPATH=src python benchmarks/bench_memory.py --mode mmap   --max-mer 8

or let it drive both::

    PYTHONPATH=src python benchmarks/bench_memory.py --max-mer 8
"""

from __future__ import annotations

import argparse
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from baymer2.backends.numba_mh import LayerConfig
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.model import ModelConfig, fit


def synthetic_table(max_mer: int, seed: int = 0) -> CountTable:
    """Counts with a realistic fraction of unobserved deep contexts."""
    tree = ContextTree(max_mer=max_mer, alternation="right")
    rng = np.random.default_rng(seed)
    n = tree.n_leaves
    depth = rng.poisson(60.0, size=n).astype(np.float64)
    counts = np.zeros((n, 4))
    counts[:, 0] = depth
    for j in range(1, 4):
        counts[:, j] = rng.binomial(depth.astype(int), 1e-3)
    counts[:, 0] = np.maximum(counts[:, 0] - counts[:, 1:].sum(axis=1), 0.0)
    return CountTable(tree=tree, leaf_counts=counts, name="bench")


def predicted_trace_bytes(max_mer: int, n_samples: int) -> int:
    """Bytes the per-node trace arrays need if held in RAM."""
    total = 0
    for layer in range(max_mer):
        n_nodes = ContextTree(max_mer=max_mer).layer_size(layer)
        # eta, phi, rates as float32 plus indicator as int8
        total += n_samples * n_nodes * 3 * (4 + 4 + 4 + 1)
    return total


def run_once(mode: str, max_mer: int, iterations: int, thin: int) -> None:
    table = synthetic_table(max_mer)
    cfg = ModelConfig(
        max_mer=max_mer,
        seed=1,
        default_layer=LayerConfig(
            n_iterations=iterations,
            burnin=iterations // 2,
            thin=thin,
            set_sigma=0.8540486,
            c=0.04472136,
            phi_proposal_sd=0.05,
        ),
    )
    if mode == "mmap":
        with tempfile.TemporaryDirectory(prefix="baymer2_traces_") as tmp:
            result = fit(table, cfg, trace_dir=tmp)
            on_disk = result.trace_store.nbytes() / 1e9
    else:
        result = fit(table, cfg)
        on_disk = 0.0

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    n_samples = cfg.default_layer.n_samples
    print(
        f"mode={mode:6s} max_mer={max_mer} samples={n_samples} "
        f"peak_rss={peak_gb:6.2f} GB  on_disk={on_disk:5.2f} GB  "
        f"elapsed={result.elapsed_seconds:6.1f}s"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["memory", "mmap"], default=None)
    ap.add_argument("--max-mer", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--thin", type=int, default=1)
    args = ap.parse_args()

    if args.mode is not None:
        run_once(args.mode, args.max_mer, args.iterations, args.thin)
        return

    n_samples = (args.iterations - args.iterations // 2) // args.thin
    predicted = predicted_trace_bytes(args.max_mer, n_samples) / 1e9
    print(f"predicted in-RAM trace footprint at max_mer={args.max_mer}, "
          f"{n_samples} samples: {predicted:.2f} GB")
    print()
    for mode in ("memory", "mmap"):
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--mode", mode, "--max-mer", str(args.max_mer),
             "--iterations", str(args.iterations), "--thin", str(args.thin)],
            check=True,
        )


if __name__ == "__main__":
    main()

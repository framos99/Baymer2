"""Compare a baymer2 fit against legacy Baymer's shipped posterior.

Legacy and baymer2 do not target a bit-identical posterior -- the ALR
reparameterisation and the corrected alpha step both change the chain --
so a raw difference is not by itself evidence of a bug. The question
that can be answered is whether the difference between implementations
is larger than the difference between two runs of the *same*
implementation under different seeds. If it is not, the two agree to
within Monte Carlo error.

Needs a checkout of the original Baymer
(https://github.com/bvoightlab/Baymer) whose tutorial has already been
run, since the comparison is against its shipped ``rate_matrix`` arrays.
Point at it with ``$BAYMER2_LEGACY_ROOT`` or ``--legacy-out``::

    BAYMER2_LEGACY_ROOT=/path/to/Baymer \\
        PYTHONPATH=src python validation/compare_to_legacy.py

Reference result on the tutorial data (AFR, cpg_islands, ALL, 5-mer,
40000 iterations / 20000 burn-in / thin 10):

    layer  b2 seed-to-seed   b2 vs legacy   ratio
        0           0.0005         0.0005    1.03
        1           0.0016         0.0020    1.22
        2           0.0032         0.0061    1.91
        3           0.0069         0.0099    1.44
        4           0.0122         0.0122    1.00

Every layer sits at or near 1.0, i.e. the implementations differ no more
than one implementation differs from itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from baymer2.backends.numba_mh import LayerConfig
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.model import ModelConfig, fit

LEGACY_ROOT_ENV = "BAYMER2_LEGACY_ROOT"
#: Where the tutorial's fitted rate matrices land inside a Baymer checkout.
LEGACY_OUT_SUBPATH = "tutorial_data/tutorial_outs/tutorial_baymer_out/AFR"
GOLDEN = Path(__file__).resolve().parent.parent / "tests/golden/count_pipeline"

# The tutorial's own parameter config.
LEGACY_C = 0.04472136
LEGACY_SET_SIGMA = 0.8540486


def load_table(max_mer: int) -> CountTable:
    tree = ContextTree(max_mer=max_mer, alternation="right")
    dicts = {
        k: json.loads(
            (GOLDEN / f"{k}mer.ALL.AFR.cpg_islands.folded_count_dict.json").read_text()
        )
        for k in range(1, max_mer + 1)
    }
    return CountTable.from_count_dict_set(dicts, tree, name="AFR")


def run(table: CountTable, seed: int, iterations: int, burnin: int) -> list[np.ndarray]:
    cfg = ModelConfig(
        max_mer=table.tree.max_mer,
        seed=seed,
        default_layer=LayerConfig(
            n_iterations=iterations,
            burnin=burnin,
            thin=10,
            set_sigma=LEGACY_SET_SIGMA,
            c=LEGACY_C,
            phi_proposal_sd=0.05,
            alpha_init=0.8,
        ),
    )
    result = fit(table, cfg)
    return [result.rates(l).astype(np.float64) for l in range(table.tree.max_mer)]


def resolve_legacy_out(explicit: Path | None) -> Path:
    """Directory holding legacy's fitted rate matrices.

    ``--legacy-out`` wins; otherwise ``$BAYMER2_LEGACY_ROOT`` is treated as
    a Baymer checkout whose tutorial has been run.
    """
    if explicit is not None:
        path = explicit
    else:
        root = os.environ.get(LEGACY_ROOT_ENV)
        if not root:
            sys.exit(
                f"no legacy Baymer output configured: set ${LEGACY_ROOT_ENV} to a "
                "checkout of https://github.com/bvoightlab/Baymer whose tutorial "
                "has been run, or pass --legacy-out."
            )
        path = Path(root) / LEGACY_OUT_SUBPATH
    if not path.is_dir():
        sys.exit(f"legacy output directory does not exist: {path}")
    return path


def legacy_means(max_mer: int, legacy_out: Path) -> list[np.ndarray]:
    out = []
    for layer in range(max_mer):
        path = (
            legacy_out
            / f"AFR_cpg_islands_ALL_rs14_rate_matrix.burned_in.thinned.layer_{layer}.npy"
        )
        if not path.exists():
            sys.exit(f"missing legacy rate matrix: {path}")
        out.append(np.load(path).mean(axis=0).astype(np.float64))
    return out


def median_relative(a: np.ndarray, b: np.ndarray) -> float:
    """Median symmetric relative difference, robust to near-zero rates."""
    scale = np.maximum((np.abs(a) + np.abs(b)) / 2.0, 1e-12)
    return float(np.median(np.abs(a - b) / scale))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-mer", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=40000)
    ap.add_argument("--burnin", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[14, 15, 16])
    ap.add_argument(
        "--legacy-out", type=Path, default=None,
        help=f"directory of legacy rate matrices; defaults to "
             f"${LEGACY_ROOT_ENV}/{LEGACY_OUT_SUBPATH}",
    )
    args = ap.parse_args()

    legacy_out = resolve_legacy_out(args.legacy_out)

    table = load_table(args.max_mer)
    print(table)

    chains = [run(table, s, args.iterations, args.burnin) for s in args.seeds]
    legacy = legacy_means(args.max_mer, legacy_out)

    print()
    print(f"{'layer':>5} {'nodes':>7} {'b2 seed-to-seed':>17} "
          f"{'b2 vs legacy':>14} {'ratio':>7}  verdict")
    print("-" * 76)

    for layer in range(args.max_mer):
        per_seed = [c[layer] for c in chains]
        within = np.mean([
            median_relative(per_seed[i], per_seed[j])
            for i in range(len(per_seed))
            for j in range(i + 1, len(per_seed))
        ])
        consensus = np.mean(per_seed, axis=0)
        across = median_relative(consensus, legacy[layer])
        ratio = across / within if within > 0 else float("inf")
        verdict = (
            "within MC noise" if ratio < 2.0
            else "mild systematic" if ratio < 4.0
            else "SYSTEMATIC - investigate"
        )
        print(f"{layer:>5} {per_seed[0].shape[0]:>7} {within:>17.4f} "
              f"{across:>14.4f} {ratio:>7.2f}  {verdict}")


if __name__ == "__main__":
    main()

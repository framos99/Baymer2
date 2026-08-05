"""Does the posterior have the width it claims?

Every other test here checks where the posterior *sits* -- that point
estimates land near the truth. None checks how *wide* it is. A sampler
can recover means perfectly while reporting intervals that are twice too
narrow, and no recovery test would notice. The consequence is
overconfident credible intervals on every context rate the model
reports.

Method
------
Simulate from the model's own generative process rather than from
arbitrary rates, otherwise the hierarchical prior is misspecified and
poor coverage would say nothing about the sampler:

1. draw layer-0 ALR coordinates in a plausible range;
2. descend the tree drawing ``z ~ Bernoulli(alpha)`` and
   ``phi ~ N(0, sigma_z**2)``, so ``eta_child = eta_parent + phi``;
3. multinomial-sample counts at the leaves;
4. fit with the same ``alpha``, ``sigma``, ``c``;
5. per layer, ask what fraction of true rates fall inside their 95%
   equal-tailed credible interval.

A correct sampler on a correctly specified model gives ~95%. Deep layers
are fitted under the *cut* posterior, which conditions on a plug-in
parent rather than propagating its uncertainty, so some loss there is
expected and is a property of the model, not a bug -- quantifying it is
the point.

Run with::

    PYTHONPATH=src python validation/check_calibration.py
"""

from __future__ import annotations

import argparse

import numpy as np

from baymer2.backends.numba_mh import LayerConfig, rates_from_eta
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.model import ModelConfig, fit


def simulate_from_prior(
    tree: ContextTree, rng, *, alpha: float, sigma: float, c: float,
    root_eta_range: tuple[float, float] = (-9.0, -6.0),
) -> list[np.ndarray]:
    """Draw true ALR coordinates for every layer, top down."""
    etas: list[np.ndarray] = []
    root = rng.uniform(*root_eta_range, size=(tree.layer_size(0), 3))
    etas.append(root)

    for layer in range(1, tree.n_layers):
        parent = etas[-1]
        n_children = tree.layer_size(layer)
        repeated = np.repeat(parent, tree.branching, axis=0)
        indicator = rng.random((n_children, 3)) < alpha
        scale = np.where(indicator, sigma, c * sigma)
        phi = rng.normal(0.0, scale)
        etas.append(repeated + phi)
    return etas


def simulate_counts(tree: ContextTree, eta_leaf: np.ndarray, depth: int, rng):
    rates = rates_from_eta(eta_leaf)
    counts = np.zeros((rates.shape[0], 4))
    for i, p in enumerate(rates):
        counts[i] = rng.multinomial(depth, [1.0 - p.sum(), *p])
    return CountTable(tree=tree, leaf_counts=counts, name="sim"), rates


def coverage_for(
    samples: np.ndarray, truth: np.ndarray, level: float
) -> tuple[float, float]:
    """Fraction of true values inside the interval, and mean interval width."""
    tail = (1.0 - level) / 2.0
    lower = np.quantile(samples, tail, axis=0)
    upper = np.quantile(samples, 1.0 - tail, axis=0)
    inside = (truth >= lower) & (truth <= upper)
    with np.errstate(divide="ignore", invalid="ignore"):
        width = np.mean((upper - lower) / np.maximum(truth, 1e-300))
    return float(inside.mean()), float(width)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-mer", type=int, default=5)
    ap.add_argument("--replicates", type=int, default=6)
    ap.add_argument("--depth", type=int, default=200_000,
                    help="sampled chromosomes per context")
    ap.add_argument("--iterations", type=int, default=12000)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--c", type=float, default=0.04472136)
    args = ap.parse_args()

    tree = ContextTree(max_mer=args.max_mer, alternation="right")
    hits = {l: [] for l in range(tree.n_layers)}
    widths = {l: [] for l in range(tree.n_layers)}
    hits_pooled = {l: [] for l in range(tree.n_layers)}
    bias = {l: [] for l in range(tree.n_layers)}

    for replicate in range(args.replicates):
        rng = np.random.default_rng(1000 + replicate)
        etas = simulate_from_prior(
            tree, rng, alpha=args.alpha, sigma=args.sigma, c=args.c
        )
        table, leaf_rates = simulate_counts(tree, etas[-1], args.depth, rng)

        # Two competing definitions of "the true rate" at an internal node.
        #
        # cascade: the value the generative process placed there.
        # pooled : the aggregate rate over that node's descendants, which is
        #          what the node's pooled multinomial likelihood identifies.
        #
        # They are not the same. phi is Gaussian on the log scale, so
        # E[exp(phi)] = exp(sigma**2 / 2) > 1 and each layer of descent
        # inflates the aggregate above the cascade value. Coverage must be
        # judged against the estimand the model actually fits.
        pooled_truth = []
        for layer in range(tree.n_layers):
            per_leaf = leaf_rates.reshape(
                tree.layer_size(layer), -1, 3
            )
            pooled_truth.append(per_leaf.mean(axis=1))

        config = ModelConfig(
            max_mer=args.max_mer,
            seed=replicate,
            default_layer=LayerConfig(
                n_iterations=args.iterations,
                burnin=args.iterations // 2,
                thin=5,
                set_sigma=args.sigma,
                c=args.c,
                alpha_init=args.alpha,
                phi_proposal_sd=0.05,
            ),
        )
        result = fit(table, config)

        for layer in range(tree.n_layers):
            cascade = rates_from_eta(etas[layer])
            pooled = pooled_truth[layer]
            samples = np.asarray(result.layers[layer].rates, dtype=np.float64)

            covered, width = coverage_for(samples, cascade, args.level)
            hits[layer].append(covered)
            widths[layer].append(width)

            covered_pooled, _ = coverage_for(samples, pooled, args.level)
            hits_pooled[layer].append(covered_pooled)
            bias[layer].append(float(np.mean(pooled / cascade)))

    print(f"simulated from the model's own prior; {args.replicates} replicates, "
          f"depth {args.depth:,}, {args.iterations} iterations")
    print(f"nominal coverage {args.level:.0%}\n")
    print(f"{'layer':>6} {'nodes':>7} {'cov(cascade)':>13} {'cov(pooled)':>12} "
          f"{'pooled/cascade':>15} {'rel. CI width':>14}  verdict")
    print("-" * 88)
    for layer in range(tree.n_layers):
        cascade_cov = float(np.mean(hits[layer]))
        pooled_cov = float(np.mean(hits_pooled[layer]))
        n_params = tree.layer_size(layer) * 3 * args.replicates
        se = np.sqrt(args.level * (1 - args.level) / n_params)
        tol = max(3 * se, 0.03)
        verdict = (
            "calibrated" if abs(pooled_cov - args.level) < tol
            else ("conservative" if pooled_cov > args.level else "OVERCONFIDENT")
        )
        print(f"{layer:>6} {tree.layer_size(layer):>7} {cascade_cov:>13.3f} "
              f"{pooled_cov:>12.3f} {np.mean(bias[layer]):>15.4f} "
              f"{np.mean(widths[layer]):>14.4f}  {verdict}")
    print()
    print("cov(cascade): true value the generative process placed at the node")
    print("cov(pooled) : aggregate rate over its descendants -- the estimand")
    print("              the node's pooled multinomial likelihood identifies")
    print(f"expected inflation per layer of descent: exp(sigma^2/2) = "
          f"{np.exp(args.sigma ** 2 / 2):.4f}")


if __name__ == "__main__":
    main()

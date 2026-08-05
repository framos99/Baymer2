"""Fit the whole context tree, layer by layer.

Linkage
-------
``cut`` (default) reproduces legacy Baymer's structure: layer ``l`` is
run to completion, its thinned draws are stored, and layer ``l+1`` then
treats those draws as an empirical prior, resampling a parent state
independently per parent node on every iteration. This is a modular or
"cut" posterior -- information flows from shallow layers to deep ones
but never back. It keeps memory bounded to two layers at a time and
keeps results comparable with existing Baymer runs.

``joint`` samples every layer in one chain with the data entering
*once*, at the leaves. Under ``cut`` each layer carries its own
multinomial on pooled counts, so the same observations are reused at
every depth; that is what makes the layers separable, but it is not a
posterior over the tree. Under ``joint`` the target is::

    sum_leaves log Multinomial(n_u | p_u)
      + sum_{v,j} log p(phi_vj | z_vj, sigma_l)
      + sum log p(z | alpha_l)

so a deep context can inform its ancestors. Results are not comparable
with existing Baymer output.

Cost is the trade: a proposal at layer ``l`` rescores that node's whole
leaf subtree, and every layer stays resident. Practical to about
``max_mer=7``; beyond that use ``cut``, whose memory stays at two
layers. At ``max_mer=5`` joint is in fact the faster of the two, because
it never recomputes pooled likelihoods.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from baymer2.backends.numba_mh import (
    LayerConfig,
    LayerResult,
    NumbaJointBackend,
    NumbaMetropolisBackend,
)
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.io.traces import TraceStore
from baymer2.likelihoods.base import MODES

__all__ = ["ModelConfig", "ModelResult", "fit"]


@lru_cache(maxsize=None)
def _ctmc_maps(tree: ContextTree, layer: int):
    """Quartet index maps for one layer, built once and reused."""
    from baymer2.likelihoods.ctmc import build_qmap, build_quartets

    q_src, q_slot, slot_base = build_qmap(tree, layer)
    members, sizes, _ = build_quartets(tree, layer)
    return q_src, q_slot, slot_base, members, sizes

LINKAGES = ("cut", "joint")


@dataclass
class ModelConfig:
    """Settings for a whole-tree fit.

    ``layers`` maps layer index to its :class:`LayerConfig`. Any layer
    without an entry falls back to ``default_layer`` with its seed
    derived from ``seed`` so chains stay reproducible and distinct.
    """

    max_mer: int
    alternation: str = "right"
    linkage: str = "cut"
    likelihood: str = "binomial"
    seed: int = 14
    growth: str = "alternating"
    #: Only used by the ctmc likelihood; see baymer2.likelihoods.ctmc.
    branch_length: float = 1.0
    default_layer: LayerConfig = field(default_factory=LayerConfig)
    layers: dict[int, LayerConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.linkage not in LINKAGES:
            raise ValueError(
                f"linkage must be one of {LINKAGES}, got {self.linkage!r}"
            )
        if self.likelihood not in MODES:
            raise ValueError(
                f"likelihood must be one of {sorted(MODES)}, "
                f"got {self.likelihood!r}"
            )

    def layer_config(self, layer: int) -> LayerConfig:
        cfg = self.layers.get(layer)
        if cfg is None:
            cfg = LayerConfig(**{**self.default_layer.__dict__})
            # Distinct but reproducible stream per layer.
            cfg.seed = self.seed
        return cfg


@dataclass
class ModelResult:
    """Per-layer posterior draws plus provenance."""

    config: ModelConfig
    tree: ContextTree
    layers: list[LayerResult]
    elapsed_seconds: float
    count_name: str = ""
    trace_store: TraceStore | None = None

    def rates(self, layer: int) -> np.ndarray:
        """Posterior mean alternate-allele rates for ``layer``."""
        return self.layers[layer].mean_rates

    def rate_dict(self, layer: int) -> dict[str, list[float]]:
        """Context -> ``[p_ref, p_alt1, p_alt2, p_alt3]``.

        Matches the shape of legacy's ``rate_dict`` outputs, with the
        reference probability first rather than slotted by base.
        """
        contexts = self.tree.contexts(layer)
        means = self.rates(layer)
        return {
            ctx: [float(1.0 - means[i].sum()), *map(float, means[i])]
            for i, ctx in enumerate(contexts)
        }

    def summary(self) -> str:
        lines = [
            f"baymer2 fit: max_mer={self.tree.max_mer} "
            f"growth={self.tree.growth} "
            + (
                f"alternation={self.tree.alternation} "
                if not self.tree.centred
                else ""
            )
            + f"linkage={self.config.linkage} "
            f"likelihood={self.config.likelihood}",
            f"  elapsed: {self.elapsed_seconds:.1f}s",
        ]
        for r in self.layers:
            n_zero = int(r.zero_mask.sum()) if r.zero_mask is not None else 0
            lines.append(
                f"  layer {r.layer}: {r.rates.shape[1]:>7} nodes "
                f"({n_zero} empty)  {r.rates.shape[0]:>5} samples  "
                f"accept(phi)={r.accept_rate_phi:.3f}"
            )
        return "\n".join(lines)


def fit(
    table: CountTable,
    config: ModelConfig,
    *,
    backend: NumbaMetropolisBackend | None = None,
    progress: bool = False,
    trace_dir: str | Path | None = None,
) -> ModelResult:
    """Fit every layer of the context tree to ``table``.

    Parameters
    ----------
    trace_dir:
        When given, thinned draws are written to memory-mapped ``.npy``
        files there instead of being held in RAM. Beyond about
        ``max_mer=7`` this is effectively required: the in-memory traces
        for one layer reach several GB and scale as ``4**layer``.
    """
    if table.tree.max_mer != config.max_mer:
        raise ValueError(
            f"count table has max_mer={table.tree.max_mer} but config says "
            f"{config.max_mer}"
        )
    if table.tree.alternation != config.alternation:
        raise ValueError(
            f"count table uses alternation {table.tree.alternation!r} but "
            f"config says {config.alternation!r}"
        )
    if table.tree.growth != config.growth:
        raise ValueError(
            f"count table uses growth {table.tree.growth!r} but config says "
            f"{config.growth!r}"
        )

    if table.total_sites <= 0:
        raise ValueError(
            "every context in this count table is empty, so there is nothing "
            "to fit. Sampling would run to completion and return the "
            "initialisation values as though they were a posterior."
        )

    backend = backend or NumbaMetropolisBackend()
    mode = MODES[config.likelihood]

    if config.linkage == "joint":
        if config.likelihood != "binomial":
            raise ValueError(
                f"joint linkage currently supports only the binomial "
                f"likelihood, not {config.likelihood!r}; the ctmc mode is "
                f"quartet-local and has no joint kernel yet"
            )
        configs = [config.layer_config(l) for l in range(table.tree.n_layers)]
        joint = NumbaJointBackend().run(table, configs, progress=progress)
        return ModelResult(
            config=config,
            tree=table.tree,
            layers=joint.layers,
            elapsed_seconds=joint.elapsed_seconds,
            count_name=table.name,
        )
    if config.likelihood == "ctmc" and not table.tree.reverse_complement_closed:
        raise ValueError(
            "the ctmc likelihood needs every layer to be "
            "reverse-complement closed, which the alternating tree is "
            "not. Build the count table with growth='centred'."
        )
    store = TraceStore(trace_dir, prefix=table.name) if trace_dir else None

    results: list[LayerResult] = []
    parent_samples: np.ndarray | None = None
    started = time.time()

    for layer in range(table.tree.n_layers):
        cfg = config.layer_config(layer)
        if progress:
            print(
                f"layer {layer}/{table.tree.n_layers - 1}: "
                f"{table.tree.layer_size(layer)} nodes, "
                f"{cfg.n_iterations} iterations"
            )
        result = backend.run_layer(
            layer,
            table.layer_stats(layer),
            table.zero_mask(layer),
            cfg,
            parent_samples=parent_samples,
            mode=mode,
            progress=progress,
            trace_store=store,
            branching=table.tree.branching,
            ctmc_maps=_ctmc_maps(table.tree, layer) if mode == MODES["ctmc"] else None,
            branch_length=config.branch_length,
        )
        results.append(result)
        # Only the ALR coordinates cross the layer boundary; everything
        # else about the parent layer can be dropped, which is what keeps
        # residency at roughly a quarter of a layer rather than all of
        # them. float32 is ample: eta sits between about -20 and -5, so
        # the relative resolution is ~1e-7.
        parent_samples = np.ascontiguousarray(result.eta, dtype=np.float32)

    elapsed = time.time() - started
    if store is not None:
        store.write_manifest({
            "max_mer": config.max_mer,
            # growth, parameterization and branch_length are recorded so a
            # reader can rebuild the tree and describe the fit from the
            # manifest alone. Runs written before this had to infer growth
            # from the layer widths; see summarize._infer_growth.
            "growth": config.growth,
            "alternation": config.alternation,
            "linkage": config.linkage,
            "likelihood": config.likelihood,
            "parameterization": config.default_layer.parameterization,
            "branch_length": config.branch_length,
            "seed": config.seed,
            "count_name": table.name,
            "elapsed_seconds": elapsed,
            "layers": [
                {
                    "layer": r.layer,
                    "n_samples": int(r.rates.shape[0]),
                    "n_nodes": int(r.rates.shape[1]),
                    "n_empty_nodes": int(r.zero_mask.sum()),
                    "accept_rate_phi": float(r.accept_rate_phi),
                    "accept_rate_sigma": float(r.accept_rate_sigma),
                }
                for r in results
            ],
        })

    return ModelResult(
        config=config,
        tree=table.tree,
        layers=results,
        elapsed_seconds=elapsed,
        count_name=table.name,
        trace_store=store,
    )

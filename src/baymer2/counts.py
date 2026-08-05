"""Observed counts and the sufficient statistics the likelihood needs.

Why this module exists
----------------------
Every node in the context tree shares one rate vector across all of its
descendant k-mers, so its multinomial likelihood collapses onto four
pooled totals::

    log L_v = S[v,0] * log(1 - sum_j p_j) + sum_j S[v,j] * log(p_j)

    S[v, a] = sum over leaves u of node v of  n[u, a]

``S`` does not depend on any parameter, so it is a constant of the data.
Legacy Baymer nonetheless recomputed it inside the innermost Metropolis
loop -- ``get_edge_likelihood_numba`` evaluates ``np.sum(leaf_counts[a::4])``
for each of four ``a`` on every proposal, reading ``6 * 4**max_mer``
elements per sweep (~1.57M at ``max_mer=9``) to obtain numbers that never
change. Precomputing ``S`` makes each likelihood evaluation O(1).

Building ``S`` is likewise cheap. Children of node ``i`` occupy the
contiguous slice ``[4i : 4i+4]`` of the next layer, so the whole
pyramid falls out of successive 4-way reductions::

    S[max_mer-1] = per-leaf counts
    S[l]         = S[l+1].reshape(-1, 4, 4).sum(axis=1)

That is O(4**max_mer) once for *all* layers, replacing an
O(max_mer * 4**max_mer) re-enumeration per layer.

Slot convention
---------------
Counts are stored *reference-adjusted*: slot 0 always holds the
unmutated (reference/ancestral) count and slots 1..3 hold the three
alternate alleles in :func:`~baymer2.contexts.alt_bases` order. So an
``A``-focal context carries ``(A, A>C, A>G, A>T)`` and a ``C``-focal
context carries ``(C, C>A, C>G, C>T)``. Legacy Baymer achieved the same
thing with a hardcoded 0<->1 swap for ``C`` contexts
(``get_adjusted_leaf_counts``); this generalises it.

Counts are held as float64. That is exact for every integer a genome can
produce (float64 is exact to 2**53, ~9e15) and it lets the alignment
path carry *fractional* counts, which is what soft ancestral-state
weights from maximum-likelihood reconstruction produce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from baymer2.contexts import (
    ALTERNATION_PATTERNS,
    BASE_INDEX,
    ContextTree,
    alt_bases,
)

__all__ = [
    "CountTable",
    "adjust_to_reference_first",
    "check_alternation",
    "infer_alternation",
]


def infer_alternation(
    count_dicts: Mapping[int, Mapping[str, Sequence[float]]],
) -> set[str]:
    """Which alternation patterns are consistent with these count dicts.

    A count dict records the focal position of each context in field 6,
    and the two patterns schedule that position differently::

        k          1  2  3  4  5
        "right"    0  0  1  1  2
        "left"     0  1  1  2  2

    They only diverge at *even* k. So a set containing just odd k -- in
    particular a lone deepest-layer dict at odd ``max_mer``, which is the
    common case -- cannot distinguish the two, and this returns both.
    Callers that need certainty should load the whole 1..k family.
    """
    candidates = set(ALTERNATION_PATTERNS)
    for k, cd in count_dicts.items():
        if not cd:
            continue
        entry = next(iter(cd.values()))
        if len(entry) < 7:
            continue
        observed = int(entry[6])
        for pattern in list(candidates):
            if ContextTree(max_mer=max(count_dicts), alternation=pattern).focal_index(
                k
            ) != observed:
                candidates.discard(pattern)
    return candidates


def check_alternation(
    count_dicts: Mapping[int, Mapping[str, Sequence[float]]],
    tree: ContextTree,
) -> None:
    """Raise if ``tree``'s alternation is ruled out by these count dicts."""
    candidates = infer_alternation(count_dicts)
    if tree.alternation not in candidates:
        raise ValueError(
            f"count dicts are inconsistent with alternation "
            f"{tree.alternation!r}; they imply {sorted(candidates)}. The "
            f"count dicts were probably built with a different alternation "
            f"pattern than the model config declares."
        )


def adjust_to_reference_first(counts: Sequence[float], ref: str) -> list[float]:
    """Reorder ``[nA, nC, nG, nT]`` so the reference base sits in slot 0.

    Generalises legacy ``get_adjusted_leaf_counts``, which hardcoded the
    ``A``/``C`` swap.
    """
    return [counts[BASE_INDEX[ref]]] + [counts[BASE_INDEX[b]] for b in alt_bases(ref)]


@dataclass
class CountTable:
    """Deepest-layer counts plus the derived per-layer sufficient statistics.

    Parameters
    ----------
    tree:
        Geometry these counts are indexed against.
    leaf_counts:
        ``(n_leaves, 4)`` float array in ``tree.contexts(max_mer - 1)``
        order, reference-adjusted (slot 0 = reference).
    name:
        Optional label carried into outputs (population, species, ...).
    """

    tree: ContextTree
    leaf_counts: np.ndarray
    name: str = ""
    _stats: list[np.ndarray] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.leaf_counts = np.ascontiguousarray(self.leaf_counts, dtype=np.float64)
        expected = (self.tree.n_leaves, 4)
        if self.leaf_counts.shape != expected:
            raise ValueError(
                f"leaf_counts must have shape {expected} for max_mer="
                f"{self.tree.max_mer}, got {self.leaf_counts.shape}"
            )
        if np.any(self.leaf_counts < 0):
            raise ValueError("leaf_counts contains negative values")
        if not np.all(np.isfinite(self.leaf_counts)):
            raise ValueError("leaf_counts contains non-finite values")

    # -- construction -----------------------------------------------------

    @classmethod
    def from_count_dict(
        cls,
        count_dict: Mapping[str, Sequence[float]],
        tree: ContextTree,
        *,
        name: str = "",
        validate_totals: bool = True,
    ) -> "CountTable":
        """Build from a legacy-format folded count dict.

        Each entry is ``[nA, nC, nG, nT, total, ref_list_index,
        ref_context_index]``; only the first four are load-bearing, but
        the rest are cross-checked when ``validate_totals`` is set.
        """
        contexts = tree.contexts(tree.n_layers - 1)
        missing = [c for c in contexts if c not in count_dict]
        if missing:
            raise KeyError(
                f"count dict is missing {len(missing)} of {len(contexts)} "
                f"{tree.max_mer}-mer contexts, e.g. {missing[:5]}"
            )

        focal_idx = tree.focal_index(tree.max_mer)
        out = np.zeros((len(contexts), 4), dtype=np.float64)
        for i, ctx in enumerate(contexts):
            entry = count_dict[ctx]
            if len(entry) < 4:
                raise ValueError(f"entry for {ctx!r} has fewer than 4 counts: {entry}")
            ref = ctx[focal_idx]
            out[i] = adjust_to_reference_first(entry[:4], ref)

            if validate_totals and len(entry) >= 5:
                if not np.isclose(float(entry[4]), float(sum(entry[:4]))):
                    raise ValueError(
                        f"entry for {ctx!r} is inconsistent: counts sum to "
                        f"{sum(entry[:4])} but total field is {entry[4]}"
                    )
            if validate_totals and len(entry) >= 7:
                if int(entry[5]) != BASE_INDEX[ref]:
                    raise ValueError(
                        f"entry for {ctx!r} declares ref index {entry[5]} but "
                        f"the tree places {ref!r} (index {BASE_INDEX[ref]}) at "
                        f"the focal position"
                    )
                if int(entry[6]) != focal_idx:
                    raise ValueError(
                        f"entry for {ctx!r} declares focal position {entry[6]} "
                        f"but the tree uses {focal_idx}; the count dict was "
                        f"probably built with a different alternation pattern"
                    )
        return cls(tree=tree, leaf_counts=out, name=name)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        tree: ContextTree,
        *,
        name: str = "",
        validate_totals: bool = True,
    ) -> "CountTable":
        """Load a legacy ``*.folded_count_dict.json`` file."""
        raw = json.loads(Path(path).read_text())
        return cls.from_count_dict(
            raw, tree, name=name or Path(path).stem, validate_totals=validate_totals
        )

    @classmethod
    def from_count_dict_set(
        cls,
        count_dicts: Mapping[int, Mapping[str, Sequence[float]]],
        tree: ContextTree,
        *,
        name: str = "",
    ) -> "CountTable":
        """Build from the full 1..max_mer family of legacy count dicts.

        Only the deepest dict carries information -- the shallower ones
        are legacy's own down-tree marginalisation of it. They are worth
        loading anyway because they pin the alternation pattern, which a
        single deepest-layer dict cannot always do (see
        :func:`check_alternation`), and because disagreeing with them
        would mean the tree geometry is wrong.
        """
        check_alternation(count_dicts, tree)
        table = cls.from_count_dict(count_dicts[tree.max_mer], tree, name=name)
        for k, cd in sorted(count_dicts.items()):
            if k == tree.max_mer:
                continue
            focal_idx = tree.focal_index(k)
            got = table.layer_stats(k - 1)
            for i, ctx in enumerate(tree.contexts(k - 1)):
                if ctx not in cd:
                    raise KeyError(f"{k}mer count dict is missing context {ctx!r}")
                expected = adjust_to_reference_first(cd[ctx][:4], ctx[focal_idx])
                if not np.allclose(got[i], expected):
                    raise ValueError(
                        f"{k}mer count dict disagrees with the reduction of the "
                        f"{tree.max_mer}mer table at context {ctx!r}: "
                        f"{expected} vs {list(got[i])}"
                    )
        return table

    # -- sufficient statistics --------------------------------------------

    @property
    def stats(self) -> list[np.ndarray]:
        """Per-layer sufficient statistics; ``stats[l]`` is ``(2*4**l, 4)``.

        Computed once, by successive 4-way reduction up the tree.
        """
        if self._stats is None:
            self._stats = self._build_stats()
        return self._stats

    def _build_stats(self) -> list[np.ndarray]:
        n_layers = self.tree.n_layers
        branching = self.tree.branching
        stats: list[np.ndarray] = [None] * n_layers  # type: ignore[list-item]
        stats[n_layers - 1] = self.leaf_counts
        for layer in range(n_layers - 2, -1, -1):
            child = stats[layer + 1]
            stats[layer] = child.reshape(-1, branching, 4).sum(axis=1)
        return stats

    def layer_stats(self, layer: int) -> np.ndarray:
        """Sufficient statistics for ``layer``: ``(layer_size, 4)``."""
        return self.stats[layer]

    def zero_mask(self, layer: int) -> np.ndarray:
        """``True`` where a node has no observed opportunities at all.

        These nodes carry no information; the sampler must hold their
        parameters fixed and exclude them from the layer's priors. Legacy
        Baymer kept an index *list* and tested membership with ``in``,
        which numba lowers to a linear scan -- inside the innermost loop
        of ``update_alpha_probabilities``, that is O(layer_size * 3 *
        n_zero) per call. A boolean mask makes it O(1).
        """
        return self.stats[layer].sum(axis=1) == 0

    def n_opportunities(self, layer: int) -> np.ndarray:
        """Total observed sites per node in ``layer``."""
        return self.stats[layer].sum(axis=1)

    def empirical_rates(self, layer: int) -> np.ndarray:
        """Naive per-node alternate-allele fractions, ``(layer_size, 3)``.

        Used for initialisation and for empirical overlays in plots. Nodes
        with no opportunities come back as zeros.
        """
        s = self.stats[layer]
        total = s.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rates = np.where(total > 0, s[:, 1:] / total, 0.0)
        return rates

    # -- convenience ------------------------------------------------------

    @property
    def total_sites(self) -> float:
        return float(self.leaf_counts.sum())

    def layer_of(self, context: str) -> int:
        """Which layer a context of this length lives in."""
        k = len(context)
        if self.tree.centred:
            if k % 2 == 0:
                raise ValueError(
                    f"centred trees hold only odd-length contexts, got {context!r}"
                )
            return (k - 1) // 2
        return k - 1

    def context_index(self, context: str) -> int:
        """Array index of ``context`` within its layer."""
        return self.tree.contexts(self.layer_of(context)).index(context)

    def counts_for(self, context: str) -> np.ndarray:
        """Reference-adjusted 4-vector for a context at any depth."""
        return self.stats[self.layer_of(context)][self.context_index(context)]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CountTable(name={self.name!r}, max_mer={self.tree.max_mer}, "
            f"alternation={self.tree.alternation!r}, "
            f"total_sites={self.total_sites:.0f})"
        )


def stack_count_tables(tables: Iterable[CountTable]) -> np.ndarray:
    """Stack leaf counts from several tables into ``(n_tables, n_leaves, 4)``.

    Used by the multi-branch and multi-population fits; every table must
    share the same tree geometry.
    """
    tables = list(tables)
    if not tables:
        raise ValueError("no count tables given")
    first = tables[0].tree
    for t in tables[1:]:
        if (t.tree.max_mer, t.tree.alternation) != (first.max_mer, first.alternation):
            raise ValueError("count tables do not share a tree geometry")
    return np.stack([t.leaf_counts for t in tables])

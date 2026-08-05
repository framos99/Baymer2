"""Continuous-time Markov chain likelihood for substitution data.

Why
---
Observed focal-versus-ancestor differences understate true mutation.
Over a branch a site can be hit twice (A>C>G looks like A>G), can revert
(A>C>A looks like no change at all), and can saturate. The binomial
likelihood counts each site as one Bernoulli trial and therefore reads
the *apparent* divergence as the mutation rate. Modelling the branch as
a CTMC inverts that::

    log L = sum_b  N[a, b] * log( [exp(Q_c * t)]_{a, b} )

where ``a`` is the ancestral base, ``b`` the focal base, and ``Q_c`` the
context-conditional 4x4 instantaneous rate matrix.

Where the rates come from
-------------------------
Baymer parameterises three rates per *folded* context -- those out of
the ``A`` or ``C`` at the focal position. A full ``Q_c`` needs rates out
of all four bases with the *same flanks*, and folding supplies the
missing two: the rate out of ``G`` in context ``c`` is the rate out of
``C`` in ``revcomp(c)``, with target bases complemented. So ``Q_c``
assembles from four existing tree nodes and introduces **no new
parameters**.

Those four nodes form a closed group -- a "quartet" -- and the
likelihood is therefore quartet-local rather than node-local. That is a
real structural difference from the binomial mode, where each node's
likelihood depends only on itself.

.. important::

   The quartet only closes when the focal base is **centred**, i.e. when
   ``k`` is odd. Reverse-complementing moves the focal base from index
   ``fi`` to ``k-1-fi``, which is a member of the same layer only if
   ``k-1-fi == fi``. Under the alternating tree that holds at
   ``k = 1, 3, 5, 7, 9`` and fails at even ``k``, where 25% of the
   required (context, base) pairs are simply not represented::

       k=3  fi=1:    0 of  128 pairs unreachable
       k=4  fi=1:  128 of  512 pairs unreachable
       k=5  fi=2:    0 of 2048 pairs unreachable

   :func:`build_qmap` raises on even ``k`` rather than silently
   substituting something. See the module docs for the options.

Time scaling
------------
Only ``Q * t`` is identified from a single branch, so ``branch_length``
defaults to 1 and the rates are read as expected substitutions per site
along the focal branch. Supplying a branch length is useful only when
comparing branches on a shared rate scale. As ``t -> 0``,
``exp(Qt) -> I + Qt`` and this reduces exactly to the binomial mode --
:func:`~baymer2.likelihoods.binomial.BinomialLikelihood` is the limit,
not an alternative.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from baymer2.contexts import (
    BASE_INDEX,
    BASES,
    ContextTree,
    alt_bases,
    reverse_complement,
)
from baymer2.likelihoods.base import MODES, LikelihoodSpec

__all__ = ["CTMCLikelihood", "build_qmap", "expm4", "ctmc_layer_loglik"]

#: Taylor terms after scaling; ||B|| <= 0.5 converges to machine precision
#: well inside this.
_TAYLOR_TERMS = 18


def build_qmap(tree: ContextTree, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index maps letting a 4x4 ``Q`` be assembled from folded tree nodes.

    Returns
    -------
    q_src:
        ``(n, 4, 4)`` int32. ``q_src[i, a, b]`` is the node holding the
        rate for ``a -> b`` at node ``i``'s flanks. Diagonal is -1.
    q_slot:
        ``(n, 4, 4)`` int8. Which of that node's three rates to read.
    slot_base:
        ``(n, 4)`` int8. ``slot_base[i, s]`` is the base index that
        count slot ``s`` of node ``i`` refers to; slot 0 is the
        ancestral base itself.
    """
    k = tree.layer_k(layer)
    if k % 2 == 0:
        raise ValueError(
            f"the CTMC likelihood needs a centred focal base so that a "
            f"context's reverse complement stays in the same layer, but "
            f"layer {layer} of this tree holds even-length (k={k}) contexts. "
            f"Use growth='centred', which makes every layer odd-length and "
            f"reverse-complement closed."
        )
    fi = tree.focal_index(k)
    if k - 1 - fi != fi:
        raise ValueError(
            f"the CTMC likelihood needs a centred focal base so that a "
            f"context's reverse complement stays in the same layer, but "
            f"layer {layer} (k={k}) puts the focal base at index {fi} and "
            f"its reverse complement at {k - 1 - fi}. Use growth='centred'."
        )

    contexts = tree.contexts(layer)
    index = {c: i for i, c in enumerate(contexts)}
    n = len(contexts)

    q_src = np.full((n, 4, 4), -1, dtype=np.int32)
    q_slot = np.full((n, 4, 4), -1, dtype=np.int8)
    slot_base = np.zeros((n, 4), dtype=np.int8)

    for i, ctx in enumerate(contexts):
        left, right = ctx[:fi], ctx[fi + 1 :]

        focal = ctx[fi]
        slot_base[i, 0] = BASE_INDEX[focal]
        for s, b in enumerate(alt_bases(focal), start=1):
            slot_base[i, s] = BASE_INDEX[b]

        for a in BASES:
            substituted = left + a + right
            complemented = a not in ("A", "C")
            source = (
                reverse_complement(substituted) if complemented else substituted
            )
            src_index = index.get(source)
            if src_index is None:  # pragma: no cover - guarded by the check above
                raise ValueError(
                    f"context {source!r} is not in layer {layer}; the tree is "
                    f"not reverse-complement closed"
                )
            src_focal = source[fi]
            src_alts = alt_bases(src_focal)

            for b in BASES:
                if b == a:
                    continue
                target = reverse_complement(b) if complemented else b
                q_src[i, BASE_INDEX[a], BASE_INDEX[b]] = src_index
                q_slot[i, BASE_INDEX[a], BASE_INDEX[b]] = src_alts.index(target)

    return q_src, q_slot, slot_base


@njit(cache=True)
def expm4(a, out):
    """Matrix exponential of a 4x4, by scaling and squaring with Taylor.

    Chosen over eigendecomposition because a rate matrix can be
    defective and because this needs no LAPACK inside ``njit``. Scaling
    to ``||B||_inf <= 0.5`` makes the Taylor series converge to machine
    precision within :data:`_TAYLOR_TERMS`.
    """
    norm = 0.0
    for i in range(4):
        row = 0.0
        for j in range(4):
            row += abs(a[i, j])
        if row > norm:
            norm = row

    squarings = 0
    scale = 1.0
    while norm * scale > 0.5:
        scale *= 0.5
        squarings += 1

    b = np.empty((4, 4))
    for i in range(4):
        for j in range(4):
            b[i, j] = a[i, j] * scale

    term = np.zeros((4, 4))
    for i in range(4):
        term[i, i] = 1.0
    for i in range(4):
        for j in range(4):
            out[i, j] = term[i, j]

    nxt = np.empty((4, 4))
    for order in range(1, _TAYLOR_TERMS + 1):
        inv = 1.0 / order
        for i in range(4):
            for j in range(4):
                acc = 0.0
                for m in range(4):
                    acc += term[i, m] * b[m, j]
                nxt[i, j] = acc * inv
        for i in range(4):
            for j in range(4):
                term[i, j] = nxt[i, j]
                out[i, j] += nxt[i, j]

    for _ in range(squarings):
        for i in range(4):
            for j in range(4):
                acc = 0.0
                for m in range(4):
                    acc += out[i, m] * out[m, j]
                nxt[i, j] = acc
        for i in range(4):
            for j in range(4):
                out[i, j] = nxt[i, j]
    return out


@njit(cache=True)
def _assemble_q(node, rates, q_src, q_slot, t, q):
    """Fill ``q`` with the context's 4x4 generator, times ``t``."""
    for a in range(4):
        diag = 0.0
        for b in range(4):
            if a == b:
                continue
            src = q_src[node, a, b]
            slot = q_slot[node, a, b]
            rate = rates[src, slot]
            q[a, b] = rate * t
            diag += rate * t
        q[a, a] = -diag
    return q


@njit(parallel=True, cache=True)
def ctmc_layer_loglik(stats, rates, zero_mask, q_src, q_slot, slot_base, t, out):
    """Per-node CTMC log-likelihood for a whole layer."""
    n = stats.shape[0]
    for i in prange(n):
        if zero_mask[i]:
            out[i] = 0.0
            continue
        q = np.empty((4, 4))
        p = np.empty((4, 4))
        _assemble_q(i, rates, q_src, q_slot, t, q)
        expm4(q, p)

        ancestral = slot_base[i, 0]
        total = 0.0
        bad = False
        for s in range(4):
            count = stats[i, s]
            if count == 0.0:
                continue
            prob = p[ancestral, slot_base[i, s]]
            if prob <= 0.0:
                bad = True
                break
            total += count * np.log(prob)
        out[i] = -np.inf if bad else total


class CTMCLikelihood:
    """Branch-aware likelihood that corrects for multiple hits.

    Parameters
    ----------
    tree:
        Needed to build the quartet index maps.
    branch_length:
        Multiplies the rate matrix. Leave at 1 unless comparing
        branches; only ``Q * t`` is identified from a single branch.
    """

    name = "ctmc"

    def __init__(self, tree: ContextTree, branch_length: float = 1.0) -> None:
        if branch_length <= 0.0:
            raise ValueError(f"branch_length must be positive, got {branch_length}")
        self.tree = tree
        self.branch_length = float(branch_length)
        self._maps: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def maps(self, layer: int):
        if layer not in self._maps:
            self._maps[layer] = build_qmap(self.tree, layer)
        return self._maps[layer]

    def spec(self, layer: int) -> LikelihoodSpec:
        return LikelihoodSpec(
            mode=MODES[self.name],
            scalars=np.array([self.branch_length], dtype=np.float64),
        )

    def log_likelihood(
        self,
        stats: np.ndarray,
        rates: np.ndarray,
        *,
        layer: int | None = None,
        zero_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Readable reference implementation, in numpy."""
        stats = np.asarray(stats, dtype=np.float64)
        rates = np.asarray(rates, dtype=np.float64)
        if layer is None:
            layer = int(np.log(stats.shape[0] / 2) / np.log(4) + 0.5)
        q_src, q_slot, slot_base = self.maps(layer)
        if zero_mask is None:
            zero_mask = stats.sum(axis=1) == 0

        out = np.zeros(stats.shape[0], dtype=np.float64)
        out_view = np.zeros(stats.shape[0], dtype=np.float64)
        ctmc_layer_loglik(
            stats,
            rates,
            np.ascontiguousarray(zero_mask, dtype=np.bool_),
            q_src,
            q_slot,
            slot_base,
            self.branch_length,
            out_view,
        )
        out[:] = out_view
        return out

    def transition_matrix(self, rates: np.ndarray, node: int, layer: int) -> np.ndarray:
        """``exp(Q_c * t)`` for one node -- useful for inspection and tests."""
        q_src, q_slot, _ = self.maps(layer)
        q = np.zeros((4, 4))
        _assemble_q(node, np.asarray(rates, dtype=np.float64), q_src, q_slot,
                    self.branch_length, q)
        p = np.zeros((4, 4))
        expm4(q, p)
        return p


def build_quartets(
    tree: ContextTree, layer: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group nodes that share a generator, for quartet-local sampling.

    The CTMC likelihood of a node depends on three other nodes' rates,
    so a Metropolis update to one node changes the likelihood of its
    whole group. Sampling must therefore iterate over groups, not nodes:
    two members updated concurrently would race, and an acceptance ratio
    computed from one member alone would be wrong.

    Groups are closed -- every member's sources lie inside the group --
    so groups are mutually independent and safe to run in parallel.

    Returns
    -------
    members:
        ``(n_groups, 4)`` int32, padded with -1.
    sizes:
        ``(n_groups,)`` int8. Usually 4; smaller when a context's flanks
        are their own reverse complement, which makes members coincide.
    group_of:
        ``(n_nodes,)`` int32 mapping node to group.
    """
    q_src, _, _ = build_qmap(tree, layer)
    n = q_src.shape[0]
    off = ~np.eye(4, dtype=bool)

    group_of = np.full(n, -1, dtype=np.int32)
    members_list: list[list[int]] = []
    for i in range(n):
        if group_of[i] != -1:
            continue
        group = sorted(set(q_src[i][off].tolist()) | {i})
        gid = len(members_list)
        for m in group:
            group_of[m] = gid
        members_list.append(group)

    members = np.full((len(members_list), 4), -1, dtype=np.int32)
    sizes = np.zeros(len(members_list), dtype=np.int8)
    for gid, group in enumerate(members_list):
        members[gid, : len(group)] = group
        sizes[gid] = len(group)
    return members, sizes, group_of


@njit(inline="always", cache=True)
def ctmc_node_loglik(node, stats, rates, q_src, q_slot, slot_base, t, q, p):
    """CTMC log-likelihood for one node; ``q``/``p`` are 4x4 scratch."""
    total = stats[node, 0] + stats[node, 1] + stats[node, 2] + stats[node, 3]
    if total == 0.0:
        return 0.0
    _assemble_q(node, rates, q_src, q_slot, t, q)
    expm4(q, p)
    ancestral = slot_base[node, 0]
    acc = 0.0
    for s in range(4):
        count = stats[node, s]
        if count == 0.0:
            continue
        prob = p[ancestral, slot_base[node, s]]
        if prob <= 0.0:
            return -np.inf
        acc += count * np.log(prob)
    return acc

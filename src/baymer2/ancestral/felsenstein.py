"""Maximum-likelihood ancestral reconstruction with soft weights.

Parsimony returns a single base or nothing. That is fast and, on short
branches, close to optimal -- but it throws away two things. It has no
notion of *how* confident a call is, so a site resolved 51/49 counts the
same as one resolved 99/1; and it ignores branch lengths entirely, so a
distant outgroup carries as much weight as a close sister.

Felsenstein pruning gives the posterior distribution over the ancestral
base instead, using the branch lengths the alignment's own tree already
carries. Sites can then contribute *fractionally* to more than one
context, which is why :class:`~baymer2.counts.CountTable` holds floats.

Topology
--------
``Phylogeny.select`` produces a nested ladder::

    ((((focal, sister), o1), o2), ...)

with ``n1`` the focal's parent -- the node whose sequence the focal
branch mutated away from, and therefore the node whose state is wanted.
The posterior there is the product of the downward partial likelihood
from ``(focal, sister)`` and the upward partial likelihood carrying
everything rootward of it.

Model
-----
Jukes-Cantor by default: it needs nothing but branch lengths, and
reconstruction is far less sensitive to the substitution model than rate
*estimation* is. A custom rate matrix can be supplied when a better one
is available -- for instance one fitted by a first pass of the CTMC
likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from baymer2.ancestral.tree import Phylogeny, TrioSelection
from baymer2.contexts import ContextTree
from baymer2.likelihoods.ctmc import expm4

__all__ = [
    "SubstitutionModel",
    "LadderTopology",
    "MLAlignmentCounter",
    "ancestral_posteriors",
]


@dataclass
class SubstitutionModel:
    """A 4x4 generator plus its stationary distribution."""

    rate_matrix: np.ndarray
    stationary: np.ndarray

    def __post_init__(self) -> None:
        self.rate_matrix = np.ascontiguousarray(self.rate_matrix, dtype=np.float64)
        self.stationary = np.ascontiguousarray(self.stationary, dtype=np.float64)
        if self.rate_matrix.shape != (4, 4):
            raise ValueError(f"rate_matrix must be 4x4, got {self.rate_matrix.shape}")
        if self.stationary.shape != (4,):
            raise ValueError(f"stationary must be length 4, got {self.stationary.shape}")
        if not np.allclose(self.rate_matrix.sum(axis=1), 0.0, atol=1e-9):
            raise ValueError("rate_matrix rows must sum to zero")
        if not np.isclose(self.stationary.sum(), 1.0):
            raise ValueError("stationary must sum to one")

    @classmethod
    def jukes_cantor(cls) -> "SubstitutionModel":
        """Equal rates, uniform stationary distribution, unit expected rate."""
        q = np.full((4, 4), 1.0 / 3.0)
        np.fill_diagonal(q, -1.0)
        return cls(rate_matrix=q, stationary=np.full(4, 0.25))

    @classmethod
    def from_rates(cls, rates: np.ndarray) -> "SubstitutionModel":
        """Build from an arbitrary 4x4 of off-diagonal rates.

        The stationary distribution is obtained from the null space of
        ``Q.T``, which is where a reversible or non-reversible chain
        settles.
        """
        q = np.array(rates, dtype=np.float64)
        np.fill_diagonal(q, 0.0)
        np.fill_diagonal(q, -q.sum(axis=1))
        values, vectors = np.linalg.eig(q.T)
        index = int(np.argmin(np.abs(values)))
        pi = np.real(vectors[:, index])
        pi = np.abs(pi)
        pi /= pi.sum()
        return cls(rate_matrix=q, stationary=pi)

    def transition(self, t: float) -> np.ndarray:
        """``exp(Q t)``."""
        if t < 0:
            raise ValueError(f"branch length must be non-negative, got {t}")
        out = np.zeros((4, 4))
        expm4(np.ascontiguousarray(self.rate_matrix * t), out)
        return out


@dataclass
class LadderTopology:
    """Transition matrices for one nested comparator ladder.

    ``focal`` and ``sister`` hang off ``n1``; ``outgroups[k]`` hangs off
    ``n_{k+2}``, reached from ``n_{k+1}`` by ``internals[k]``.
    """

    focal: np.ndarray
    sister: np.ndarray
    internals: np.ndarray  # (m, 4, 4)
    outgroups: np.ndarray  # (m, 4, 4)
    stationary: np.ndarray

    @classmethod
    def build(
        cls,
        phylogeny: Phylogeny,
        selection: TrioSelection,
        model: SubstitutionModel | None = None,
        *,
        scale: float = 1.0,
    ) -> "LadderTopology":
        model = model or SubstitutionModel.jukes_cantor()
        lengths = phylogeny.ladder(selection)
        m = len(selection.outgroups)
        return cls(
            focal=model.transition(lengths["focal_length"] * scale),
            sister=model.transition(lengths["sister_length"] * scale),
            internals=np.stack(
                [model.transition(l * scale) for l in lengths["internal_lengths"]]
            ) if m else np.zeros((0, 4, 4)),
            outgroups=np.stack(
                [model.transition(l * scale) for l in lengths["outgroup_lengths"]]
            ),
            stationary=model.stationary.copy(),
        )


@njit(cache=True)
def _posterior_at_n1(
    focal_base, sister_base, outgroup_bases, n_out,
    p_focal, p_sister, p_internal, p_outgroup, pi, out,
):
    """Posterior over the ancestral base at ``n1``, by pruning.

    Returns ``False`` when any observation is missing, in which case
    ``out`` is not meaningful.
    """
    if focal_base == 255 or sister_base == 255:
        return False
    for i in range(n_out):
        if outgroup_bases[i] == 255:
            return False

    # Downward partial likelihood at n1 from (focal, sister).
    down_n1 = np.empty(4)
    for x in range(4):
        down_n1[x] = p_focal[x, focal_base] * p_sister[x, sister_base]

    # Downward partial likelihoods at each successive internal node.
    downs = np.empty((n_out + 1, 4))
    for x in range(4):
        downs[0, x] = down_n1[x]
    for k in range(n_out):
        for y in range(4):
            acc = 0.0
            for x in range(4):
                acc += p_internal[k, y, x] * downs[k, x]
            downs[k + 1, y] = acc * p_outgroup[k, y, outgroup_bases[k]]

    # Upward partial likelihoods, starting from the stationary root.
    up = np.empty(4)
    for y in range(4):
        up[y] = pi[y]

    for k in range(n_out - 1, -1, -1):
        nxt = np.empty(4)
        for x in range(4):
            acc = 0.0
            for y in range(4):
                acc += (
                    up[y]
                    * p_outgroup[k, y, outgroup_bases[k]]
                    * p_internal[k, y, x]
                )
            nxt[x] = acc
        for x in range(4):
            up[x] = nxt[x]

    total = 0.0
    for x in range(4):
        out[x] = down_n1[x] * up[x]
        total += out[x]
    if total <= 0.0:
        return False
    for x in range(4):
        out[x] /= total
    return True


def ancestral_posteriors(
    focal: str,
    sister: str,
    outgroups: list[str],
    topology: LadderTopology,
) -> np.ndarray | None:
    """Posterior over the ancestral base for one site, as a length-4 array."""
    from baymer2.ancestral.parsimony import _CODE

    codes = [_CODE[ord(b)] for b in (focal, sister, *outgroups)]
    out = np.zeros(4)
    ok = _posterior_at_n1(
        np.uint8(codes[0]),
        np.uint8(codes[1]),
        np.asarray(codes[2:], dtype=np.uint8),
        len(outgroups),
        topology.focal,
        topology.sister,
        topology.internals,
        topology.outgroups,
        topology.stationary,
        out,
    )
    return out if ok else None


@njit(cache=True)
def _count_block_ml(
    focal_seq, other_seqs, n_other, k,
    p_focal, p_sister, p_internal, p_outgroup, pi,
    node_of, complemented, folded_focal, slot_table,
    flank_threshold, min_weight,
    counts, n_dropped_flank,
):
    """Accumulate one block using soft ancestral weights.

    Flanking positions are hard-called above ``flank_threshold`` -- a
    k-mer has ``4**(k-1)`` flank combinations, so carrying them all
    softly is not tractable and is not worth it: flanks only need to be
    *known*, whereas the centre is where the mutation call lives. The
    centre's posterior is spread across every candidate base above
    ``min_weight``, so an ambiguous site contributes fractionally to
    several contexts rather than being discarded.
    """
    n_cols = focal_seq.shape[0]
    posteriors = np.zeros((n_cols, 4))
    best_base = np.full(n_cols, 255, dtype=np.uint8)
    best_prob = np.zeros(n_cols)
    focal_bases = np.empty(n_cols, dtype=np.uint8)
    n_sites = 0

    outgroups = np.empty(n_other - 1, dtype=np.uint8)
    site_post = np.empty(4)

    for col in range(n_cols):
        fb = focal_seq[col]
        if fb == 255:
            continue

        sb = other_seqs[0, col]
        for r in range(1, n_other):
            outgroups[r - 1] = other_seqs[r, col]

        ok = _posterior_at_n1(
            fb, sb, outgroups, n_other - 1,
            p_focal, p_sister, p_internal, p_outgroup, pi, site_post,
        )
        focal_bases[n_sites] = fb
        if ok:
            top = 0
            for x in range(4):
                posteriors[n_sites, x] = site_post[x]
                if site_post[x] > site_post[top]:
                    top = x
            best_base[n_sites] = top
            best_prob[n_sites] = site_post[top]
        else:
            best_prob[n_sites] = 0.0
        n_sites += 1

    half = k // 2
    for i in range(half, n_sites - half):
        if best_prob[i] <= 0.0:
            continue

        flanks_ok = True
        for j in range(i - half, i + half + 1):
            if j == i:
                continue
            if best_prob[j] < flank_threshold:
                flanks_ok = False
                break
        if not flanks_ok:
            n_dropped_flank[0] += 1
            continue

        for centre in range(4):
            weight = posteriors[i, centre]
            if weight < min_weight:
                continue
            code = 0
            for j in range(i - half, i + half + 1):
                base = centre if j == i else best_base[j]
                code = code * 4 + base
            node = node_of[code]
            observed = focal_bases[i]
            if complemented[code]:
                observed = 3 - observed
            slot = slot_table[folded_focal[code], observed]
            counts[node, slot] += weight


@dataclass
class MLAlignmentCounter:
    """Accumulate context counts using maximum-likelihood ancestral states.

    Differs from :class:`~baymer2.ancestral.parsimony.AlignmentCounter`
    in two ways that matter: branch lengths inform the reconstruction,
    and sites contribute fractionally rather than all-or-nothing. The
    second is why sites parsimony calls ``unresolved`` still carry
    information here.
    """

    tree: "ContextTree"
    focal: str
    sister: str
    outgroups: list
    topology: LadderTopology
    flank_threshold: float = 0.95
    min_weight: float = 0.01
    counts: np.ndarray = None
    n_blocks: int = 0
    n_dropped_flank: np.ndarray = None

    def __post_init__(self) -> None:
        from baymer2.ancestral.parsimony import build_kmer_lookup

        if self.tree.max_mer % 2 == 0:
            raise ValueError(
                f"alignment counting builds a centred k-mer, so max_mer must "
                f"be odd; got {self.tree.max_mer}"
            )
        if not 0.0 < self.flank_threshold <= 1.0:
            raise ValueError(
                f"flank_threshold must be in (0, 1], got {self.flank_threshold}"
            )
        self.counts = np.zeros((self.tree.n_leaves, 4), dtype=np.float64)
        self.n_dropped_flank = np.zeros(1, dtype=np.int64)
        self._lookup = build_kmer_lookup(self.tree)

    @classmethod
    def from_selection(
        cls, tree, phylogeny, selection, model=None, **kwargs
    ) -> "MLAlignmentCounter":
        return cls(
            tree=tree,
            focal=selection.focal,
            sister=selection.sister,
            outgroups=list(selection.outgroups),
            topology=LadderTopology.build(phylogeny, selection, model),
            **kwargs,
        )

    def add_block(self, block: dict) -> None:
        from baymer2.ancestral.parsimony import _SLOT_TABLE, _encode

        order = [self.sister, *self.outgroups]
        focal_seq = _encode(block[self.focal].seq)
        others = np.stack([_encode(block[g].seq) for g in order])
        node_of, complemented, folded_focal = self._lookup

        _count_block_ml(
            focal_seq, others, others.shape[0], self.tree.max_mer,
            self.topology.focal, self.topology.sister,
            self.topology.internals, self.topology.outgroups,
            self.topology.stationary,
            node_of, complemented, folded_focal, _SLOT_TABLE,
            self.flank_threshold, self.min_weight,
            self.counts, self.n_dropped_flank,
        )
        self.n_blocks += 1

    def add_blocks(self, blocks) -> "MLAlignmentCounter":
        for block in blocks:
            self.add_block(block)
        return self

    def to_count_table(self, name: str = ""):
        from baymer2.counts import CountTable

        return CountTable(
            tree=self.tree, leaf_counts=self.counts, name=name or self.focal
        )

    @property
    def divergence(self) -> float:
        total = self.counts.sum()
        return float(self.counts[:, 1:].sum() / total) if total else 0.0

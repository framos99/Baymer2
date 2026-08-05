"""Fitch parsimony over a focal genome and its nested outgroups.

Generalises ``call_block`` from
``trio_alignment_and_polarization.py`` from a hard-coded three-taxon
rule to arbitrarily many nested outgroups, while reproducing the
three-taxon behaviour exactly (see ``tests/test_parsimony.py``), so the
27 species already processed stay comparable.

Topology
--------
Comparators come from :meth:`~baymer2.ancestral.tree.Phylogeny.select`,
which walks rootward and takes one representative per successively more
distant clade. That gives the nested ladder Fitch expects::

    ((((focal, sister), outgroup1), outgroup2), ...)

The state of interest is at the focal's immediate ancestor -- the MRCA
of focal and sister -- because that is the sequence the focal branch
mutated away from.

Ancestral context, not focal context
------------------------------------
Context k-mers are built from the *reconstructed ancestral* sequence
rather than the focal one. The existing script tallies focal k-mers
(``_focal_kmer`` over ``ungapped_focal``), which agrees at invariant
sites but differs at exactly the sites that carry a substitution, and
the error is systematic: a CpG that mutated to TpG is counted as a
``TpG`` opportunity when the mutation actually occurred in a ``CpG``.
Since CpG is where context dependence is strongest, that bias lands
precisely where it does the most damage.

A site is counted only when the *whole* ancestral k-mer is resolved, so
an unresolved neighbour removes the site rather than contaminating it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from baymer2.contexts import BASES, ContextTree, reverse_complement
from baymer2.counts import CountTable

__all__ = [
    "CATEGORIES",
    "AlignmentCounter",
    "build_kmer_lookup",
    "fitch_ancestral",
]

#: Site classes, matching the vocabulary of the existing trio caller so
#: its outputs stay directly comparable.
CATEGORIES = (
    "invariant",
    "derived_on_focal",
    "derived_on_sister",
    "derived_on_outgroup",
    "unresolved",
)

_INVARIANT, _FOCAL, _SISTER, _OUTGROUP, _UNRESOLVED = range(5)

# Base codes; 255 marks anything unusable (gap, N, soft-mask rejected).
_CODE = np.full(256, 255, dtype=np.uint8)
for _i, _b in enumerate(BASES):
    _CODE[ord(_b)] = _i
    _CODE[ord(_b.lower())] = _i

#: slot_table[folded_ancestral, observed] -> count slot.
#: Slot 0 is the unmutated count; 1..3 follow alt_bases order.
_SLOT_TABLE = np.array(
    [
        [0, 1, 2, 3],  # ancestral A: alts C, G, T
        [1, 0, 2, 3],  # ancestral C: alts A, G, T
    ],
    dtype=np.int8,
)


@njit(inline="always", cache=True)
def _fitch_masks(focal, sister, outgroups, n_out):
    """Fitch down-pass then up-pass; returns the MRCA(focal, sister) state.

    Returns a base index, or 255 when the ancestral state is ambiguous.
    """
    f = np.uint8(1) << focal
    s = np.uint8(1) << sister

    node = f & s
    if node == 0:
        node = f | s
    first = node

    # Down-pass through the nested outgroups, recording each node so the
    # up-pass can resolve ambiguity from the rootward side.
    states = np.empty(n_out + 1, dtype=np.uint8)
    states[0] = first
    for i in range(n_out):
        o = np.uint8(1) << outgroups[i]
        nxt = states[i] & o
        if nxt == 0:
            nxt = states[i] | o
        states[i + 1] = nxt

    # Up-pass: intersect with the parent state where that disambiguates.
    resolved = states[n_out]
    for i in range(n_out - 1, -1, -1):
        candidate = states[i] & resolved
        resolved = candidate if candidate != 0 else states[i]

    # A single set bit means an unambiguous ancestral base.
    if resolved == 1:
        return np.uint8(0)
    if resolved == 2:
        return np.uint8(1)
    if resolved == 4:
        return np.uint8(2)
    if resolved == 8:
        return np.uint8(3)
    return np.uint8(255)


def fitch_ancestral(focal: str, sister: str, outgroups: list[str]) -> str | None:
    """Python-level Fitch call, for testing and inspection."""
    codes = [_CODE[ord(b)] for b in (focal, sister, *outgroups)]
    if any(c == 255 for c in codes):
        return None
    out = np.asarray(codes[2:], dtype=np.uint8)
    result = _fitch_masks(np.uint8(codes[0]), np.uint8(codes[1]), out, len(out))
    return None if result == 255 else BASES[result]


def build_kmer_lookup(tree: ContextTree) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map every k-mer to its folded node, complement flag, and focal base.

    A dense table over all ``4**k`` k-mers turns folding and index lookup
    into two array reads inside the counting loop, instead of string
    manipulation per site.
    """
    k = tree.max_mer
    layer = tree.n_layers - 1
    contexts = tree.contexts(layer)
    index = {c: i for i, c in enumerate(contexts)}
    focal_idx = tree.focal_index(k)

    size = 4**k
    node_of = np.full(size, -1, dtype=np.int32)
    complemented = np.zeros(size, dtype=np.bool_)
    folded_focal = np.full(size, -1, dtype=np.int8)

    for code in range(size):
        digits = []
        rest = code
        for _ in range(k):
            digits.append(rest & 3)
            rest >>= 2
        digits.reverse()
        kmer = "".join(BASES[d] for d in digits)

        if kmer[focal_idx] in ("A", "C"):
            folded, comp = kmer, False
        else:
            folded, comp = reverse_complement(kmer), True
        node_of[code] = index[folded]
        complemented[code] = comp
        folded_focal[code] = 0 if folded[focal_idx] == "A" else 1

    return node_of, complemented, folded_focal


@njit(cache=True)
def _count_block(
    focal_seq, other_seqs, n_other, k, focal_offset,
    node_of, complemented, folded_focal, slot_table,
    counts, category_totals,
):
    """Accumulate one MAF block into ``counts``.

    ``other_seqs`` is ``(n_other, n_columns)`` with the sister first.
    """
    n_cols = focal_seq.shape[0]

    # Pass 1: ungapped focal positions, with the ancestral base per site.
    focal_bases = np.empty(n_cols, dtype=np.uint8)
    anc_bases = np.empty(n_cols, dtype=np.uint8)
    n_sites = 0
    outgroups = np.empty(n_other - 1, dtype=np.uint8)

    for col in range(n_cols):
        fb = focal_seq[col]
        if fb == 255:
            # Focal gap or ambiguous: the focal has no position here at
            # all, so it contributes nothing and consumes no coordinate.
            if focal_seq[col] == 255 and focal_offset >= 0:
                pass
            continue

        usable = True
        for r in range(n_other):
            if other_seqs[r, col] == 255:
                usable = False
                break

        if not usable:
            focal_bases[n_sites] = fb
            anc_bases[n_sites] = 255
            n_sites += 1
            continue

        sb = other_seqs[0, col]
        for r in range(1, n_other):
            outgroups[r - 1] = other_seqs[r, col]
        anc = _fitch_masks(fb, sb, outgroups, n_other - 1)

        focal_bases[n_sites] = fb
        anc_bases[n_sites] = anc
        n_sites += 1

        if anc == 255:
            category_totals[4] += 1
        elif fb != anc:
            category_totals[1] += 1
        elif sb != anc:
            category_totals[2] += 1
        else:
            derived_out = False
            for r in range(1, n_other):
                if other_seqs[r, col] != anc:
                    derived_out = True
                    break
            category_totals[3 if derived_out else 0] += 1

    # Pass 2: k-mers of the ancestral sequence, in focal coordinates.
    half = k // 2
    for i in range(half, n_sites - half):
        if anc_bases[i] == 255:
            continue
        code = 0
        ok = True
        for j in range(i - half, i + half + 1):
            base = anc_bases[j]
            if base == 255:
                ok = False
                break
            code = code * 4 + base
        if not ok:
            continue

        node = node_of[code]
        observed = focal_bases[i]
        if complemented[code]:
            observed = 3 - observed  # A<->T, C<->G under this ordering
        slot = slot_table[folded_focal[code], observed]
        counts[node, slot] += 1.0


@dataclass
class AlignmentCounter:
    """Accumulate context counts from MAF blocks by parsimony.

    Parameters
    ----------
    tree:
        Must be reverse-complement closed (``growth="centred"``) if the
        counts will be fitted under the CTMC likelihood.
    focal, sister, outgroups:
        Genome names as they appear in the MAF ``src`` fields.
    """

    tree: ContextTree
    focal: str
    sister: str
    outgroups: list[str]
    counts: np.ndarray = field(default=None, repr=False)
    category_totals: np.ndarray = field(default=None, repr=False)
    n_blocks: int = 0

    def __post_init__(self) -> None:
        if self.tree.max_mer % 2 == 0:
            raise ValueError(
                f"alignment counting builds a centred k-mer around each site, "
                f"so max_mer must be odd; got {self.tree.max_mer}"
            )
        if not self.outgroups:
            raise ValueError("at least one outgroup is required for parsimony")
        n_nodes = self.tree.n_leaves
        self.counts = np.zeros((n_nodes, 4), dtype=np.float64)
        self.category_totals = np.zeros(5, dtype=np.int64)
        self._lookup = build_kmer_lookup(self.tree)

    @property
    def genomes(self) -> list[str]:
        return [self.focal, self.sister, *self.outgroups]

    def add_block(self, block: dict) -> None:
        """Accumulate one block keyed by genome name."""
        order = [self.sister, *self.outgroups]
        focal_seq = _encode(block[self.focal].seq)
        others = np.stack([_encode(block[g].seq) for g in order])

        node_of, complemented, folded_focal = self._lookup
        _count_block(
            focal_seq, others, others.shape[0], self.tree.max_mer,
            block[self.focal].start,
            node_of, complemented, folded_focal, _SLOT_TABLE,
            self.counts, self.category_totals,
        )
        self.n_blocks += 1

    def add_blocks(self, blocks) -> "AlignmentCounter":
        for block in blocks:
            self.add_block(block)
        return self

    def to_count_table(self, name: str = "") -> CountTable:
        return CountTable(
            tree=self.tree,
            leaf_counts=self.counts,
            name=name or self.focal,
        )

    def category_summary(self) -> dict[str, int]:
        return dict(zip(CATEGORIES, (int(v) for v in self.category_totals)))

    @property
    def divergence(self) -> float:
        """Fraction of resolved sites carrying a focal-branch substitution.

        The apparent divergence -- what a binomial fit would read as the
        rate, and what the CTMC likelihood exists to correct.
        """
        total = self.counts.sum()
        return float(self.counts[:, 1:].sum() / total) if total else 0.0


def _encode(seq: str) -> np.ndarray:
    """ASCII sequence to base codes, with 255 for gaps and ambiguity."""
    return _CODE[np.frombuffer(seq.encode(), dtype=np.uint8)]

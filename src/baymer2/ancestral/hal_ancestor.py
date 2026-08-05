"""Read the ancestral sequence Cactus already reconstructed.

Progressive Cactus reconstructs an ancestral genome at every internal
node and stores it in the HAL alongside the extant ones. The VGP 577-way
alignment carries 576 of them -- one per internal node, every one
present as a genome, verified by probing the file on math1:

    577 leaves + 576 named internal nodes = 1153 genomes

So the focal branch's ancestor can simply be *read* rather than
re-inferred from a trio. That is cheaper -- no outgroup needs to be
pulled into the MAF, and no per-site inference runs -- and it uses the
full alignment's evidence rather than three genomes' worth.

The trade-off is that Cactus's reconstruction is a fixed input: its
errors cannot be inspected or re-weighted here, and it was produced
under Cactus's own model rather than one fitted to these data. Where
that matters, ``--method ml`` reconstructs from the trio with explicit,
inspectable posteriors instead.

Contexts come from the ancestral sequence, which here is exactly what
the HAL provides -- no reconstruction step stands between the alignment
and the k-mer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from baymer2.ancestral.parsimony import _CODE, _SLOT_TABLE, _encode, build_kmer_lookup
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable

__all__ = ["AncestorAlignmentCounter"]


@njit(cache=True)
def _count_block_ancestor(
    focal_seq, ancestor_seq, k,
    node_of, complemented, folded_focal, slot_table,
    counts, tallies,
):
    """Accumulate one block against a supplied ancestral sequence.

    ``tallies`` collects (aligned, substituted, unusable) so the caller
    can report an apparent divergence without a second pass.
    """
    n_cols = focal_seq.shape[0]
    focal_bases = np.empty(n_cols, dtype=np.uint8)
    anc_bases = np.empty(n_cols, dtype=np.uint8)
    n_sites = 0

    for col in range(n_cols):
        fb = focal_seq[col]
        if fb == 255:
            continue
        ab = ancestor_seq[col]
        focal_bases[n_sites] = fb
        anc_bases[n_sites] = ab
        n_sites += 1
        if ab == 255:
            tallies[2] += 1
        else:
            tallies[0] += 1
            if fb != ab:
                tallies[1] += 1

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
            observed = 3 - observed
        counts[node, slot_table[folded_focal[code], observed]] += 1.0


@dataclass
class AncestorAlignmentCounter:
    """Count substitutions against the HAL's own reconstructed ancestor.

    Needs only the focal genome and its ancestor in each MAF block, so
    the extraction is a duo rather than a trio.
    """

    tree: ContextTree
    focal: str
    ancestor: str
    counts: np.ndarray = field(default=None, repr=False)
    tallies: np.ndarray = field(default=None, repr=False)
    n_blocks: int = 0

    def __post_init__(self) -> None:
        if self.tree.max_mer % 2 == 0:
            raise ValueError(
                f"alignment counting builds a centred k-mer, so max_mer must "
                f"be odd; got {self.tree.max_mer}"
            )
        if not self.ancestor:
            raise ValueError(
                "an ancestor genome name is required; the tree must name the "
                "focal's parent and the HAL must carry it as a genome"
            )
        self.counts = np.zeros((self.tree.n_leaves, 4), dtype=np.float64)
        self.tallies = np.zeros(3, dtype=np.int64)
        self._lookup = build_kmer_lookup(self.tree)

    @classmethod
    def from_selection(cls, tree: ContextTree, selection) -> "AncestorAlignmentCounter":
        return cls(tree=tree, focal=selection.focal, ancestor=selection.ancestor)

    @property
    def genomes(self) -> list[str]:
        return [self.focal, self.ancestor]

    def add_block(self, block: dict) -> None:
        focal_seq = _encode(block[self.focal].seq)
        ancestor_seq = _encode(block[self.ancestor].seq)
        if ancestor_seq.shape != focal_seq.shape:
            raise ValueError(
                f"block rows differ in length: focal {focal_seq.shape[0]} vs "
                f"ancestor {ancestor_seq.shape[0]}; MAF rows of one block must "
                f"be aligned"
            )
        node_of, complemented, folded_focal = self._lookup
        _count_block_ancestor(
            focal_seq, ancestor_seq, self.tree.max_mer,
            node_of, complemented, folded_focal, _SLOT_TABLE,
            self.counts, self.tallies,
        )
        self.n_blocks += 1

    def add_blocks(self, blocks) -> "AncestorAlignmentCounter":
        for block in blocks:
            self.add_block(block)
        return self

    def to_count_table(self, name: str = "") -> CountTable:
        return CountTable(
            tree=self.tree, leaf_counts=self.counts, name=name or self.focal
        )

    def category_summary(self) -> dict[str, int]:
        aligned, substituted, unusable = (int(v) for v in self.tallies)
        return {
            "invariant": aligned - substituted,
            "derived_on_focal": substituted,
            "unresolved": unusable,
        }

    @property
    def divergence(self) -> float:
        total = self.counts.sum()
        return float(self.counts[:, 1:].sum() / total) if total else 0.0

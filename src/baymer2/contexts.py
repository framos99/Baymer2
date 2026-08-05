"""The nested k-mer context tree.

This module is the single source of truth for context enumeration,
strand folding, and array indexing. In legacy Baymer the same rules were
reimplemented in ``context_counter.py``, ``generate_count_json.py`` and
``run_baymer.py``, and they drifted: ``run_baymer.py`` honours
``alternation_pattern: "left"`` while ``generate_count_json.py`` never
passes the flag through, so left-alternating count dicts cannot be built
by the documented pipeline at all. Everything here is derived once and
consumed everywhere.

Layout
------
Layer ``l`` holds contexts of length ``k = l + 1``; there are
``2 * 4**l`` of them. The leading factor of two is the strand-folded
focal base, which is always ``A`` or ``C`` -- contexts centred on ``G``
or ``T`` are reverse-complemented into their equivalents.

Children of node ``i`` occupy the contiguous slice ``[4*i : 4*i+4]`` of
the next layer, a flat 4-ary heap. That contiguity is what lets
:mod:`baymer2.counts` reduce leaf counts up the tree with a reshape
instead of re-enumerating leaves per layer.

Growth
------
``"alternating"`` (the legacy Baymer structure) adds one flanking base
per layer, alternating sides so the focal base migrates toward the
centre. Under the default ``"right"`` pattern a 5-mer model ends with
the focal base at index 2::

    layer 0   A                 focal index 0
    layer 1   A N               focal index 0    (appended right)
    layer 2   N A N             focal index 1    (prepended left)
    layer 3   N A N N           focal index 1    (appended right)
    layer 4   N N A N N         focal index 2    (prepended left)

``"left"`` mirrors the schedule.

``"centred"`` adds one base on *each* side per layer, so ``k`` is always
odd and the focal base is always central::

    layer 0   A                 focal index 0     2 nodes
    layer 1   N A N             focal index 1    32 nodes
    layer 2   N N A N N         focal index 2   512 nodes

This exists because the substitution (CTMC) likelihood needs the rates
out of all four bases at a given set of flanks, and folding only
supplies the ``G``/``T`` rows when reverse-complementing keeps the focal
base in the same position -- that is, when ``k - 1 - focal == focal``,
i.e. when ``k`` is odd. Under alternating growth that holds only on
alternate layers; under centred growth it holds everywhere.

Centred growth is coarser -- 16 children per node rather than 4 -- but
it is not smaller: it reaches the *same* leaf set. A 9-mer model has
``2 * 4**8 = 131,072`` leaves either way, in 5 centred layers instead of
9 alternating ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

BASES: tuple[str, ...] = ("A", "C", "G", "T")
BASE_INDEX: dict[str, int] = {b: i for i, b in enumerate(BASES)}

#: Focal bases retained after strand folding.
FOLDED_BASES: tuple[str, ...] = ("A", "C")

_COMPLEMENT = str.maketrans("ACGTacgtN", "TGCAtgcaN")

ALTERNATION_PATTERNS = ("right", "left")
GROWTH_MODES = ("alternating", "centred")


def reverse_complement(seq: str) -> str:
    """Reverse complement of ``seq``, preserving case and passing ``N``."""
    return seq.translate(_COMPLEMENT)[::-1]


def alt_bases(ref: str) -> tuple[str, str, str]:
    """The three non-reference bases, in the order the model uses.

    The rate vector ``p`` for a node is ordered against this tuple, so
    an ``A``-focal context carries ``(A>C, A>G, A>T)`` and a ``C``-focal
    context carries ``(C>A, C>G, C>T)``.
    """
    if ref not in BASE_INDEX:
        raise ValueError(f"not a base: {ref!r}")
    return tuple(b for b in BASES if b != ref)  # type: ignore[return-value]


@dataclass(frozen=True)
class ContextTree:
    """Geometry of the nested k-mer tree for a given depth and pattern.

    Parameters
    ----------
    max_mer:
        Length of the deepest context, i.e. the number of layers.
    alternation:
        ``"right"`` (default) or ``"left"``; which side gains the first
        flanking base.
    """

    max_mer: int
    alternation: str = "right"
    growth: str = "alternating"

    def __post_init__(self) -> None:
        if self.max_mer < 1:
            raise ValueError(f"max_mer must be >= 1, got {self.max_mer}")
        if self.alternation not in ALTERNATION_PATTERNS:
            raise ValueError(
                f"alternation must be one of {ALTERNATION_PATTERNS}, "
                f"got {self.alternation!r}"
            )
        if self.growth not in GROWTH_MODES:
            raise ValueError(
                f"growth must be one of {GROWTH_MODES}, got {self.growth!r}"
            )
        if self.growth == "centred" and self.max_mer % 2 == 0:
            raise ValueError(
                f"centred growth keeps the focal base central, so max_mer must "
                f"be odd; got {self.max_mer}"
            )

    # -- basic geometry ---------------------------------------------------

    @property
    def centred(self) -> bool:
        return self.growth == "centred"

    @property
    def branching(self) -> int:
        """Children per node: 4 when adding one base, 16 when adding two."""
        return 16 if self.centred else 4

    @property
    def n_layers(self) -> int:
        return (self.max_mer + 1) // 2 if self.centred else self.max_mer

    def layer_k(self, layer: int) -> int:
        """Context length at ``layer``."""
        return 2 * layer + 1 if self.centred else layer + 1

    def layer_size(self, layer: int) -> int:
        """Number of nodes in ``layer`` (``2 * branching**layer``)."""
        return 2 * self.branching**layer

    @property
    def n_leaves(self) -> int:
        """Number of deepest-layer contexts."""
        return self.layer_size(self.n_layers - 1)

    def _appends_right(self, parent_len: int) -> bool:
        """Whether a child of a length-``parent_len`` context is suffixed."""
        odd = parent_len % 2
        if self.alternation == "right":
            return bool(odd)
        return not odd

    def focal_index(self, k: int) -> int:
        """Index of the focal base within a length-``k`` context."""
        if not 1 <= k <= self.max_mer:
            raise ValueError(f"k must be in [1, {self.max_mer}], got {k}")
        if self.centred:
            if k % 2 == 0:
                raise ValueError(
                    f"centred growth only produces odd-length contexts, got k={k}"
                )
            return (k - 1) // 2
        idx = k // 2
        if k % 2 == 0 and self.alternation == "right":
            idx -= 1
        return idx

    @property
    def reverse_complement_closed(self) -> bool:
        """Whether every layer's contexts survive reverse complementing.

        Required by the CTMC likelihood, which must read the ``G`` and
        ``T`` rows of a generator off the opposite strand.
        """
        return all(
            self.layer_k(l) - 1 - self.focal_index(self.layer_k(l))
            == self.focal_index(self.layer_k(l))
            for l in range(self.n_layers)
        )

    def focal_base(self, context: str) -> str:
        """The (folded) focal base of ``context``."""
        return context[self.focal_index(len(context))]

    # -- enumeration ------------------------------------------------------

    def children(self, parent: str) -> list[str]:
        """The children of ``parent``, in array order.

        Four under alternating growth (one base on one side), sixteen
        under centred growth (one base on each side, left index major).
        """
        if self.centred:
            return [left + parent + right for left in BASES for right in BASES]
        if self._appends_right(len(parent)):
            return [parent + nuc for nuc in BASES]
        return [nuc + parent for nuc in BASES]

    @lru_cache(maxsize=None)
    def contexts(self, layer: int) -> tuple[str, ...]:
        """All contexts in ``layer``, ordered by array index.

        Index ``i`` in layer ``l`` has children at
        ``[branching*i : branching*i + branching]`` of layer ``l + 1``.
        """
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer must be in [0, {self.n_layers}), got {layer}")
        current: tuple[str, ...] = FOLDED_BASES
        for _ in range(layer):
            current = tuple(
                child for parent in current for child in self.children(parent)
            )
        return current

    def leaf_contexts(self, context: str) -> list[str]:
        """The deepest-layer contexts descending from ``context``.

        Flanking bases are distributed to match how the tree would have
        grown, so the returned k-mers keep the focal base at
        ``focal_index(max_mer)``.
        """
        n_add = self.max_mer - len(context)
        if n_add < 0:
            raise ValueError(
                f"context {context!r} is longer than max_mer={self.max_mer}"
            )
        if n_add == 0:
            return [context]

        if self.centred:
            half = n_add // 2
            return [
                "".join(combo)[:half] + context + "".join(combo)[half:]
                for combo in product(BASES, repeat=n_add)
            ]

        max_is_odd = self.max_mer % 2
        extra = int(
            n_add % 2
            and (
                (max_is_odd and self.alternation == "right")
                or (not max_is_odd and self.alternation == "left")
            )
        )
        left = n_add // 2 + extra
        return [
            "".join(combo)[:left] + context + "".join(combo)[left:]
            for combo in product(BASES, repeat=n_add)
        ]

    # -- folding ----------------------------------------------------------

    def fold(self, context: str) -> str:
        """Reverse-complement ``context`` if its focal base is ``G``/``T``.

        Folding is an involution on the focal position because
        ``focal_index`` is symmetric under the reversal for the patterns
        this class supports; :func:`is_folded` is the check to use when
        that matters.
        """
        if self.focal_base(context) in FOLDED_BASES:
            return context
        return reverse_complement(context)

    def is_folded(self, context: str) -> bool:
        return self.focal_base(context) in FOLDED_BASES

    def fold_substitution(self, context: str, alt: str) -> tuple[str, str]:
        """Fold a (context, alt allele) pair onto the ``A``/``C`` strand."""
        if self.is_folded(context):
            return context, alt
        return reverse_complement(context), reverse_complement(alt)


def build_index_map(tree: ContextTree, layer: int) -> dict[str, int]:
    """Map context string to array index for ``layer``."""
    return {ctx: i for i, ctx in enumerate(tree.contexts(layer))}

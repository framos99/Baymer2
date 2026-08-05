"""Phylogeny handling: parse a tree, pick a focal species' comparators.

The 577-way Cactus alignment carries its own tree, and ``halStats --tree``
prints it as Newick with internal nodes named after the reconstructed
ancestral genomes. Everything needed to choose which genomes to compare
against a focal species is derivable from it, which is what this module
does -- replacing the hand-curated ``Selected_trio_sets.tsv`` that
currently lists 577 trios one per line.

The parser is deliberately self-contained. Newick is small, the input
comes from a tool whose output format is fixed, and a hard dependency on
dendropy for ~100 lines of parsing would be a poor trade for a package
that otherwise needs only numpy and numba.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "PhyloNode",
    "Phylogeny",
    "TrioSelection",
    "binomial",
    "parse_newick",
]

#: Assembly nicknames that carry no genus/species, mapped to binomials.
#: Large alignments label a few reference assemblies this way.
SPECIES_ALIASES: dict[str, str] = {
    "hg38": "Homo_sapiens",
    "hg19": "Homo_sapiens",
    "grch38": "Homo_sapiens",
    "chm13": "Homo_sapiens",
    "mm39": "Mus_musculus",
    "mm10": "Mus_musculus",
    "grcm39": "Mus_musculus",
    "rn7": "Rattus_norvegicus",
    "galgal6": "Gallus_gallus",
    "danrer11": "Danio_rerio",
}


def binomial(label: str) -> str:
    """Reduce a species label to genus + species.

    Subspecies must collapse onto their species: the VGP 577-way carries
    ``Canis_lupus_orion``, ``Canis_lupus_baileyi`` and
    ``Canis_lupus_familiaris``, which are one species and must not be
    used as each other's sister. Genuinely distinct congeners --
    ``Larus_argentatus`` versus ``Larus_michahellis`` -- must survive,
    which is why this trims to two tokens rather than one.

    Nicknames without a binomial (``hg38``, ``mm39``) are resolved
    through :data:`SPECIES_ALIASES`; anything unrecognised is returned
    as-is and simply will not match another label.
    """
    if not label:
        return label
    key = label.strip().lower().replace(" ", "_")
    if key in SPECIES_ALIASES:
        return SPECIES_ALIASES[key]
    parts = label.strip().replace(" ", "_").split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else label

_TOKEN = re.compile(r"[(),;]|[^(),;:]+(?::[^(),;]+)?")


@dataclass
class PhyloNode:
    """One node of a rooted tree."""

    name: str = ""
    branch_length: float = 0.0
    children: list["PhyloNode"] = field(default_factory=list, repr=False)
    parent: "PhyloNode | None" = field(default=None, repr=False, compare=False)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def walk(self):
        """Pre-order traversal from this node."""
        yield self
        for child in self.children:
            yield from child.walk()

    def leaves(self) -> list["PhyloNode"]:
        return [n for n in self.walk() if n.is_leaf]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "leaf" if self.is_leaf else f"internal({len(self.children)})"
        return f"PhyloNode({self.name!r}, {kind}, bl={self.branch_length:g})"


def _parse_label(token: str) -> tuple[str, float]:
    """Split ``name:length`` into its parts. Quotes are stripped."""
    if ":" in token:
        name, _, length = token.rpartition(":")
        try:
            value = float(length)
        except ValueError:
            # A colon inside a name with no branch length.
            return token.strip().strip("'\""), 0.0
        return name.strip().strip("'\""), value
    return token.strip().strip("'\""), 0.0


def parse_newick(text: str) -> PhyloNode:
    """Parse a Newick string into a rooted tree.

    Handles the forms ``halStats --tree`` emits: named internal nodes,
    branch lengths, optional quoting, and a trailing semicolon.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty Newick string")
    if text.endswith(";"):
        text = text[:-1]

    root = PhyloNode()
    current = root
    stack: list[PhyloNode] = []
    expect_child = False

    for token in _TOKEN.findall(text):
        if token == "(":
            child = PhyloNode(parent=current)
            current.children.append(child)
            stack.append(current)
            current = child
            expect_child = True
        elif token == ",":
            if not stack:
                raise ValueError("malformed Newick: ',' outside any clade")
            parent = stack[-1]
            sibling = PhyloNode(parent=parent)
            parent.children.append(sibling)
            current = sibling
            expect_child = True
        elif token == ")":
            if not stack:
                raise ValueError("malformed Newick: unbalanced ')'")
            current = stack.pop()
            expect_child = False
        elif token == ";":
            break
        else:
            name, length = _parse_label(token)
            current.name = name
            current.branch_length = length
            expect_child = False

    if stack:
        raise ValueError("malformed Newick: unbalanced '('")
    # A single-taxon tree parses into the root itself.
    if len(root.children) == 1 and not root.name:
        only = root.children[0]
        only.parent = None
        return only
    return root


@dataclass
class TrioSelection:
    """Comparator genomes chosen for one focal species."""

    focal: str
    sister: str
    outgroups: list[str]
    #: Reconstructed ancestor of the focal branch, when the tree names it.
    ancestor: str | None = None
    #: Focal branch length in the tree's own units.
    focal_branch_length: float = 0.0

    #: Patristic distance from the focal to each comparator, when known.
    distances: dict[str, float] = field(default_factory=dict)

    @property
    def targets(self) -> list[str]:
        """Genomes to request from ``hal2maf`` alongside the focal."""
        return [self.sister, *self.outgroups]

    @property
    def sister_more_diverged_than_outgroup(self) -> bool:
        """QC flag: is the topological sister further away than an outgroup?

        Sister and outgroup are topological roles, so this is legitimate
        and happens whenever the sister lineage has a long branch. It
        does, though, weaken parsimony polarisation -- the outgroup then
        carries more information about the ancestral state than the
        sister does -- so it is worth flagging across a 577-species
        sweep rather than discovering it per-species later.
        """
        if not self.distances or self.sister not in self.distances:
            return False
        sister_d = self.distances[self.sister]
        return any(
            self.distances.get(o, float("inf")) < sister_d for o in self.outgroups
        )

    def as_row(self) -> dict:
        return {
            "focal": self.focal,
            "sister": self.sister,
            "outgroups": ",".join(self.outgroups),
            "ancestor": self.ancestor or "",
            "focal_branch_length": self.focal_branch_length,
            "sister_distance": self.distances.get(self.sister, float("nan")),
            "sister_more_diverged_than_outgroup": int(
                self.sister_more_diverged_than_outgroup
            ),
        }


class Phylogeny:
    """A rooted tree with lookups for comparator selection."""

    def __init__(self, root: PhyloNode) -> None:
        self.root = root
        self._by_name: dict[str, PhyloNode] = {}
        for node in root.walk():
            if node.name:
                if node.name in self._by_name:
                    raise ValueError(f"duplicate node name in tree: {node.name!r}")
                self._by_name[node.name] = node

    @classmethod
    def from_newick(cls, text: str) -> "Phylogeny":
        return cls(parse_newick(text))

    @classmethod
    def from_file(cls, path) -> "Phylogeny":
        with open(path) as handle:
            return cls.from_newick(handle.read())

    # -- lookups ----------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def node(self, name: str) -> PhyloNode:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not in this tree; it has {len(self._by_name)} "
                f"named nodes"
            ) from None

    @property
    def leaf_names(self) -> list[str]:
        return [n.name for n in self.root.leaves()]

    @property
    def internal_names(self) -> list[str]:
        return [n.name for n in self.root.walk() if not n.is_leaf and n.name]

    def path_to_root(self, name: str) -> list[PhyloNode]:
        """Nodes from ``name`` up to and including the root."""
        node = self.node(name)
        path = [node]
        while node.parent is not None:
            node = node.parent
            path.append(node)
        return path

    def distance(self, a: str, b: str) -> float:
        """Patristic distance in the tree's branch-length units."""
        path_a = self.path_to_root(a)
        depth_a = {id(n): 0.0 for n in path_a}
        total = 0.0
        for node in path_a[:-1]:
            total += node.branch_length
            depth_a[id(node.parent)] = total

        total = 0.0
        node = self.node(b)
        while node is not None:
            if id(node) in depth_a:
                return depth_a[id(node)] + total
            total += node.branch_length
            node = node.parent
        raise ValueError(f"{a!r} and {b!r} are not connected")

    def closest_leaf(self, clade: PhyloNode, to: str) -> PhyloNode:
        """Least-diverged leaf of ``clade``, measured from ``to``.

        Comparator quality falls off with divergence -- deeper
        comparisons align worse and saturate sooner -- so a clade is
        always represented by its nearest member.
        """
        leaves = clade.leaves()
        if not leaves:
            raise ValueError("clade has no leaves")
        return min(leaves, key=lambda leaf: self.distance(to, leaf.name))

    # -- comparator selection ---------------------------------------------

    def select(
        self,
        focal: str,
        n_outgroups: int = 1,
        *,
        min_divergence: float = 0.0,
        species_of: "dict[str, str] | None" = None,
    ) -> TrioSelection:
        """Choose a sister and ``n_outgroups`` successive outgroups.

        Walks rootward from ``focal``: the sibling clade at each ancestor
        supplies the next comparator, each one more distant than the
        last. This is what ``Selected_trio_sets.tsv`` encodes by hand.

        Two guards keep degenerate comparators out.

        ``min_divergence`` skips comparators closer than the given
        patristic distance. A sister that near is useless for
        polarisation: essentially every site is invariant between focal
        and sister, so the trio degenerates and the focal branch carries
        almost no information.

        ``species_of`` maps assembly to species and skips comparators
        from the *same species* as the focal. This is the criterion the
        hand-curated ``Selected_trio_sets.tsv`` actually applies, and
        distance cannot substitute for it: in the VGP 577-way, GRCh38
        and T2T-CHM13 sit 0.00054 apart and must not be paired, while
        two genuinely distinct gull species sit 0.00006 apart and
        should be. Without species labels, selection reproduces ~98.6%
        of that table; the residual is exactly the same-species cases.
        """
        if n_outgroups < 1:
            raise ValueError(f"n_outgroups must be >= 1, got {n_outgroups}")
        if min_divergence < 0:
            raise ValueError(
                f"min_divergence must be non-negative, got {min_divergence}"
            )

        node = self.node(focal)
        if not node.is_leaf:
            raise ValueError(
                f"{focal!r} is an internal node; the focal genome must be a leaf"
            )

        focal_species = (species_of or {}).get(focal)

        comparators: list[str] = []
        walker = node
        while walker.parent is not None and len(comparators) < n_outgroups + 1:
            parent = walker.parent
            for sibling in parent.children:
                if sibling is walker:
                    continue
                candidate = self._nearest_usable_leaf(
                    sibling, focal, min_divergence, species_of, focal_species
                )
                if candidate is None:
                    continue
                comparators.append(candidate)
                if len(comparators) == n_outgroups + 1:
                    break
            walker = parent

        if len(comparators) < n_outgroups + 1:
            raise ValueError(
                f"{focal!r} has only {len(comparators)} available comparators "
                f"in this tree but {n_outgroups + 1} were requested (1 sister "
                f"+ {n_outgroups} outgroups); the tree is too shallow above it"
            )

        ancestor = node.parent.name or None
        return TrioSelection(
            focal=focal,
            sister=comparators[0],
            outgroups=comparators[1:],
            ancestor=ancestor,
            focal_branch_length=node.branch_length,
            distances={c: self.distance(focal, c) for c in comparators},
        )

    def _nearest_usable_leaf(
        self,
        clade: PhyloNode,
        focal: str,
        min_divergence: float,
        species_of: "dict[str, str] | None" = None,
        focal_species: str | None = None,
    ) -> str | None:
        """Closest admissible leaf of ``clade``.

        Returns ``None`` when every leaf is excluded, so the caller moves
        outward rather than accepting a degenerate comparator.
        """
        usable: list[tuple[float, str]] = []
        for leaf in clade.leaves():
            if focal_species is not None and species_of is not None:
                if species_of.get(leaf.name) == focal_species:
                    continue
            distance = self.distance(focal, leaf.name)
            if distance < min_divergence:
                continue
            usable.append((distance, leaf.name))
        if not usable:
            return None
        return min(usable)[1]

    def select_all(
        self,
        n_outgroups: int = 1,
        *,
        min_divergence: float = 0.0,
        species_of: "dict[str, str] | None" = None,
    ) -> list[TrioSelection]:
        """Comparators for every leaf that has enough of them."""
        out = []
        for name in self.leaf_names:
            try:
                out.append(
                    self.select(
                        name,
                        n_outgroups,
                        min_divergence=min_divergence,
                        species_of=species_of,
                    )
                )
            except ValueError:
                continue
        return out

    def ladder(self, selection: TrioSelection) -> dict:
        """Branch lengths of the nested comparator ladder.

        ``select`` produces the topology ``((((focal, sister), o1), o2), ...)``,
        and maximum-likelihood reconstruction needs every branch of it,
        not just the tip-to-tip distances. Internal node ``n1`` is the
        focal's parent; ``n_{k+1}`` is the parent of ``n_k``, where
        outgroup ``k`` attaches.

        A comparator may represent a clade rather than being a direct
        child, so its branch is measured as the residual of its patristic
        distance to the focal after the shared path is removed. That is
        the correct length for a two-state transition along that
        lineage, and it is what keeps the reconstruction honest when the
        chosen representative sits deep inside its clade.
        """
        focal = self.node(selection.focal)

        # Depth of each internal node above the focal.
        internals: list[PhyloNode] = []
        depths: list[float] = []
        depth = focal.branch_length
        walker = focal.parent
        while walker is not None and len(internals) < len(selection.outgroups) + 1:
            internals.append(walker)
            depths.append(depth)
            depth += walker.branch_length
            walker = walker.parent

        if len(internals) < len(selection.outgroups) + 1:
            raise ValueError(
                f"tree is too shallow above {selection.focal!r} to build a "
                f"ladder for {len(selection.outgroups)} outgroups"
            )

        sister_length = self.distance(selection.focal, selection.sister) - depths[0]
        outgroup_lengths = [
            self.distance(selection.focal, name) - depths[i + 1]
            for i, name in enumerate(selection.outgroups)
        ]
        internal_lengths = [internals[i].branch_length for i in range(len(internals) - 1)]

        return {
            "focal_length": focal.branch_length,
            "sister_length": sister_length,
            "internal_lengths": internal_lengths,
            "outgroup_lengths": outgroup_lengths,
            "ancestor": internals[0].name or None,
        }

    def subtree_branch_lengths(self, selection: TrioSelection) -> dict[str, float]:
        """Branch lengths from the focal to each comparator.

        Used to scale the CTMC branch length and to weight ancestral
        reconstruction.
        """
        return {
            name: self.distance(selection.focal, name)
            for name in selection.targets
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Phylogeny({len(self.leaf_names)} leaves, "
            f"{len(self.internal_names)} named internal nodes)"
        )

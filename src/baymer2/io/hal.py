"""Read a Cactus HAL alignment: tree, genomes, and MAF blocks.

The HAL toolkit is a set of C++ binaries, so everything here shells out.
Two invocation routes are supported, because clusters tend to have one or
the other: native binaries (``halStats``, ``hal2maf``) and the Cactus
Apptainer image. Point at them with ``$BAYMER2_CACTUS_BIN`` and
``$BAYMER2_CACTUS_SIF``; a ``halStats`` already on ``PATH`` is found
without either.

The MAF handling is ported from the trio alignment and polarization
script used for an earlier 27-species run -- specifically its
``extract_trio_maf``, ``_apptainer_prefix`` and ``parse_maf`` -- so those
species remain reproducible here. The MAF parser is a streaming block
iterator rather than a whole-file read: a single focal species against
the 577-way alignment produced 5.19M blocks and peaked at 162 GB when
materialised.
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from baymer2.ancestral.tree import Phylogeny

__all__ = [
    "BlockTally",
    "HalRuntime",
    "MafRecord",
    "order_by_specificity",
    "sample_src_names",
    "hal_tree",
    "hal_genomes",
    "extract_maf",
    "parse_maf",
    "iter_maf_blocks",
]

#: Environment variables naming the two invocation routes.
CACTUS_BIN_ENV = "BAYMER2_CACTUS_BIN"
CACTUS_SIF_ENV = "BAYMER2_CACTUS_SIF"


@dataclass
class HalRuntime:
    """How to invoke the HAL binaries.

    Prefers native binaries when present; falls back to Apptainer. A large
    alignment usually lives on node-local ``/scratch``, so bind mounts must
    cover both the HAL and the output directory or the container sees
    neither.
    """

    bin_dir: Path | None = None
    sif: Path | None = None
    bind_paths: tuple[str, ...] = ("/project", "/scratch", "/home")

    @classmethod
    def detect(cls) -> "HalRuntime":
        """Pick whichever route is available on this machine.

        ``$BAYMER2_CACTUS_BIN`` and ``$BAYMER2_CACTUS_SIF`` win when set;
        otherwise a ``halStats`` on ``PATH`` supplies the bin directory.
        """
        bin_dir = None
        env_bin = os.environ.get(CACTUS_BIN_ENV)
        if env_bin and (Path(env_bin) / "halStats").exists():
            bin_dir = Path(env_bin)

        sif = None
        env_sif = os.environ.get(CACTUS_SIF_ENV)
        if env_sif and Path(env_sif).exists():
            sif = Path(env_sif)

        if bin_dir is None and sif is None:
            on_path = shutil.which("halStats")
            if on_path:
                bin_dir = Path(on_path).parent
        return cls(bin_dir=bin_dir, sif=sif)

    @property
    def available(self) -> bool:
        return self.bin_dir is not None or self.sif is not None

    def command(self, tool: str, args: Sequence[str]) -> list[str]:
        """Full argv for ``tool`` with ``args``."""
        if self.bin_dir is not None:
            executable = self.bin_dir / tool
            if executable.exists():
                return [str(executable), *args]
        if self.sif is not None:
            binds: list[str] = []
            for path in self.bind_paths:
                if Path(path).exists():
                    binds += ["--bind", path]
            return ["apptainer", "exec", *binds, str(self.sif), tool, *args]
        raise RuntimeError(
            f"cannot run {tool!r}: no cactus bin directory and no .sif image. "
            f"Set ${CACTUS_BIN_ENV} to a directory containing halStats, or "
            f"${CACTUS_SIF_ENV} to a Cactus Apptainer image, or put halStats "
            f"on PATH."
        )

    def run(self, tool: str, args: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
        argv = self.command(tool, args)
        return subprocess.run(argv, check=True, text=True, capture_output=True, **kwargs)


def hal_tree(hal_path: str | Path, runtime: HalRuntime | None = None) -> Phylogeny:
    """The alignment's own phylogeny, via ``halStats --tree``."""
    runtime = runtime or HalRuntime.detect()
    result = runtime.run("halStats", ["--tree", str(hal_path)])
    newick = result.stdout.strip()
    if not newick:
        raise RuntimeError(f"halStats --tree returned nothing for {hal_path}")
    return Phylogeny.from_newick(newick)


def hal_genomes(hal_path: str | Path, runtime: HalRuntime | None = None) -> list[str]:
    """Every genome in the alignment, ancestral reconstructions included."""
    runtime = runtime or HalRuntime.detect()
    result = runtime.run("halStats", ["--genomes", str(hal_path)])
    return result.stdout.split()


def ancestral_genomes(hal_path: str | Path, runtime: HalRuntime | None = None) -> list[str]:
    """Genomes that are internal nodes, i.e. Cactus's reconstructed ancestors.

    Their presence is what makes the ``hal-ancestor`` method possible:
    the focal branch's ancestor can be read directly instead of being
    re-inferred from a trio.
    """
    tree = hal_tree(hal_path, runtime)
    present = set(hal_genomes(hal_path, runtime))
    return [name for name in tree.internal_names if name in present]


def extract_maf(
    hal_path: str | Path,
    focal: str,
    targets: Iterable[str],
    output: str | Path,
    *,
    runtime: HalRuntime | None = None,
    no_dupes: bool = True,
    only_orthologs: bool = True,
    no_ancestors: bool = True,
    ref_sequence: str | None = None,
    extra_args: Sequence[str] = (),
) -> Path:
    """Run ``hal2maf`` for one focal genome against its comparators.

    ``--refGenome`` puts the focal on the ``+`` strand and makes its
    coordinates the reference frame, which every downstream step
    assumes. ``--noDupes``/``--onlyOrthologs`` drop paralogous
    alignments, without which a block can carry a species twice and
    ancestral inference becomes ill-posed.
    """
    runtime = runtime or HalRuntime.detect()
    targets = list(targets)
    if not targets:
        raise ValueError("at least one target genome is required")

    args = [
        str(hal_path),
        str(output),
        "--refGenome",
        focal,
        "--targetGenomes",
        ",".join(targets),
    ]
    if no_dupes:
        args.append("--noDupes")
    if only_orthologs:
        args.append("--onlyOrthologs")
    if no_ancestors:
        args.append("--noAncestors")
    if ref_sequence:
        args += ["--refSequence", ref_sequence]
    args += list(extra_args)

    runtime.run("hal2maf", args)
    return Path(output)


@dataclass
class MafRecord:
    """One ``s`` line of a MAF block.

    The ``src`` field is nominally ``genome.sequence``, but the split
    point cannot be recovered from the string alone: genome names may
    themselves contain dots. Every VGP assembly accession does --
    ``GCA_048771995.1`` -- so an s-line reads
    ``GCA_048771995.1.CM088821.1`` and splitting on the first dot yields
    ``GCA_048771995``, which matches nothing. Resolution therefore
    requires the set of genome names actually being looked for; see
    :meth:`genome_in`.
    """

    src: str
    start: int
    length: int
    strand: str
    src_size: int
    seq: str

    def genome_in(self, known: Sequence[str]) -> str | None:
        """Which of ``known`` this line belongs to, or ``None``.

        ``known`` must be ordered longest-first so that a genome name
        which is a strict prefix of another cannot capture it.
        """
        for name in known:
            if self.src == name or self.src.startswith(name + "."):
                return name
        return None

    def sequence_in(self, genome: str) -> str:
        """Sequence name, given the genome this line resolved to."""
        if self.src == genome:
            return ""
        return self.src[len(genome) + 1 :]


def order_by_specificity(names: Iterable[str]) -> list[str]:
    """Longest first, so prefix collisions resolve to the longer name."""
    return sorted(names, key=len, reverse=True)


@dataclass
class BlockTally:
    """Why blocks were kept or dropped.

    Reported by ``hal-counts``. Without it, a run whose genome names do
    not resolve is indistinguishable from one where the alignment simply
    has no coverage -- both just yield zero blocks.
    """

    total: int = 0
    kept: int = 0
    unresolved_src: int = 0
    no_focal: int = 0
    missing_required: int = 0
    too_short: int = 0
    duplicate_genome: int = 0

    def summary(self) -> dict[str, int]:
        return {
            "blocks read": self.total,
            "kept": self.kept,
            "dropped: unresolved src name": self.unresolved_src,
            "dropped: focal absent": self.no_focal,
            "dropped: comparator absent": self.missing_required,
            "dropped: focal below min-block": self.too_short,
            "dropped: duplicated genome": self.duplicate_genome,
        }


def sample_src_names(path: str | Path, limit: int = 5) -> list[str]:
    """First few distinct ``src`` fields, for diagnostics.

    Shown next to the requested genome names when nothing resolves,
    which makes a name-format mismatch obvious immediately rather than
    after a wasted run.
    """
    seen: list[str] = []
    with _open_text(path) as handle:
        for line in handle:
            if not line.startswith("s"):
                continue
            parts = line.split()
            if len(parts) == 7 and parts[1] not in seen:
                seen.append(parts[1])
                if len(seen) >= limit:
                    break
    return seen


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def parse_maf(path: str | Path) -> Iterator[list[MafRecord]]:
    """Stream MAF blocks, one list of records at a time.

    Deliberately a generator: whole-file parsing of a 577-way focal
    extraction is millions of blocks and hundreds of GB.
    """
    block: list[MafRecord] = []
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("a"):
                if block:
                    yield block
                block = []
            elif line.startswith("s"):
                parts = line.split()
                if len(parts) != 7:
                    raise ValueError(f"malformed MAF 's' line: {line.rstrip()}")
                _, src, start, length, strand, src_size, seq = parts
                block.append(
                    MafRecord(
                        src=src,
                        start=int(start),
                        length=int(length),
                        strand=strand,
                        src_size=int(src_size),
                        seq=seq,
                    )
                )
            elif not line.strip():
                if block:
                    yield block
                    block = []
    if block:
        yield block


def iter_maf_blocks(
    path: str | Path,
    focal: str,
    required: Sequence[str] = (),
    *,
    min_block: int = 50,
    tally: BlockTally | None = None,
) -> Iterator[dict[str, MafRecord]]:
    """Blocks keyed by genome, filtered to usable ones.

    A block is skipped when the focal is absent, any required genome is
    absent, the focal segment is shorter than ``min_block``, or a genome
    appears more than once. That last case means a paralogous alignment
    slipped past ``--noDupes``; treating it as orthologous would
    silently corrupt the ancestral call, so the block is dropped.

    Genome names are resolved against the requested set rather than by
    splitting ``src`` on a dot, because assembly accessions contain
    dots of their own. ``tally``, when supplied, records why blocks were
    dropped -- in particular how many carried an ``src`` matching none
    of the requested names, which is the signature of a name-format
    mismatch rather than absent coverage.
    """
    wanted = order_by_specificity({focal, *required})
    for block in parse_maf(path):
        if tally is not None:
            tally.total += 1

        by_genome: dict[str, MafRecord] = {}
        duplicated = False
        resolved_any = False
        for record in block:
            genome = record.genome_in(wanted)
            if genome is None:
                continue
            resolved_any = True
            if genome in by_genome:
                duplicated = True
                break
            by_genome[genome] = record

        if duplicated:
            if tally is not None:
                tally.duplicate_genome += 1
            continue
        if not resolved_any:
            if tally is not None:
                tally.unresolved_src += 1
            continue
        if focal not in by_genome:
            if tally is not None:
                tally.no_focal += 1
            continue
        if any(name not in by_genome for name in required):
            if tally is not None:
                tally.missing_required += 1
            continue
        if by_genome[focal].length < min_block:
            if tally is not None:
                tally.too_short += 1
            continue
        if by_genome[focal].strand != "+":
            # hal2maf --refGenome guarantees this; a violation means the
            # MAF was produced some other way and coordinates cannot be
            # trusted.
            raise ValueError(
                f"focal genome {focal!r} is on the {by_genome[focal].strand} "
                f"strand; MAF must be produced with --refGenome {focal}"
            )
        if tally is not None:
            tally.kept += 1
        yield by_genome

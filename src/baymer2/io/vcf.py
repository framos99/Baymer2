"""Count contexts and mutations from FASTA + VCF, the polymorphism path.

Ports ``context_counter.py`` and ``mutation_counter.py``, fixing three
things that were wrong or awkward in them.

The parity bug
--------------
Legacy tags every base and every variant by ``position % 2`` so the data
can be split into two disjoint, interleaved, size-matched halves for
cross-validation. The two halves are mislabelled relative to each other:

* ``context_counter.py`` computes ``even_odd_index = fasta_pos % 2`` and
  writes index 0 to a column it names ``.odd_bp`` -- so ``.odd_bp``
  actually holds positions where ``pos % 2 == 0``.
* ``mutation_counter.py`` computes the same index and stores it as
  ``even_odd_bool``.
* ``generate_count_json.py`` then selects ``even_odd_bool == 0`` for
  ``"EVEN"`` but reads the ``.even_bp`` *column* for the denominator --
  which is the other parity class.

So every ``ODD`` and ``EVEN`` run draws its numerator from one parity
class and its denominator from the other. ``ALL`` sums both and is
unaffected. Both the tutorial and the production species configs run
ODD/EVEN, so this is live.

Here parity is a single concept -- ``position % 2``, 0 is even -- applied
identically to numerators and denominators, and
:func:`build_count_table` asserts the two agree before returning.

Two smaller fixes
-----------------
``context_counter`` accepted gzipped FASTA while ``mutation_counter``
did not; both do here. And legacy discards every multiallelic site
outright (``len(REF) + len(ALT) > 2``), which throws away the clearest
signature of recurrent mutation; ``keep_multiallelic`` retains them for
the ``recurrent`` likelihood.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from baymer2.contexts import BASE_INDEX, ContextTree
from baymer2.counts import CountTable

__all__ = [
    "DATASETS",
    "VariantFilter",
    "count_contexts",
    "count_mutations",
    "build_count_table",
    "iter_fasta",
]

#: Parity selections. ``EVEN`` keeps ``pos % 2 == 0``.
DATASETS = ("ALL", "EVEN", "ODD")

#: bedtools getfasta headers look like ``name:start-end``.
_REGION = re.compile(r"^(?P<name>.+):(?P<start>\d+)-(?P<end>\d+)$")


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix in (".gz", ".bgz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


@dataclass
class FastaRegion:
    """One FASTA record with its genomic coordinates."""

    name: str
    start: int  # 0-based, inclusive
    sequence: str

    @property
    def end(self) -> int:
        return self.start + len(self.sequence)


def iter_fasta(path: str | Path) -> Iterator[FastaRegion]:
    """Stream FASTA records, parsing bedtools-style coordinate headers.

    A plain header with no ``name:start-end`` is treated as starting at
    zero, so whole-chromosome FASTAs work too -- legacy raised
    ``IndexError`` on those.
    """
    name: str | None = None
    start = 0
    chunks: list[str] = []

    with _open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    yield FastaRegion(name, start, "".join(chunks))
                header = line[1:].strip().split()[0]
                match = _REGION.match(header)
                if match:
                    name = match.group("name")
                    start = int(match.group("start"))
                else:
                    name = header
                    start = 0
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield FastaRegion(name, start, "".join(chunks))


def _normalise_chrom(name: str) -> str:
    """Strip a leading ``chr`` so FASTA and VCF naming can differ.

    Legacy had two counters that disagreed about this -- one stripped,
    one did not -- which is why ``testingname.mutation_counter.py``
    exists alongside ``mutation_counter.py``.
    """
    return name[3:] if name.startswith("chr") else name


def count_contexts(
    fasta_path: str | Path,
    tree: ContextTree,
    *,
    high_confidence: bool = True,
) -> np.ndarray:
    """Denominators: occurrences of each context, split by position parity.

    Returns ``(n_contexts, 2)`` indexed by ``position % 2``.

    With ``high_confidence`` only uppercase ACGT count; lowercase marks
    soft-masked or low-confidence ancestral calls, and ``.``/``N`` mark
    no call. A window containing any of those is skipped entirely rather
    than partially counted.
    """
    k = tree.max_mer
    layer = tree.n_layers - 1
    contexts = tree.contexts(layer)
    index = {c: i for i, c in enumerate(contexts)}
    focal_offset = tree.focal_index(k)

    counts = np.zeros((len(contexts), 2), dtype=np.float64)
    valid = set("ACGT") if high_confidence else set("ACGTacgt")

    for region in iter_fasta(fasta_path):
        seq = region.sequence if high_confidence else region.sequence.upper()
        for i in range(len(seq) - k + 1):
            window = seq[i : i + k]
            if not set(window) <= valid:
                continue
            kmer = window.upper()
            folded = tree.fold(kmer)
            position = region.start + i + focal_offset
            counts[index[folded], position % 2] += 1

    return counts


@dataclass
class VariantFilter:
    """Selection criteria applied to VCF records."""

    min_ac: int = 1
    max_ac: int | None = None
    min_af: float = 0.0
    max_af: float = 0.85
    min_quality: float | None = None
    ac_key: str = "AC"
    an_key: str = "AN"
    quality_key: str | None = None
    #: Legacy drops these outright; they are the clearest signature of
    #: recurrent mutation, so the recurrent likelihood wants them.
    keep_multiallelic: bool = False

    def accepts(self, ac: float, an: float, quality: float | None) -> bool:
        if ac < self.min_ac:
            return False
        if self.max_ac is not None and ac > self.max_ac:
            return False
        if an > 0:
            af = ac / an
            if not (self.min_af <= af < self.max_af):
                return False
        if self.min_quality is not None:
            if quality is None or quality < self.min_quality:
                return False
        return True


def _parse_info(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for field_ in info.split(";"):
        if "=" in field_:
            key, _, value = field_.partition("=")
            out[key] = value
        else:
            out[field_] = ""
    return out


@dataclass
class MutationCounts:
    """Numerators, and the QC tallies legacy printed to stdout."""

    counts: np.ndarray  # (n_contexts, 4, 2)
    n_records: int = 0
    n_kept: int = 0
    n_multiallelic: int = 0
    n_filtered: int = 0
    n_outside_regions: int = 0
    n_ref_mismatch: int = 0
    n_invalid_context: int = 0

    def summary(self) -> dict[str, int]:
        return {
            "records": self.n_records,
            "kept": self.n_kept,
            "multiallelic": self.n_multiallelic,
            "failed_filters": self.n_filtered,
            "outside_regions": self.n_outside_regions,
            "reference_mismatch": self.n_ref_mismatch,
            "invalid_context": self.n_invalid_context,
        }


def count_mutations(
    vcf_path: str | Path,
    fasta_path: str | Path,
    tree: ContextTree,
    *,
    filters: VariantFilter | None = None,
    high_confidence: bool = True,
) -> MutationCounts:
    """Numerators: observed alternate alleles per context, by parity.

    Returns counts shaped ``(n_contexts, 4, 2)`` where the middle axis is
    the alternate base index *after folding*, and the last is
    ``position % 2``.
    """
    filters = filters or VariantFilter()
    k = tree.max_mer
    layer = tree.n_layers - 1
    contexts = tree.contexts(layer)
    index = {c: i for i, c in enumerate(contexts)}
    focal_offset = tree.focal_index(k)
    half_before = focal_offset
    valid = set("ACGT") if high_confidence else set("ACGTacgt")

    # Regions are held per chromosome so a VCF that is sorted differently
    # from the FASTA still joins correctly. Legacy assumed a single
    # coordinated pass and silently dropped variants when that failed.
    regions: dict[str, list[FastaRegion]] = {}
    for region in iter_fasta(fasta_path):
        regions.setdefault(_normalise_chrom(region.name), []).append(region)
    for entries in regions.values():
        entries.sort(key=lambda r: r.start)

    starts = {c: [r.start for r in entries] for c, entries in regions.items()}
    result = MutationCounts(counts=np.zeros((len(contexts), 4, 2), dtype=np.float64))

    import bisect

    with _open_text(vcf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            result.n_records += 1

            chrom = _normalise_chrom(parts[0])
            position = int(parts[1]) - 1  # VCF is 1-based
            ref, alt_field, info = parts[3], parts[4], parts[7]

            alts = alt_field.split(",")
            if len(ref) != 1 or any(len(a) != 1 for a in alts):
                continue  # indel
            if len(alts) > 1:
                result.n_multiallelic += 1
                if not filters.keep_multiallelic:
                    continue

            annotations = _parse_info(info)
            try:
                ac_values = [float(v) for v in annotations.get(filters.ac_key, "").split(",") if v]
                an = float(annotations.get(filters.an_key, "0") or 0)
            except ValueError:
                result.n_filtered += 1
                continue
            if not ac_values:
                result.n_filtered += 1
                continue
            quality = None
            if filters.quality_key:
                raw = annotations.get(filters.quality_key)
                quality = float(raw) if raw else None

            entries = regions.get(chrom)
            if not entries:
                result.n_outside_regions += 1
                continue
            slot = bisect.bisect_right(starts[chrom], position) - 1
            if slot < 0:
                result.n_outside_regions += 1
                continue
            region = entries[slot]
            offset = position - region.start
            window_start = offset - half_before
            if window_start < 0 or window_start + k > len(region.sequence):
                result.n_outside_regions += 1
                continue

            window = region.sequence[window_start : window_start + k]
            if not set(window) <= valid:
                result.n_invalid_context += 1
                continue
            kmer = window.upper()
            if kmer[focal_offset] != ref.upper():
                result.n_ref_mismatch += 1
                continue

            for alt, ac in zip(alts, ac_values):
                if not filters.accepts(ac, an, quality):
                    result.n_filtered += 1
                    continue
                folded_kmer, folded_alt = tree.fold_substitution(kmer, alt.upper())
                if folded_alt not in BASE_INDEX:
                    result.n_invalid_context += 1
                    continue
                result.counts[
                    index[folded_kmer], BASE_INDEX[folded_alt], position % 2
                ] += 1
                result.n_kept += 1

    return result


def build_count_table(
    context_counts: np.ndarray,
    mutation_counts: np.ndarray,
    tree: ContextTree,
    *,
    dataset: str = "ALL",
    name: str = "",
) -> CountTable:
    """Combine denominators and numerators into a fitted-ready table.

    ``dataset`` selects a parity class -- and selects it from *both*
    inputs, which is the fix. The reference slot is the denominator less
    the observed alternates, so a negative value means the two inputs
    disagree and is raised rather than silently clipped.
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")

    if dataset == "ALL":
        denominators = context_counts.sum(axis=1)
        numerators = mutation_counts.sum(axis=2)
    else:
        parity = 0 if dataset == "EVEN" else 1
        denominators = context_counts[:, parity]
        numerators = mutation_counts[:, :, parity]

    layer = tree.n_layers - 1
    contexts = tree.contexts(layer)
    focal_offset = tree.focal_index(tree.max_mer)

    leaf_counts = np.zeros((len(contexts), 4), dtype=np.float64)
    for i, ctx in enumerate(contexts):
        ref = ctx[focal_offset]
        alt_total = 0.0
        for base, base_index in BASE_INDEX.items():
            if base == ref:
                continue
            slot = _slot_for(ref, base)
            leaf_counts[i, slot] = numerators[i, base_index]
            alt_total += numerators[i, base_index]
        reference = denominators[i] - alt_total
        if reference < 0:
            raise ValueError(
                f"context {ctx!r} has {alt_total:g} observed alternates but "
                f"only {denominators[i]:g} opportunities in dataset "
                f"{dataset!r}. Numerator and denominator disagree, which is "
                f"what the legacy parity mismatch produced."
            )
        leaf_counts[i, 0] = reference

    return CountTable(tree=tree, leaf_counts=leaf_counts, name=name)


def _slot_for(ref: str, alt: str) -> int:
    from baymer2.contexts import alt_bases

    return 1 + alt_bases(ref).index(alt)

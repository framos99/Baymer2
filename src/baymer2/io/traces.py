"""Disk-backed posterior traces.

Why
---
Legacy Baymer holds every thinned draw of every quantity in RAM for the
whole layer. At ``max_mer=9`` the deepest layer has 131,072 nodes, so
each of ``phis``, ``rate_matrix``, ``indicator`` and
``set_probabilities`` is ``n_samples x 131072 x 3``; at 2000 samples
that is ~3.1 GB apiece, ~12.6 GB live, before counting the retained
parent layer. ``likelihood_matrix`` additionally stores *every*
un-thinned iteration. This is what forces the
``rusage[mem=25GB]`` in the production wrapper, and it scales as
``4**layer``.

How
---
Traces are allocated as ``.npy`` files opened through
``numpy.lib.format.open_memmap`` and written a sample at a time. The
result is an ordinary ``.npy`` on disk -- readable by ``np.load`` and by
the legacy plotting scripts -- while the process only ever holds the
pages it is touching. Page-cache residency is evictable, so the job's
committed memory stays flat instead of growing with ``4**layer``.

The one array that genuinely must stay resident is the parent layer's
ALR coordinates, because the cut linkage indexes into it randomly on
every iteration. That is ``n_samples x n_nodes/4 x 3``, a quarter of a
layer, and it is kept in float32.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

__all__ = [
    "PER_NODE_ARRAYS",
    "SCALAR_ARRAYS",
    "TraceStore",
    "load_layer_traces",
]

#: name -> dtype for arrays shaped ``(n_samples, n_nodes, 3)``.
PER_NODE_ARRAYS: dict[str, type] = {
    "eta": np.float32,
    "phi": np.float32,
    "rates": np.float32,
    "indicator": np.int8,
}

#: name -> dtype for arrays shaped ``(n_samples,)``.
SCALAR_ARRAYS: dict[str, type] = {
    "alpha": np.float64,
    "sigma": np.float64,
}

MANIFEST_NAME = "manifest.json"


def _array_path(directory: Path, name: str, layer: int, prefix: str) -> Path:
    stem = f"{prefix}." if prefix else ""
    return directory / f"{stem}{name}.layer_{layer}.npy"


@dataclass
class TraceStore:
    """Directory of memory-mapped posterior traces for one model fit.

    Parameters
    ----------
    directory:
        Created if absent. One ``.npy`` per quantity per layer.
    prefix:
        Optional filename prefix, e.g. a population or species label.
    """

    directory: Path
    prefix: str = ""
    _open: dict[int, dict[str, np.ndarray]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- allocation -------------------------------------------------------

    def allocate_layer(
        self, layer: int, n_samples: int, n_nodes: int
    ) -> dict[str, np.ndarray]:
        """Create the on-disk arrays for one layer and return them.

        The returned objects are ``np.memmap`` instances, so ordinary
        assignment (``arr[i] = ...``) writes through to disk and
        ordinary reduction (``arr.mean(axis=0)``) streams back.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        out: dict[str, np.ndarray] = {}
        for name, dtype in PER_NODE_ARRAYS.items():
            out[name] = open_memmap(
                _array_path(self.directory, name, layer, self.prefix),
                mode="w+",
                dtype=dtype,
                shape=(n_samples, n_nodes, 3),
            )
        for name, dtype in SCALAR_ARRAYS.items():
            out[name] = open_memmap(
                _array_path(self.directory, name, layer, self.prefix),
                mode="w+",
                dtype=dtype,
                shape=(n_samples,),
            )
        self._open.setdefault(layer, {}).update(out)
        return out

    def allocate_loglik(self, layer: int, n_iterations: int) -> np.ndarray:
        """Full un-thinned log-likelihood trace for one layer."""
        arr = open_memmap(
            _array_path(self.directory, "loglik", layer, self.prefix),
            mode="w+",
            dtype=np.float64,
            shape=(n_iterations,),
        )
        self._open.setdefault(layer, {})["loglik"] = arr
        return arr

    # -- finalisation -----------------------------------------------------

    def release_layer_pages(self, layer: int) -> None:
        """Write back and then evict this layer's resident trace pages.

        Flushing alone does not reduce resident memory: a written page
        stays in the process's RSS until the kernel has a reason to
        reclaim it, and on an unloaded node it never does. The result is
        that a purely mmap-backed trace measures almost as large as an
        in-RAM one -- 2.92 GB against 3.40 GB at ``max_mer=8`` -- even
        though the data is file-backed.

        ``MADV_DONTNEED`` on a shared file mapping drops the pages
        outright once they are clean, which is what actually bounds RSS.
        Re-reading a released region simply faults it back in from disk,
        so this is safe at any point; it is only a performance
        trade-off. Called periodically during sampling rather than once
        at the end, because otherwise a whole layer accumulates first --
        ~12.6 GB at ``max_mer=9``.
        """
        import mmap as _mmap

        for arr in self._open.get(layer, {}).values():
            flush = getattr(arr, "flush", None)
            if flush is not None:
                flush()
            backing = getattr(arr, "_mmap", None)
            if backing is None:
                continue
            advise = getattr(backing, "madvise", None)
            if advise is None:  # pragma: no cover - needs Python < 3.8
                continue
            try:
                advise(_mmap.MADV_DONTNEED)
            except (OSError, ValueError):  # pragma: no cover - platform dependent
                pass

    def flush_layer(self, layer: int) -> None:
        """Push dirty pages to disk and drop the writable maps.

        Must happen before anything reopens these paths, otherwise a
        reader can observe a partially written file.
        """
        self.release_layer_pages(layer)
        self._open.pop(layer, None)

    def truncate_layer(
        self, layer: int, n_written: int, n_allocated: int
    ) -> dict[str, np.ndarray]:
        """Shrink a layer's arrays to the number of samples actually written.

        Thinning arithmetic can leave the final slot unused; rewriting
        the header rather than leaving a zero row keeps downstream means
        honest.
        """
        result: dict[str, np.ndarray] = {}
        for name in list(PER_NODE_ARRAYS) + list(SCALAR_ARRAYS):
            path = _array_path(self.directory, name, layer, self.prefix)
            if n_written == n_allocated:
                result[name] = np.load(path, mmap_mode="r")
                continue
            data = np.load(path, mmap_mode="r")
            trimmed = np.array(data[:n_written])
            del data
            np.save(path, trimmed)
            result[name] = np.load(path, mmap_mode="r")
        return result

    def write_manifest(self, payload: dict) -> Path:
        path = self.directory / MANIFEST_NAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path

    def read_manifest(self) -> dict:
        path = self.directory / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(f"no manifest in {self.directory}")
        return json.loads(path.read_text())

    def layer_arrays(self, layer: int) -> dict[str, np.ndarray]:
        """Reopen a layer's traces read-only."""
        return load_layer_traces(self.directory, layer, prefix=self.prefix)

    def nbytes(self) -> int:
        return sum(p.stat().st_size for p in self.directory.glob("*.npy"))


def load_layer_traces(
    directory: str | Path, layer: int, *, prefix: str = ""
) -> dict[str, np.ndarray]:
    """Load one layer's traces as read-only memory maps."""
    directory = Path(directory)
    out: dict[str, np.ndarray] = {}
    for name in list(PER_NODE_ARRAYS) + list(SCALAR_ARRAYS) + ["loglik"]:
        path = _array_path(directory, name, layer, prefix)
        if path.exists():
            out[name] = np.load(path, mmap_mode="r")
    if not out:
        raise FileNotFoundError(
            f"no trace arrays for layer {layer} in {directory}"
        )
    return out

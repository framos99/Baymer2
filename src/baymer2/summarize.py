"""Turn on-disk posterior traces into per-layer rate and PIP JSONs.

Why
---
``fit`` writes only the deepest layer's rates (``cmd_fit`` takes
``layer = tree.n_layers - 1``), and it writes them one seed at a time. The
shallower layers, the spike-and-slab inclusion probabilities and the pooled
across-seed estimate all exist in the traces but had no reader. This module
is that reader: it never refits and never needs the count table, it only
streams ``.npy`` traces that are already on disk.

Legacy Baymer did the equivalent in
``plot_baymer_posterior_distributions.py``, which concatenated every chain
and then took the posterior mean. Adding per-seed sufficient statistics is
the same operation without materialising the concatenation, so one pass per
seed yields both the per-seed files and the pooled one.

Layout
------
Rate vectors are slotted by destination base, ``[p_A, p_C, p_G, p_T]``, with
the reference slot holding ``1 - sum(alts)`` -- legacy's ``get_post_rates``
convention, not the reference-first ordering of
:meth:`baymer2.model.ModelResult.rate_dict`. ``pip``, ``rate_sd`` and
``phi_mean`` use the same four slots so every field lines up by base index.

Ref-slot conventions:

``rates``
    ``1 - sum(alts)``.
``rate_sd``
    The genuine posterior SD of ``1 - sum(alts)``, accumulated in the same
    pass. Well defined, so it is a number rather than null.
``pip``, ``phi_mean``
    ``null``. There is no indicator and no phi for a non-substitution.

Layer 0 carries no indicator at all: ``sample_indicators`` sits behind
``if not is_root:`` in both backends, so ``indicator.layer_0.npy`` is left at
its zero initialisation. Its PIPs are written as ``null``; writing ``0.0``
would assert "never in the slab", which is not what the sampler measured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baymer2.contexts import BASES, BASE_INDEX, ContextTree, alt_bases
from baymer2.io.traces import MANIFEST_NAME, load_layer_traces

__all__ = [
    "SCHEMA",
    "LayerAccumulator",
    "accumulate_layer",
    "layer_payload",
    "read_manifest",
    "summarize_run",
    "tree_from_manifests",
]

#: Bumped when the emitted JSON changes shape.
SCHEMA = "baymer2.rates/1"

#: Draws converted to float64 per chunk. 50 draws of the deepest layer is
#: ~160 MB, and slicing the sample axis is a contiguous read; chunking the
#: node axis instead would stride across the whole file for every chunk.
CHUNK_DRAWS = 50

#: Significant digits retained in the JSON. Traces are float32, so this
#: discards nothing real -- and at sweep scale it is the difference between
#: a ~60 MB and a ~35 MB file for the deepest layer, times three files times
#: 577 species.
DIGITS = 8

#: Manifest keys that must agree across seeds before their draws are pooled.
_SHARED_KEYS = ("max_mer", "growth", "alternation", "likelihood", "linkage")


# ---------------------------------------------------------------------------
# sufficient statistics
# ---------------------------------------------------------------------------


@dataclass
class LayerAccumulator:
    """Pooled sufficient statistics for one layer of one or more chains.

    Everything needed downstream is a sum over draws, so chains combine by
    addition. That is exactly legacy's concatenate-then-average, and it lets
    a single streaming pass per seed serve both the per-seed output and the
    pooled one.
    """

    n_draws: int
    sum_rates: np.ndarray  # (n_nodes, 3)
    sum_rates_sq: np.ndarray  # (n_nodes, 3)
    sum_ref: np.ndarray  # (n_nodes,)
    sum_ref_sq: np.ndarray  # (n_nodes,)
    sum_phi: np.ndarray  # (n_nodes, 3)
    sum_indicator: np.ndarray | None  # (n_nodes, 3); None at the root layer

    def __add__(self, other: "LayerAccumulator") -> "LayerAccumulator":
        if self.sum_rates.shape != other.sum_rates.shape:
            raise ValueError(
                f"cannot pool layers of different width: "
                f"{self.sum_rates.shape} and {other.sum_rates.shape}"
            )
        if (self.sum_indicator is None) != (other.sum_indicator is None):
            raise ValueError(
                "cannot pool a layer that carries indicators with one that "
                "does not"
            )
        return LayerAccumulator(
            n_draws=self.n_draws + other.n_draws,
            sum_rates=self.sum_rates + other.sum_rates,
            sum_rates_sq=self.sum_rates_sq + other.sum_rates_sq,
            sum_ref=self.sum_ref + other.sum_ref,
            sum_ref_sq=self.sum_ref_sq + other.sum_ref_sq,
            sum_phi=self.sum_phi + other.sum_phi,
            sum_indicator=(
                None
                if self.sum_indicator is None
                else self.sum_indicator + other.sum_indicator
            ),
        )

    # -- posterior summaries ----------------------------------------------

    @property
    def mean_rates(self) -> np.ndarray:
        return self.sum_rates / self.n_draws

    @property
    def mean_phi(self) -> np.ndarray:
        return self.sum_phi / self.n_draws

    @property
    def mean_ref(self) -> np.ndarray:
        return self.sum_ref / self.n_draws

    @property
    def pip(self) -> np.ndarray | None:
        if self.sum_indicator is None:
            return None
        return self.sum_indicator / self.n_draws

    def sd_rates(self) -> np.ndarray:
        return _sd(self.sum_rates, self.sum_rates_sq, self.n_draws)

    def sd_ref(self) -> np.ndarray:
        return _sd(self.sum_ref, self.sum_ref_sq, self.n_draws)


def _sd(total: np.ndarray, total_sq: np.ndarray, n: int) -> np.ndarray:
    """Population SD from raw sums, clamped at zero.

    ``E[x^2] - E[x]^2`` can go slightly negative through cancellation when
    the posterior is nearly degenerate, which is common for the rate of a
    context the sampler pinned to the spike.
    """
    mean = total / n
    var = total_sq / n - mean * mean
    return np.sqrt(np.maximum(var, 0.0))


def accumulate_layer(
    trace_dir: str | Path,
    layer: int,
    *,
    prefix: str = "",
    chunk: int = CHUNK_DRAWS,
) -> LayerAccumulator:
    """Stream one layer of one chain into sufficient statistics.

    Resident memory is bounded by ``chunk``; the traces themselves stay
    memory-mapped and are read a contiguous slab of draws at a time.
    """
    arrays = load_layer_traces(trace_dir, layer, prefix=prefix)
    for name in ("rates", "phi"):
        if name not in arrays:
            raise FileNotFoundError(
                f"{trace_dir}: layer {layer} has no {name} trace, so it "
                f"cannot be summarised"
            )
    rates = arrays["rates"]
    phi = arrays["phi"]

    # The root layer's indicator file exists but was never sampled -- both
    # backends guard sample_indicators with `is_root = layer == 0`. Reading
    # it would report a confident 0.0 for every context.
    indicator = arrays.get("indicator") if layer != 0 else None

    n_draws, n_nodes, _ = rates.shape
    if phi.shape != rates.shape:
        raise ValueError(
            f"{trace_dir}: layer {layer} phi is {phi.shape} but rates is "
            f"{rates.shape}"
        )
    if indicator is not None and indicator.shape != rates.shape:
        raise ValueError(
            f"{trace_dir}: layer {layer} indicator is {indicator.shape} but "
            f"rates is {rates.shape}"
        )

    sum_rates = np.zeros((n_nodes, 3))
    sum_rates_sq = np.zeros((n_nodes, 3))
    sum_ref = np.zeros(n_nodes)
    sum_ref_sq = np.zeros(n_nodes)
    sum_phi = np.zeros((n_nodes, 3))
    sum_ind = np.zeros((n_nodes, 3)) if indicator is not None else None

    for lo in range(0, n_draws, chunk):
        hi = min(lo + chunk, n_draws)
        block = np.asarray(rates[lo:hi], dtype=np.float64)
        sum_rates += block.sum(axis=0)
        sum_rates_sq += np.square(block).sum(axis=0)

        # p_ref is a per-draw quantity, so its SD has to be accumulated
        # here; it is not recoverable from the alt-wise sums afterwards.
        ref = 1.0 - block.sum(axis=2)
        sum_ref += ref.sum(axis=0)
        sum_ref_sq += np.square(ref).sum(axis=0)

        sum_phi += np.asarray(phi[lo:hi], dtype=np.float64).sum(axis=0)
        if sum_ind is not None:
            sum_ind += np.asarray(indicator[lo:hi], dtype=np.float64).sum(axis=0)

    return LayerAccumulator(
        n_draws=int(n_draws),
        sum_rates=sum_rates,
        sum_rates_sq=sum_rates_sq,
        sum_ref=sum_ref,
        sum_ref_sq=sum_ref_sq,
        sum_phi=sum_phi,
        sum_indicator=sum_ind,
    )


# ---------------------------------------------------------------------------
# base slotting
# ---------------------------------------------------------------------------


def slot_map(tree: ContextTree, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """``(ref_slot, alt_slots)`` for every context in ``layer``.

    ``alt_slots[i, j]`` is the ``ACGT`` index that the model's ``j``-th
    alternate rate belongs in. Derived from :func:`baymer2.contexts.alt_bases`
    so the ordering rule lives in one place -- legacy reimplemented it in
    three files and they drifted.
    """
    contexts = tree.contexts(layer)
    ref_slot = np.empty(len(contexts), dtype=np.int64)
    alt_slots = np.empty((len(contexts), 3), dtype=np.int64)
    for i, ctx in enumerate(contexts):
        ref = tree.focal_base(ctx)
        ref_slot[i] = BASE_INDEX[ref]
        alt_slots[i] = [BASE_INDEX[b] for b in alt_bases(ref)]
    return ref_slot, alt_slots


def _slotted(
    values: np.ndarray,
    ref_values: np.ndarray | None,
    ref_slot: np.ndarray,
    alt_slots: np.ndarray,
) -> np.ndarray:
    """Scatter per-alt values into ``[A, C, G, T]`` columns.

    ``ref_values`` fills the reference slot, or ``NaN`` when the quantity is
    undefined there. NaN is the in-array marker only; :func:`_jsonable`
    turns it into ``null`` before anything is written, since JSON has no NaN.
    """
    out = np.empty((values.shape[0], 4))
    rows = np.arange(values.shape[0])[:, None]
    out[rows, alt_slots] = values
    out[np.arange(values.shape[0]), ref_slot] = (
        np.nan if ref_values is None else ref_values
    )
    return out


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


def _round_sig(a: np.ndarray, digits: int) -> np.ndarray:
    """Round to ``digits`` significant figures, elementwise and vectorised."""
    a = np.asarray(a, dtype=np.float64)
    out = np.array(a, copy=True)
    finite = np.isfinite(a) & (a != 0.0)
    if not finite.any():
        return out
    scale = 10.0 ** (digits - 1 - np.floor(np.log10(np.abs(a[finite]))))
    out[finite] = np.round(a[finite] * scale) / scale
    return out


def _jsonable(row: np.ndarray) -> list[float | None]:
    """One slotted row as JSON, with NaN rendered as ``null``."""
    return [None if not np.isfinite(v) else float(v) for v in row]


FIELD_DOCS = {
    "rates": (
        "posterior mean substitution probability; slot = destination base, "
        "ref slot = 1 - sum(alts)"
    ),
    "pip": (
        "posterior inclusion probability, the mean spike-and-slab indicator; "
        "null at the ref slot, and null throughout layer 0, which has no "
        "indicator"
    ),
    "rate_sd": "posterior SD of the rate across the pooled draws",
    "phi_mean": "posterior mean phi; null at the ref slot",
}


def layer_payload(
    acc: LayerAccumulator,
    tree: ContextTree,
    layer: int,
    source: dict,
    *,
    digits: int = DIGITS,
) -> dict:
    """Build the JSON document for one layer."""
    contexts = tree.contexts(layer)
    if len(contexts) != acc.sum_rates.shape[0]:
        raise ValueError(
            f"layer {layer} of the tree has {len(contexts)} contexts but the "
            f"traces have {acc.sum_rates.shape[0]} nodes; the manifest and "
            f"the trace files disagree"
        )
    ref_slot, alt_slots = slot_map(tree, layer)

    pip = acc.pip
    fields = {
        "rates": _slotted(acc.mean_rates, acc.mean_ref, ref_slot, alt_slots),
        "pip": _slotted(
            np.full((len(contexts), 3), np.nan) if pip is None else pip,
            None,
            ref_slot,
            alt_slots,
        ),
        "rate_sd": _slotted(acc.sd_rates(), acc.sd_ref(), ref_slot, alt_slots),
        "phi_mean": _slotted(acc.mean_phi, None, ref_slot, alt_slots),
    }
    fields = {k: _round_sig(v, digits) for k, v in fields.items()}

    k = tree.layer_k(layer)
    return {
        "schema": SCHEMA,
        "layer": layer,
        "k": k,
        "max_mer": tree.max_mer,
        "growth": tree.growth,
        "alternation": tree.alternation,
        "focal_index": tree.focal_index(k),
        "bases": list(BASES),
        "source": {**source, "n_draws": acc.n_draws},
        "fields": FIELD_DOCS,
        "contexts": {
            ctx: {name: _jsonable(arr[i]) for name, arr in fields.items()}
            for i, ctx in enumerate(contexts)
        },
    }


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------


def read_manifest(trace_dir: str | Path) -> dict:
    path = Path(trace_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{trace_dir} has no {MANIFEST_NAME}; it is written only after "
            f"every layer completes, so this fit did not finish"
        )
    return json.loads(path.read_text())


def _infer_growth(manifest: dict) -> str:
    """Recover ``growth`` from a manifest written before it was recorded.

    Layer 1 has ``2 * branching`` nodes, and branching is 4 under
    alternating growth against 16 under centred, so the width is
    unambiguous. A single-layer model has no layer 1, but then growth
    cannot affect anything either.
    """
    growth = manifest.get("growth")
    if growth is not None:
        return growth
    layers = {int(entry["layer"]): entry for entry in manifest.get("layers", [])}
    if 1 not in layers:
        return "alternating"
    n_nodes = int(layers[1]["n_nodes"])
    if n_nodes == 32:
        return "centred"
    if n_nodes == 8:
        return "alternating"
    raise ValueError(
        f"cannot infer growth: layer 1 has {n_nodes} nodes, which is neither "
        f"8 (alternating) nor 32 (centred)"
    )


def tree_from_manifests(
    manifests: list[dict], labels: list[str]
) -> tuple[ContextTree, dict]:
    """Reconcile the seeds' manifests into one tree plus shared provenance.

    Disagreement is an error rather than a warning. Pooling draws from two
    different models would produce a plausible-looking file with no valid
    interpretation, and nothing downstream could detect it.
    """
    resolved = [{**m, "growth": _infer_growth(m)} for m in manifests]
    shared = {}
    for key in _SHARED_KEYS:
        values = {m.get(key) for m in resolved}
        if len(values) > 1:
            detail = ", ".join(
                f"{label}={m.get(key)!r}" for label, m in zip(labels, resolved)
            )
            raise ValueError(
                f"traces disagree on {key!r} and cannot be pooled: {detail}"
            )
        shared[key] = values.pop()

    names = {m.get("count_name") for m in resolved}
    if len(names) > 1:
        raise ValueError(
            f"traces come from different count tables and cannot be pooled: "
            f"{sorted(str(n) for n in names)}"
        )

    tree = ContextTree(
        max_mer=int(shared["max_mer"]),
        growth=shared["growth"],
        alternation=shared["alternation"],
    )
    provenance = {
        "count_name": names.pop(),
        "likelihood": shared["likelihood"],
        "linkage": shared["linkage"],
        # Absent from manifests written before these were recorded. Reported
        # as null rather than back-filled with the default, which would state
        # as fact something the run did not record.
        "parameterization": _shared_optional(resolved, "parameterization"),
        "branch_length": _shared_optional(resolved, "branch_length"),
    }
    return tree, provenance


def _shared_optional(manifests: list[dict], key: str):
    values = {m.get(key) for m in manifests}
    return values.pop() if len(values) == 1 else None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def summarize_run(
    trace_dirs: list[str | Path],
    out_dir: str | Path,
    *,
    prefix: str = "rates",
    final_label: str = "final",
    digits: int = DIGITS,
    chunk: int = CHUNK_DRAWS,
    progress: bool = False,
) -> list[Path]:
    """Write per-seed and pooled rate/PIP JSONs for every layer.

    Returns the paths written, in the order they were written.
    """
    if not trace_dirs:
        raise ValueError("no trace directories given")
    trace_dirs = [Path(d) for d in trace_dirs]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = [read_manifest(d) for d in trace_dirs]
    tree, provenance = tree_from_manifests(manifests, [str(d) for d in trace_dirs])
    seeds = [m.get("seed") for m in manifests]
    trace_prefix = provenance["count_name"] or ""

    written: list[Path] = []
    for layer in range(tree.n_layers):
        k = tree.layer_k(layer)
        per_seed = []
        for trace_dir, seed in zip(trace_dirs, seeds):
            if progress:
                print(f"layer {layer} ({k}mer): reading {trace_dir}")
            acc = accumulate_layer(
                trace_dir, layer, prefix=trace_prefix, chunk=chunk
            )
            per_seed.append(acc)
            payload = layer_payload(
                acc,
                tree,
                layer,
                {
                    **provenance,
                    "seeds": [seed],
                    "traces": [str(trace_dir)],
                    "n_draws_per_seed": [acc.n_draws],
                },
                digits=digits,
            )
            written.append(
                _write(out_dir / f"{prefix}.{k}mer.rs{seed}.json", payload)
            )

        pooled = per_seed[0]
        for acc in per_seed[1:]:
            pooled = pooled + acc
        payload = layer_payload(
            pooled,
            tree,
            layer,
            {
                **provenance,
                "seeds": seeds,
                "traces": [str(d) for d in trace_dirs],
                "n_draws_per_seed": [a.n_draws for a in per_seed],
            },
            digits=digits,
        )
        written.append(
            _write(out_dir / f"{prefix}.{k}mer.{final_label}.json", payload)
        )
    return written


def _write(path: Path, payload: dict) -> Path:
    # allow_nan=False so a NaN can never be written as the bare token `NaN`,
    # which json.loads accepts but no other parser does. Undefined entries
    # have already been converted to None by _jsonable.
    path.write_text(json.dumps(payload, allow_nan=False))
    return path

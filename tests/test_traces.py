"""Disk-backed traces must be transparent and byte-faithful.

Streaming to disk is only useful if it changes nothing about the answer,
so the central test runs the same fit twice -- once in RAM, once
memory-mapped -- with the same seed and demands identical output.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from baymer2.backends.numba_mh import LayerConfig
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.io.traces import (
    PER_NODE_ARRAYS,
    SCALAR_ARRAYS,
    TraceStore,
    load_layer_traces,
)
from baymer2.model import ModelConfig, fit

from conftest import GOLDEN_ALTERNATION, GOLDEN_MAX_MER


@pytest.fixture(scope="module")
def table(golden_count_dicts) -> CountTable:
    tree = ContextTree(max_mer=GOLDEN_MAX_MER, alternation=GOLDEN_ALTERNATION)
    return CountTable.from_count_dict_set(golden_count_dicts, tree, name="AFR")


def _config(seed: int = 14, iterations: int = 600) -> ModelConfig:
    return ModelConfig(
        max_mer=GOLDEN_MAX_MER,
        seed=seed,
        default_layer=LayerConfig(
            n_iterations=iterations,
            burnin=iterations // 2,
            thin=5,
            set_sigma=0.8540486,
            c=0.04472136,
            phi_proposal_sd=0.05,
        ),
    )


# --------------------------------------------------------------------------
# Store mechanics
# --------------------------------------------------------------------------


def test_allocate_layer_creates_expected_arrays(tmp_path):
    store = TraceStore(tmp_path, prefix="AFR")
    arrays = store.allocate_layer(layer=2, n_samples=10, n_nodes=32)
    for name in PER_NODE_ARRAYS:
        assert arrays[name].shape == (10, 32, 3)
    for name in SCALAR_ARRAYS:
        assert arrays[name].shape == (10,)
    assert (tmp_path / "AFR.eta.layer_2.npy").exists()


def test_written_values_survive_reopening(tmp_path):
    store = TraceStore(tmp_path)
    arrays = store.allocate_layer(layer=0, n_samples=4, n_nodes=2)
    expected = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    arrays["eta"][:] = expected
    arrays["alpha"][:] = [0.1, 0.2, 0.3, 0.4]
    store.flush_layer(0)

    reloaded = load_layer_traces(tmp_path, 0)
    np.testing.assert_array_equal(reloaded["eta"], expected)
    np.testing.assert_allclose(reloaded["alpha"], [0.1, 0.2, 0.3, 0.4])


def test_truncate_drops_unwritten_samples(tmp_path):
    store = TraceStore(tmp_path)
    arrays = store.allocate_layer(layer=1, n_samples=10, n_nodes=8)
    arrays["eta"][:6] = 1.0
    store.flush_layer(1)
    final = store.truncate_layer(1, n_written=6, n_allocated=10)
    assert final["eta"].shape == (6, 8, 3)
    assert np.all(final["eta"] == 1.0)


def test_truncate_is_a_noop_when_full(tmp_path):
    store = TraceStore(tmp_path)
    arrays = store.allocate_layer(layer=1, n_samples=5, n_nodes=8)
    arrays["eta"][:] = 2.0
    store.flush_layer(1)
    final = store.truncate_layer(1, n_written=5, n_allocated=5)
    assert final["eta"].shape == (5, 8, 3)


def test_missing_layer_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no trace arrays"):
        load_layer_traces(tmp_path, 3)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no manifest"):
        TraceStore(tmp_path).read_manifest()


def test_zero_samples_rejected(tmp_path):
    with pytest.raises(ValueError, match="n_samples"):
        TraceStore(tmp_path).allocate_layer(layer=0, n_samples=0, n_nodes=2)


# --------------------------------------------------------------------------
# Transparency: streaming must not change the answer
# --------------------------------------------------------------------------


def test_streamed_fit_is_identical_to_in_memory_fit(table, tmp_path):
    """The whole point: disk backing changes storage, not results."""
    in_memory = fit(table, _config(seed=14))
    streamed = fit(table, _config(seed=14), trace_dir=tmp_path)

    for layer in range(GOLDEN_MAX_MER):
        np.testing.assert_array_equal(
            np.asarray(in_memory.layers[layer].rates),
            np.asarray(streamed.layers[layer].rates),
            err_msg=f"layer {layer} rates differ between storage modes",
        )
        np.testing.assert_array_equal(
            np.asarray(in_memory.layers[layer].eta),
            np.asarray(streamed.layers[layer].eta),
        )


def test_streamed_fit_writes_every_layer(table, tmp_path):
    fit(table, _config(), trace_dir=tmp_path)
    for layer in range(GOLDEN_MAX_MER):
        arrays = load_layer_traces(tmp_path, layer, prefix="AFR")
        assert set(PER_NODE_ARRAYS).issubset(arrays)
        assert arrays["rates"].shape[1] == ContextTree(max_mer=GOLDEN_MAX_MER).layer_size(layer)


def test_manifest_records_the_run(table, tmp_path):
    result = fit(table, _config(), trace_dir=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["max_mer"] == GOLDEN_MAX_MER
    assert manifest["likelihood"] == "binomial"
    assert manifest["count_name"] == "AFR"
    assert len(manifest["layers"]) == GOLDEN_MAX_MER
    assert manifest["layers"][4]["n_nodes"] == 512
    assert result.trace_store is not None


def test_traces_are_reloadable_without_refitting(table, tmp_path):
    result = fit(table, _config(), trace_dir=tmp_path)
    reloaded = load_layer_traces(tmp_path, 4, prefix="AFR")
    np.testing.assert_allclose(
        np.asarray(reloaded["rates"]).mean(axis=0),
        result.rates(4),
        rtol=1e-6,
    )


def test_streamed_arrays_are_memory_mapped(table, tmp_path):
    """Confirm the arrays are genuinely on disk, not silently in RAM."""
    result = fit(table, _config(), trace_dir=tmp_path)
    assert isinstance(result.layers[4].rates, np.memmap)
    assert result.trace_store.nbytes() > 0

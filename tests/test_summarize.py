"""Summarising traces must preserve legacy's layout and the model's meaning.

Two things can silently go wrong here and produce a file that looks fine:
the rate vector can be slotted against the wrong base, and the pooled
estimate can weight chains incorrectly. Both are checked against oracles
that do not share code with the implementation -- legacy Baymer's own
slotting rule, reimplemented below, and an explicit concatenate-then-average
over the raw traces.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from baymer2.backends.numba_mh import LayerConfig
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.io.traces import load_layer_traces
from baymer2.model import ModelConfig, fit
from baymer2.summarize import (
    SCHEMA,
    accumulate_layer,
    layer_payload,
    summarize_run,
    tree_from_manifests,
)

from conftest import GOLDEN_ALTERNATION, GOLDEN_MAX_MER


# --------------------------------------------------------------------------
# Independent oracle: legacy's own slotting rule
# --------------------------------------------------------------------------


def legacy_ref_pos(mer_length: int, alternation_pattern: str) -> int:
    """Verbatim port of ``get_ref_pos`` from legacy Baymer.

    Reimplemented rather than imported so that a drift in
    ``ContextTree.focal_index`` actually fails this test instead of moving
    the oracle with it.
    """
    flank = mer_length / 2
    ref_pos = flank
    if mer_length % 2 == 0:
        if alternation_pattern == "left":
            ref_pos = flank
        elif alternation_pattern == "right":
            ref_pos = flank - 1
        else:
            raise ValueError("bad alternation pattern")
    return int(ref_pos)


def legacy_post_rates(context_post_rates, ref_nuc_id):
    """Verbatim port of legacy ``get_post_rates``."""
    rates = [0, 0, 0, 0]
    sub_index = 0
    for index in range(4):
        if index == ref_nuc_id:
            continue
        rates[index] = context_post_rates[sub_index]
        sub_index += 1
    rates[ref_nuc_id] = 1.0 - sum(rates)
    return rates


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table(golden_count_dicts) -> CountTable:
    tree = ContextTree(max_mer=GOLDEN_MAX_MER, alternation=GOLDEN_ALTERNATION)
    return CountTable.from_count_dict_set(golden_count_dicts, tree, name="AFR")


def _config(seed: int, iterations: int = 600) -> ModelConfig:
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


@pytest.fixture(scope="module")
def two_seed_run(table, tmp_path_factory):
    """Two finished fits plus their summary, shared across tests."""
    root = tmp_path_factory.mktemp("two_seed")
    dirs = []
    for seed in (14, 15):
        d = root / f"traces.rs{seed}"
        fit(table, _config(seed), trace_dir=d)
        dirs.append(d)
    out = root / "out"
    paths = summarize_run(dirs, out)
    return dirs, out, paths


def _load(out_dir, k, label):
    return json.loads((out_dir / f"rates.{k}mer.{label}.json").read_text())


# --------------------------------------------------------------------------
# Coverage: every layer, named by k
# --------------------------------------------------------------------------


def test_every_layer_is_written_for_every_seed_and_pooled(two_seed_run):
    _, out, paths = two_seed_run
    assert len(paths) == GOLDEN_MAX_MER * 3
    for layer in range(GOLDEN_MAX_MER):
        k = layer + 1  # alternating growth
        for label in ("rs14", "rs15", "final"):
            assert (out / f"rates.{k}mer.{label}.json").exists()


def test_files_are_named_by_context_length_not_layer_index():
    """Centred growth makes these differ: layers 0..4 are 1/3/5/7/9-mers."""
    tree = ContextTree(max_mer=9, growth="centred")
    assert [tree.layer_k(l) for l in range(tree.n_layers)] == [1, 3, 5, 7, 9]


def test_context_count_matches_the_tree(two_seed_run):
    _, out, _ = two_seed_run
    tree = ContextTree(max_mer=GOLDEN_MAX_MER, alternation=GOLDEN_ALTERNATION)
    for layer in range(GOLDEN_MAX_MER):
        doc = _load(out, layer + 1, "final")
        assert len(doc["contexts"]) == tree.layer_size(layer)
        assert set(doc["contexts"]) == set(tree.contexts(layer))
        assert doc["schema"] == SCHEMA
        assert doc["k"] == layer + 1
        assert doc["layer"] == layer


# --------------------------------------------------------------------------
# Layout: legacy base slotting
# --------------------------------------------------------------------------


def test_slotting_matches_the_legacy_oracle(two_seed_run):
    dirs, out, _ = two_seed_run
    tree = ContextTree(max_mer=GOLDEN_MAX_MER, alternation=GOLDEN_ALTERNATION)
    bases = ["A", "C", "G", "T"]

    for layer in range(GOLDEN_MAX_MER):
        k = layer + 1
        doc = _load(out, k, "rs14")
        traces = load_layer_traces(dirs[0], layer, prefix="AFR")
        means = np.asarray(traces["rates"]).mean(axis=0)

        assert doc["focal_index"] == legacy_ref_pos(k, GOLDEN_ALTERNATION)

        for i, ctx in enumerate(tree.contexts(layer)):
            ref_nuc = ctx[legacy_ref_pos(k, GOLDEN_ALTERNATION)]
            expected = legacy_post_rates(means[i], bases.index(ref_nuc))
            np.testing.assert_allclose(
                doc["contexts"][ctx]["rates"], expected, rtol=1e-6, atol=1e-9,
                err_msg=f"layer {layer} context {ctx} slotted wrongly",
            )


def test_reference_slot_is_one_minus_the_alternates(two_seed_run):
    _, out, _ = two_seed_run
    for layer in range(GOLDEN_MAX_MER):
        doc = _load(out, layer + 1, "final")
        focal = doc["focal_index"]
        for ctx, entry in doc["contexts"].items():
            ref_slot = "ACGT".index(ctx[focal])
            rates = entry["rates"]
            alts = [v for j, v in enumerate(rates) if j != ref_slot]
            assert rates[ref_slot] == pytest.approx(1.0 - sum(alts), abs=1e-7)


def test_every_vector_has_four_slots(two_seed_run):
    _, out, _ = two_seed_run
    for layer in range(GOLDEN_MAX_MER):
        doc = _load(out, layer + 1, "final")
        for entry in doc["contexts"].values():
            for name in ("rates", "pip", "rate_sd", "phi_mean"):
                assert len(entry[name]) == 4, name


# --------------------------------------------------------------------------
# Inclusion probabilities
# --------------------------------------------------------------------------


def test_layer_zero_pips_are_null_not_zero(two_seed_run):
    """The root has no indicator; 0.0 would assert 'never in the slab'."""
    _, out, _ = two_seed_run
    for label in ("rs14", "rs15", "final"):
        doc = _load(out, 1, label)
        for entry in doc["contexts"].values():
            assert entry["pip"] == [None, None, None, None]


def test_deeper_pips_are_probabilities_with_a_null_reference_slot(two_seed_run):
    _, out, _ = two_seed_run
    for layer in range(1, GOLDEN_MAX_MER):
        doc = _load(out, layer + 1, "final")
        focal = doc["focal_index"]
        seen_nonzero = False
        for ctx, entry in doc["contexts"].items():
            ref_slot = "ACGT".index(ctx[focal])
            assert entry["pip"][ref_slot] is None
            for j, v in enumerate(entry["pip"]):
                if j == ref_slot:
                    continue
                assert v is not None
                assert 0.0 <= v <= 1.0
                seen_nonzero |= v > 0.0
        assert seen_nonzero, f"layer {layer} PIPs are all zero"


def test_pip_equals_the_mean_indicator_from_the_raw_trace(two_seed_run):
    dirs, out, _ = two_seed_run
    tree = ContextTree(max_mer=GOLDEN_MAX_MER, alternation=GOLDEN_ALTERNATION)
    layer = 3
    traces = load_layer_traces(dirs[0], layer, prefix="AFR")
    expected = np.asarray(traces["indicator"]).mean(axis=0)
    doc = _load(out, layer + 1, "rs14")
    focal = doc["focal_index"]
    for i, ctx in enumerate(tree.contexts(layer)):
        ref_slot = "ACGT".index(ctx[focal])
        got = [v for j, v in enumerate(doc["contexts"][ctx]["pip"]) if j != ref_slot]
        np.testing.assert_allclose(got, expected[i], rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------


def test_pooled_mean_equals_concatenating_the_chains(two_seed_run):
    """The oracle is the concatenation legacy actually performed."""
    dirs, out, _ = two_seed_run
    tree = ContextTree(max_mer=GOLDEN_MAX_MER, alternation=GOLDEN_ALTERNATION)
    layer = 2
    stacked = np.concatenate(
        [np.asarray(load_layer_traces(d, layer, prefix="AFR")["rates"]) for d in dirs]
    )
    expected = stacked.mean(axis=0)
    doc = _load(out, layer + 1, "final")
    focal = doc["focal_index"]
    for i, ctx in enumerate(tree.contexts(layer)):
        ref_slot = "ACGT".index(ctx[focal])
        got = [v for j, v in enumerate(doc["contexts"][ctx]["rates"]) if j != ref_slot]
        np.testing.assert_allclose(got, expected[i], rtol=1e-6, atol=1e-9)


def test_pooled_sd_is_across_chains_not_within(two_seed_run):
    dirs, out, _ = two_seed_run
    tree = ContextTree(max_mer=GOLDEN_MAX_MER, alternation=GOLDEN_ALTERNATION)
    layer = 2
    stacked = np.concatenate(
        [np.asarray(load_layer_traces(d, layer, prefix="AFR")["rates"]) for d in dirs]
    )
    expected = stacked.std(axis=0)
    doc = _load(out, layer + 1, "final")
    focal = doc["focal_index"]
    for i, ctx in enumerate(tree.contexts(layer)):
        ref_slot = "ACGT".index(ctx[focal])
        got = [
            v for j, v in enumerate(doc["contexts"][ctx]["rate_sd"]) if j != ref_slot
        ]
        np.testing.assert_allclose(got, expected[i], rtol=1e-5, atol=1e-9)


def test_pooling_weights_by_draw_count(two_seed_run):
    """Unequal chains must not be averaged as if they were equal."""
    dirs, _, _ = two_seed_run
    a = accumulate_layer(dirs[0], 2, prefix="AFR")
    b = accumulate_layer(dirs[1], 2, prefix="AFR")

    # Halve one chain's weight by hand and check the combination tracks it.
    half = type(a)(
        n_draws=a.n_draws // 2,
        sum_rates=a.sum_rates / 2,
        sum_rates_sq=a.sum_rates_sq / 2,
        sum_ref=a.sum_ref / 2,
        sum_ref_sq=a.sum_ref_sq / 2,
        sum_phi=a.sum_phi / 2,
        sum_indicator=a.sum_indicator / 2,
    )
    pooled = half + b
    assert pooled.n_draws == a.n_draws // 2 + b.n_draws
    expected = (half.sum_rates + b.sum_rates) / pooled.n_draws
    np.testing.assert_allclose(pooled.mean_rates, expected)
    # The unweighted average of the two means would be a different number.
    naive = (half.mean_rates + b.mean_rates) / 2
    assert not np.allclose(pooled.mean_rates, naive, rtol=1e-9)


def test_single_seed_still_writes_a_final_file(table, tmp_path):
    d = tmp_path / "traces.rs14"
    fit(table, _config(14), trace_dir=d)
    summarize_run([d], tmp_path / "out")
    doc = _load(tmp_path / "out", 3, "final")
    assert doc["source"]["seeds"] == [14]
    assert doc["source"]["n_draws"] == doc["source"]["n_draws_per_seed"][0]


# --------------------------------------------------------------------------
# Provenance and refusal to pool incompatible runs
# --------------------------------------------------------------------------


def test_source_records_what_produced_the_numbers(two_seed_run):
    dirs, out, _ = two_seed_run
    doc = _load(out, 5, "final")
    src = doc["source"]
    assert src["seeds"] == [14, 15]
    assert src["count_name"] == "AFR"
    assert src["likelihood"] == "binomial"
    assert src["linkage"] == "cut"
    assert src["parameterization"] == "alr"
    assert src["n_draws"] == sum(src["n_draws_per_seed"])
    assert [str(d) for d in dirs] == src["traces"]


def test_manifest_now_records_growth_and_parameterization(table, tmp_path):
    fit(table, _config(14), trace_dir=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["growth"] == "alternating"
    assert manifest["parameterization"] == "alr"
    assert manifest["branch_length"] == 1.0


def test_growth_is_inferred_when_the_manifest_predates_it():
    """Runs already on disk have no `growth` key; layer widths give it."""
    centred = {
        "max_mer": 9, "alternation": "right", "likelihood": "recurrent",
        "linkage": "cut", "count_name": "X",
        "layers": [{"layer": 1, "n_nodes": 32}],
    }
    tree, _ = tree_from_manifests([centred], ["x"])
    assert tree.growth == "centred"
    assert tree.n_layers == 5

    alternating = {**centred, "max_mer": 5,
                   "layers": [{"layer": 1, "n_nodes": 8}]}
    tree, _ = tree_from_manifests([alternating], ["x"])
    assert tree.growth == "alternating"


def test_incompatible_runs_are_refused_rather_than_pooled():
    base = {
        "max_mer": 5, "growth": "alternating", "alternation": "right",
        "likelihood": "binomial", "linkage": "cut", "count_name": "AFR",
    }
    with pytest.raises(ValueError, match="max_mer"):
        tree_from_manifests([base, {**base, "max_mer": 7}], ["a", "b"])
    with pytest.raises(ValueError, match="likelihood"):
        tree_from_manifests([base, {**base, "likelihood": "ctmc"}], ["a", "b"])
    with pytest.raises(ValueError, match="different count tables"):
        tree_from_manifests([base, {**base, "count_name": "EUR"}], ["a", "b"])


def test_unrecorded_settings_are_null_not_guessed():
    """A default must not be reported as though the run recorded it."""
    manifest = {
        "max_mer": 5, "growth": "alternating", "alternation": "right",
        "likelihood": "binomial", "linkage": "cut", "count_name": "AFR",
    }
    _, provenance = tree_from_manifests([manifest], ["a"])
    assert provenance["parameterization"] is None
    assert provenance["branch_length"] is None


def test_missing_manifest_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="did not finish"):
        summarize_run([tmp_path], tmp_path / "out")


# --------------------------------------------------------------------------
# JSON hygiene
# --------------------------------------------------------------------------


def test_no_nan_reaches_the_file(two_seed_run):
    """json.dumps writes a bare NaN token that only Python can read back."""
    _, out, paths = two_seed_run
    for path in paths:
        assert "NaN" not in path.read_text()
        assert "Infinity" not in path.read_text()


def test_rounding_keeps_float32_precision(two_seed_run):
    dirs, out, _ = two_seed_run
    tree = ContextTree(max_mer=GOLDEN_MAX_MER, alternation=GOLDEN_ALTERNATION)
    layer = 4
    # The oracle must accumulate in float64 too. A float32 mean over 60
    # draws carries ~1e-7 relative error of its own, which is larger than
    # the rounding this test is trying to measure.
    means = np.asarray(
        load_layer_traces(dirs[0], layer, prefix="AFR")["rates"], dtype=np.float64
    ).mean(axis=0)
    doc = _load(out, layer + 1, "rs14")
    focal = doc["focal_index"]
    for i, ctx in enumerate(tree.contexts(layer)):
        ref_slot = "ACGT".index(ctx[focal])
        got = [v for j, v in enumerate(doc["contexts"][ctx]["rates"]) if j != ref_slot]
        # Only the deliberate rounding should remain. Round-to-8-significant-
        # figures is worst when the leading digit is 1, where half an ulp of
        # the last digit is 5e-8 relative -- so 1e-7 is the honest bound, and
        # it still fails loudly if the rounding ever drops to 6 digits.
        np.testing.assert_allclose(got, means[i], rtol=1e-7, atol=1e-15)


def test_payload_rejects_a_tree_that_does_not_match_the_traces(two_seed_run):
    dirs, _, _ = two_seed_run
    acc = accumulate_layer(dirs[0], 2, prefix="AFR")
    wrong = ContextTree(max_mer=9, growth="centred")
    with pytest.raises(ValueError, match="disagree"):
        layer_payload(acc, wrong, 2, {})

"""Shared fixtures and paths for the baymer2 test suite.

Two kinds of test need something this repository cannot ship. Tests
marked ``legacy`` compare against a checkout of the original Baymer
(https://github.com/bvoightlab/Baymer), the only oracle that exists for
the count pipeline and the sampler. Tests marked ``hal`` need a real HAL
alignment and the HAL toolkit. Both are located through environment
variables and skip cleanly when unset:

===========================  ==================================================
``BAYMER2_LEGACY_ROOT``      Baymer checkout; must contain ``tutorial_data/``.
``BAYMER2_EVOLVER_HAL``      ``evolverMammals.hal``, shipped with Cactus.
``BAYMER2_CACTUS_BIN``       Directory holding ``halStats``/``hal2maf``.
===========================  ==================================================

Everything else runs from the golden fixtures under ``tests/golden/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
GOLDEN_DIR = TESTS_DIR / "golden"
COUNT_GOLDEN = GOLDEN_DIR / "count_pipeline"
INDEX_GOLDEN = GOLDEN_DIR / "index_dicts"

LEGACY_ROOT_ENV = "BAYMER2_LEGACY_ROOT"
EVOLVER_HAL_ENV = "BAYMER2_EVOLVER_HAL"
CACTUS_BIN_ENV = "BAYMER2_CACTUS_BIN"


def _from_env(var: str) -> Path:
    """The configured path, or a self-describing one that cannot exist.

    Returning a ``Path`` rather than ``None`` keeps module-level path
    composition (``LEGACY_TUTORIAL / "chr1...vcf.bgz"``) working when the
    variable is unset, and every gate is already an ``.exists()`` check --
    which is simply False, so the test skips and says which variable to set.
    """
    value = os.environ.get(var)
    return Path(value) if value else Path(f"/nonexistent/<unset:${var}>")


#: Legacy Baymer checkout used as a behavioural oracle.
LEGACY_ROOT = _from_env(LEGACY_ROOT_ENV)
LEGACY_TUTORIAL = LEGACY_ROOT / "tutorial_data"

#: Small real HAL shipped with the cactus binaries; used for alignment tests.
EVOLVER_HAL = _from_env(EVOLVER_HAL_ENV)
CACTUS_BIN = _from_env(CACTUS_BIN_ENV)

#: Settings the golden fixtures were generated under.
GOLDEN_MAX_MER = 5
GOLDEN_ALTERNATION = "right"


def _require(path: Path, why: str):
    if not path.exists():
        pytest.skip(f"{why} not available at {path}")
    return path


@pytest.fixture(scope="session")
def legacy_root() -> Path:
    return _require(LEGACY_ROOT, "legacy Baymer checkout")


@pytest.fixture(scope="session")
def evolver_hal() -> Path:
    return _require(EVOLVER_HAL, "evolverMammals.hal")


@pytest.fixture(scope="session")
def legacy_index_dicts() -> dict[int, list[str]]:
    """Context ordering emitted by legacy ``run_baymer.py``, per layer.

    ``index_dict.layer_N.json`` maps array index to context string; this
    returns it as a positional list so it can be compared directly with
    ``ContextTree.contexts(layer)``.
    """
    out: dict[int, list[str]] = {}
    for path in sorted(INDEX_GOLDEN.glob("index_dict.layer_*.json")):
        layer = int(path.stem.rsplit("_", 1)[1])
        raw = json.loads(path.read_text())
        out[layer] = [raw[str(i)] for i in range(len(raw))]
    if not out:
        pytest.skip("no golden index dicts checked in")
    return out


@pytest.fixture(scope="session")
def golden_count_dicts() -> dict[int, dict[str, list[int]]]:
    """Legacy folded count dicts, keyed by k."""
    out: dict[int, dict[str, list[int]]] = {}
    for k in range(1, GOLDEN_MAX_MER + 1):
        path = COUNT_GOLDEN / f"{k}mer.ALL.AFR.cpg_islands.folded_count_dict.json"
        if path.exists():
            out[k] = json.loads(path.read_text())
    if not out:
        pytest.skip("no golden count dicts checked in")
    return out

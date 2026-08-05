"""The likelihood seam.

A backend must be able to evaluate a node's log-likelihood inside a
numba ``njit`` region, which rules out calling Python objects. So a
likelihood is split in two:

* a small Python object (:class:`Likelihood`) that validates
  configuration, precomputes whatever per-context data it needs, and
  reports how the sampler should be set up;
* a :class:`LikelihoodSpec`, a plain bundle of an integer mode code and
  float arrays, which is what actually crosses into compiled code.

Dispatching on an integer inside one compiled kernel is less elegant
than passing jitted callables around, but it compiles predictably and
keeps every mode in a single specialisation rather than forcing a
recompile per combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

#: Integer codes crossing into compiled kernels. Values are part of the
#: on-disk trace format, so append rather than renumber.
MODES = {
    "binomial": 0,
    "ctmc": 1,
    "recurrent": 2,
}

MODE_NAMES = {v: k for k, v in MODES.items()}


@dataclass(frozen=True)
class LikelihoodSpec:
    """Numba-facing payload describing a configured likelihood.

    Attributes
    ----------
    mode:
        One of :data:`MODES`.
    scalars:
        Mode-specific scalar parameters, e.g. ``[branch_length]`` for
        ``ctmc``. Always float64, always at least length 1 so numba sees
        a consistent type.
    per_node:
        Optional ``(n_nodes, m)`` float64 array of per-node auxiliary
        data, e.g. observed event multiplicities. Shape ``(0, 0)`` when
        the mode needs none.
    """

    mode: int
    scalars: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    per_node: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
    )

    def __post_init__(self) -> None:
        if self.mode not in MODE_NAMES:
            raise ValueError(f"unknown likelihood mode code {self.mode}")
        object.__setattr__(
            self, "scalars", np.ascontiguousarray(self.scalars, dtype=np.float64)
        )
        object.__setattr__(
            self, "per_node", np.ascontiguousarray(self.per_node, dtype=np.float64)
        )
        if self.scalars.ndim != 1 or self.scalars.size < 1:
            raise ValueError("scalars must be a 1-D array of length >= 1")
        if self.per_node.ndim != 2:
            raise ValueError("per_node must be 2-D (possibly empty)")

    @property
    def name(self) -> str:
        return MODE_NAMES[self.mode]


@runtime_checkable
class Likelihood(Protocol):
    """Configuration object for one likelihood mode."""

    #: Human-readable mode name; must be a key of :data:`MODES`.
    name: str

    def spec(self, layer: int) -> LikelihoodSpec:
        """Numba-facing payload for ``layer``."""
        ...

    def log_likelihood(self, stats: np.ndarray, rates: np.ndarray) -> np.ndarray:
        """Reference implementation, in pure numpy.

        Parameters
        ----------
        stats:
            ``(n_nodes, 4)`` reference-first sufficient statistics.
        rates:
            ``(n_nodes, 3)`` alternate-allele rate vectors.

        Returns
        -------
        ``(n_nodes,)`` per-node log-likelihoods.

        This is what the tests compare compiled kernels against; it must
        stay readable rather than fast.
        """
        ...

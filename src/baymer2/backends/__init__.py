"""Sampler backends behind a common interface.

A backend consumes sufficient statistics and a likelihood spec and
produces posterior draws. Swapping one for another must not require
changes to :mod:`baymer2.counts` or :mod:`baymer2.likelihoods`.
"""

from baymer2.backends.numba_mh import (
    LayerConfig,
    LayerResult,
    NumbaMetropolisBackend,
)

__all__ = ["LayerConfig", "LayerResult", "NumbaMetropolisBackend"]

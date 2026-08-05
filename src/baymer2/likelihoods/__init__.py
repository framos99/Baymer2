"""Selectable likelihoods over the same sufficient statistics.

Each likelihood answers one question: given a node's pooled counts and a
candidate rate vector, how probable are the data? They differ in what
generated those counts.

``binomial``
    One Bernoulli/multinomial trial per site. The legacy Baymer model.
    Correct for polymorphism data when the per-site mutation probability
    is small enough that a site is never hit twice.

``ctmc``
    A continuous-time Markov chain run along a branch. Observed
    focal-versus-ancestor differences understate true mutation because
    of back mutation, parallel mutation, and saturation; this inverts
    that. Reduces to ``binomial`` as branch length goes to zero.

``recurrent``
    Allows a site to be mutated more than once, either through an
    at-least-one-hit correction or through explicit per-site event
    multiplicities.

All of them consume :meth:`~baymer2.counts.CountTable.layer_stats` and
none of them may require changes to a sampler backend.
"""

from baymer2.likelihoods.base import Likelihood, LikelihoodSpec, MODES
from baymer2.likelihoods.binomial import BinomialLikelihood
from baymer2.likelihoods.ctmc import CTMCLikelihood
from baymer2.likelihoods.recurrent import RecurrentLikelihood

__all__ = [
    "Likelihood",
    "LikelihoodSpec",
    "MODES",
    "BinomialLikelihood",
    "CTMCLikelihood",
    "RecurrentLikelihood",
]

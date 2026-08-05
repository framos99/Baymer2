"""Baymer2 - context-dependent mutation spectrum inference.

A rewrite of Baymer (Adams et al., https://github.com/bvoightlab/Baymer) with:

* sufficient-statistic likelihoods, so a rate evaluation is O(1) in the
  k-mer depth rather than O(4**k);
* selectable likelihoods for polymorphism, substitution (multiple-hit
  aware), and recurrent mutation data;
* alignment input, so a Cactus HAL plus a focal species is a valid
  starting point alongside FASTA + VCF.

The three seams that matter are :class:`~baymer2.counts.CountTable`
(data in), the ``Likelihood`` protocol (the model), and the ``Backend``
protocol (the sampler). Adding a likelihood must never require touching
a backend.
"""

__version__ = "0.1.0"

from baymer2.contexts import ContextTree

__all__ = ["ContextTree", "__version__"]

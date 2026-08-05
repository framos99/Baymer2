# Validation, performance, and calibration

What was checked against the original implementation, what was measured
rather than projected, and the defects in the original that are fixed here.

## Validation

The count pipeline is deterministic and reproduces legacy **exactly**: TSVs are
byte-identical and count JSONs differ only in key ordering.

Sufficient statistics are checked against two independent legacy oracles — leaf
enumeration as `run_baymer.initialize_layer_tree_edge_list` did it, and the
separately-coded down-tree marginalisation that produced the shipped 1–4mer
dicts.

The sampler cannot be compared bit-for-bit: the ALR reparameterisation changes
the proposal geometry and the corrected alpha step changes the chain. So the
test asks whether the two implementations differ by more than one implementation
differs from *itself* under another seed. On the tutorial data (AFR, cpg_islands,
5-mer, 40000 iterations):

| layer | b2 seed-to-seed | b2 vs legacy | ratio |
|---|---|---|---|
| 0 | 0.0005 | 0.0005 | 1.03 |
| 1 | 0.0016 | 0.0020 | 1.22 |
| 2 | 0.0032 | 0.0061 | 1.91 |
| 3 | 0.0069 | 0.0099 | 1.44 |
| 4 | 0.0122 | 0.0122 | 1.00 |

Every layer sits at or near 1.0 — agreement within Monte Carlo error. Layer 0,
which has a flat prior on `p`, additionally lands on the empirical MLE.

## Measured performance

Benchmarked on one cluster node, Python 3.12, numba 0.61, 8 threads.

**Zero-context membership test** — the dominant bottleneck. Legacy's
`if i in zero_contexts` is lowered by numba to a linear scan, sitting inside the
per-allele loop of `update_alpha_probabilities`, which runs on ~79% of
iterations. Deepest layer of a 9-mer model (131,072 nodes), one call:

| unobserved contexts | legacy scan | boolean mask | speedup |
|---|---|---|---|
| 0.1% | 16.0 ms | 0.267 ms | 60× |
| 1% | 163.5 ms | 0.259 ms | 631× |
| 10% | 1509.3 ms | 0.350 ms | **4306×** |

At 10% unobserved × 40,000 iterations that is ~13 hours in one function, at one
layer. It only bites when contexts are genuinely unobserved — the norm for
9-mers on anything but a very large callable region.

**End to end.** Both implementations on identical inputs and iteration counts
(`benchmarks/bench_end_to_end.sh`):

| case | legacy | baymer2 | speedup | legacy RSS | baymer2 RSS |
|---|---|---|---|---|---|
| k=5, 8000 iter, real tutorial data | 31.09 s | 8.04 s | **3.9×** | 253 MB | 182 MB |
| k=9, 400 iter, 10% unobserved | 841.76 s | 30.25 s | **27.8×** | 489 MB | 371 MB |

The speedup grows with depth because the zero-context scan does: legacy's cost
there is proportional to the number of empty contexts, and deep k-mers on any
realistic callable region have many.

**Likelihood term** — real but modest, and smaller than first estimated. Legacy
reads `2·4^max_mer` elements per sweep regardless of layer; sufficient statistics
read `8·4^layer`. These converge at the deepest layer, where each node has one
leaf and `S` simply *is* the leaf counts.

| max_mer | legacy | sufficient stats | speedup |
|---|---|---|---|
| 5 | 0.220 ms | 0.194 ms | 1.1× |
| 7 | 1.139 ms | 0.533 ms | 2.1× |
| 9 | 10.602 ms | 3.846 ms | 2.8× |

**Memory.** Traces are memory-mapped `.npy` files, written back and evicted every
50 samples. Whole 8-mer fit, 2000 samples (`benchmarks/bench_memory.py`):

| storage | peak RSS | on disk | elapsed |
|---|---|---|---|
| in RAM | 3.45 GB | — | 41.0 s |
| mmap + page release | **0.49 GB** | 3.41 GB | 50.3 s |

The page release is the load-bearing part. Memory-mapping *alone* only got peak
RSS to 2.92 GB, because written pages stay resident until the kernel has a reason
to reclaim them and on an unloaded node it never does. `MADV_DONTNEED` after
writeback drops them outright. Costs ~20% runtime.

## Calibration

Recovery tests check where a posterior *sits*. This checks how *wide* it is --
a sampler can recover means perfectly while reporting intervals twice too
narrow. Simulating from the model's own generative process
(`validation/check_calibration.py`, 6 replicates, k=5):

| layer | coverage vs cascade | coverage vs pooled | pooled/cascade |
|---|---|---|---|
| 0 | 0.194 | **1.000** | 1.1215 |
| 1 | 0.139 | **0.979** | 1.0725 |
| 2 | 0.318 | **0.967** | 1.0518 |
| 3 | 0.670 | **0.977** | 1.0279 |
| 4 | 0.966 | **0.966** | 1.0000 |

**The sampler is calibrated** — 95% intervals contain the truth ~95% of the time,
against the estimand the model actually fits.

That qualifier is load-bearing, and it exposes a real property of Baymer's
formulation. An internal node's likelihood is a multinomial on its *pooled*
descendant counts, so the parameter it identifies is the **arithmetic mean of its
descendants' rates** — not the multiplicative cascade value sitting at that node.
The two differ because `phi` is Gaussian on the log scale, so `E[exp(phi)] > 1`
by Jensen and every layer of descent inflates the aggregate. Predicted inflation
`alpha*exp(sigma^2/2) + (1-alpha)*exp((c*sigma)^2/2)` per layer matches
observation to ~1%:

| layers of descent | predicted | observed |
|---|---|---|
| 1 | 1.0277 | 1.0279 |
| 2 | 1.0561 | 1.0518 |
| 4 | 1.1153 | 1.1215 |

So a shallow-layer rate is ~12% above the cascade value at k=5, and reading it as
a cascade value conflates two different quantities. This is inherent to the
formulation, in legacy as much as here.


## Corrections to legacy

Nine defects in `run_baymer.py`, all fixed and pinned by tests:

| # | Defect |
|---|---|
| a | `sample_new_sigmas` called with 14 args, defined with 13 → `TypeError` whenever sigma is estimated |
| b | Layer-0 out-of-bounds write into a length-3 acceptance array under `@njit` |
| c | Alpha's exact Beta-Gibbs draw filtered through an MH ratio, wrongly rejecting ~21% of valid draws |
| d | Truncated phi proposals broke detailed balance; invalid parent draws silently kept the old parent |
| e | `zero_contexts_bool` reset per parent rather than per child |
| f | `accepted_array[i + sub_index]` index error plus a `prange` write race |
| g | Dead `phi_alpha = 0.5` overwriting the adaptive value |
| h | Results depended on `-t` thread count despite a fixed seed |
| i | `set_sigma: 0` read as "do not fix", which then triggered (a) |

Plus, outside the sampler: the EVEN/ODD parity swap between numerator and
denominator, and `alternation_pattern: "left"` being unusable because
`generate_count_json.py` never forwarded the flag.

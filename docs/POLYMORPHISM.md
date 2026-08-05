# Polymorphism input (FASTA + VCF)

Ports `context_counter.py` and `mutation_counter.py`. Reproduces legacy's own
totals on the tutorial data exactly — 799,381 context occurrences and 12,000
variants kept.

### The EVEN/ODD parity mismatch

Legacy tags every base and every variant by `position % 2` so the data can be
split into two interleaved, size-matched halves for cross-validation. The halves
are mislabelled relative to each other:

- `context_counter.py` computes `even_odd_index = fasta_pos % 2` and writes index
  0 to a column named `.odd_bp` — so `.odd_bp` holds `pos % 2 == 0`.
- `generate_count_json.py` selects `even_odd_bool == 0` for `"EVEN"` but reads
  the `.even_bp` *column* for the denominator, which is the other parity class.

So every `ODD`/`EVEN` run draws its numerator from one parity class and its
denominator from the other. `ALL` sums both and is unaffected. Both the tutorial
and the production species configs run ODD/EVEN.

Reproducing both wirings on the tutorial data:

| | aggregate rate | error | per-context median | 95th pct | max |
|---|---|---|---|---|---|
| correct | 0.012862 | — | — | — | — |
| legacy (swapped) | 0.012858 | 0.03% | **3.3%** | **15%** | **39%** |

The aggregate barely moves, because the two parity classes hold almost identical
*total* opportunity. But individual contexts split unevenly between parities, and
per-context rates are exactly what the model estimates — so the damage is real
where it counts. `build_count_table` now selects the parity class from numerator
and denominator together, and raises if the two disagree.

Two smaller fixes: gzipped FASTA works in both counters (legacy accepted it in
one and not the other, which is why `testingname.mutation_counter.py` exists),
and `--keep-multiallelic` retains multiallelic sites instead of discarding them
at `len(REF) + len(ALT) > 2` — they are the clearest signature of recurrent
mutation and the `recurrent` likelihood wants them.

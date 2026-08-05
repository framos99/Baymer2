#!/bin/bash
# Whole-fit wall clock and peak memory, legacy Baymer versus baymer2.
#
# The other benchmarks here time individual kernels, which cannot answer
# "how much faster is a run". This one drives both implementations end to
# end on identical inputs with identical iteration counts.
#
# Two cases, because the answer depends strongly on depth:
#
#   k=5  the real tutorial data (AFR, cpg_islands), 8000 iterations
#   k=9  synthetic counts, 131072 contexts, 10% unobserved, 400 iterations
#
# 10% unobserved is the point of the k=9 case. Legacy tests membership of
# an index list with `if i in zero_contexts`, which numba lowers to a
# linear scan inside the per-allele loop, so its cost scales with the
# number of empty contexts -- and deep k-mers on any realistic callable
# region have many.
#
# Measured on one cluster node, Python 3.12, numba 0.61, 8 threads:
#
#   case  legacy wall  baymer2 wall  speedup   legacy RSS  baymer2 RSS
#   k=5      31.09 s        8.04 s     3.9x       253 MB       182 MB
#   k=9     841.76 s       30.25 s    27.8x       489 MB       371 MB
#
# Needs a checkout of the original Baymer to time against; point at it
# with $BAYMER2_LEGACY_ROOT. Override the interpreter with $PY.
#
# Usage:  BAYMER2_LEGACY_ROOT=/path/to/Baymer \
#             bash benchmarks/bench_end_to_end.sh [workdir]

set -euo pipefail

WORK="${1:-/tmp/baymer2_bench}"
PY="${PY:-python}"
LEGACY="${BAYMER2_LEGACY_ROOT:?set BAYMER2_LEGACY_ROOT to a checkout of https://github.com/bvoightlab/Baymer}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-8}"
mkdir -p "$WORK"/{k5,k9}/legacy_out

echo "workdir: $WORK   threads: $NUMBA_NUM_THREADS"
echo

# ---------------------------------------------------------------- k = 5
TUT="$LEGACY/tutorial_data/tutorial_outs/tutorial_jsons/AFR"
"$PY" - "$WORK/k5" "$TUT" <<'PYEOF'
import sys, yaml
work, tut = sys.argv[1], sys.argv[2]
yaml.safe_dump(
    {"AFR": {"cpg_islands": {"5mer": {"ALL": f"{tut}/5mer.ALL.AFR.cpg_islands.folded_count_dict.json"}}}},
    open(f"{work}/pointer.yaml", "w"))
p = {"max_mer": 5, "pop": "AFR", "feature": "cpg_islands", "c": 0.04472136,
     "set_sigma": 0.8540486, "posterior_dir": f"{work}/legacy_out/",
     "alternation_pattern": "right", "random_seeds": [14], "dataset": "ALL"}
for l in range(5):
    p[l] = {"iteration_burnin": [8000, 4000], "thinning_parameter": 5,
            "alpha": 0.8, "slab_sigma": 0.2,
            "proposal_sigmas": {"alpha_sigma": "beta", "sigma_sigma": 0.02,
                                "phi_sigma": 0.005, "indicator_sampling": "gibbs"}}
yaml.safe_dump(p, open(f"{work}/params.yaml", "w"))
PYEOF

echo "=== k=5, 8000 iterations, real tutorial data ==="
( cd "$LEGACY" && /usr/bin/time -f "LEGACY  wall=%e s  peak_rss=%M kB" \
    "$PY" baymer/run_baymer.py -c "$WORK/k5/pointer.yaml" \
    -p "$WORK/k5/params.yaml" -r 0 -t "$NUMBA_NUM_THREADS" \
    > "$WORK/k5/legacy.log" 2> "$WORK/k5/legacy.time" ) || true
tail -1 "$WORK/k5/legacy.time"

/usr/bin/time -f "BAYMER2 wall=%e s  peak_rss=%M kB" \
  env PYTHONPATH="$HERE/src" "$PY" - "$HERE" <<'PYEOF'
import json, sys
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.backends.numba_mh import LayerConfig
from baymer2.model import ModelConfig, fit
G = f"{sys.argv[1]}/tests/golden/count_pipeline"
tree = ContextTree(max_mer=5, alternation="right")
d = {k: json.load(open(f"{G}/{k}mer.ALL.AFR.cpg_islands.folded_count_dict.json"))
     for k in range(1, 6)}
fit(CountTable.from_count_dict_set(d, tree, name="AFR"),
    ModelConfig(max_mer=5, seed=14, default_layer=LayerConfig(
        n_iterations=8000, burnin=4000, thin=5, set_sigma=0.8540486,
        c=0.04472136, phi_proposal_sd=0.005, alpha_init=0.8)))
PYEOF
echo

# ---------------------------------------------------------------- k = 9
echo "=== k=9, 400 iterations, 131072 contexts, 10% unobserved ==="
PYTHONPATH="$HERE/src" "$PY" - "$WORK/k9" <<'PYEOF'
import json, sys, numpy as np, yaml
from baymer2.contexts import ContextTree, BASE_INDEX, alt_bases
work = sys.argv[1]
tree = ContextTree(max_mer=9, alternation="right")
ctxs = tree.contexts(8); fi = tree.focal_index(9)
rng = np.random.default_rng(0)
depth = rng.poisson(40.0, size=len(ctxs))
depth[rng.random(len(ctxs)) < 0.10] = 0        # 10% unobserved
cd = {}
for i, c in enumerate(ctxs):
    ref = c[fi]; n = int(depth[i])
    alts = rng.binomial(n, 1e-3, size=3) if n else np.zeros(3, int)
    counts = [0, 0, 0, 0]
    counts[BASE_INDEX[ref]] = int(n - alts.sum())
    for s, b in enumerate(alt_bases(ref)):
        counts[BASE_INDEX[b]] = int(alts[s])
    cd[c] = counts + [n, BASE_INDEX[ref], fi]
json.dump(cd, open(f"{work}/9mer.ALL.SIM.bench.folded_count_dict.json", "w"))
yaml.safe_dump({"SIM": {"bench": {"9mer": {"ALL": f"{work}/9mer.ALL.SIM.bench.folded_count_dict.json"}}}},
               open(f"{work}/pointer.yaml", "w"))
p = {"max_mer": 9, "pop": "SIM", "feature": "bench", "c": 0.04472136,
     "set_sigma": 0.8540486, "posterior_dir": f"{work}/legacy_out/",
     "alternation_pattern": "right", "random_seeds": [14], "dataset": "ALL"}
for l in range(9):
    p[l] = {"iteration_burnin": [400, 200], "thinning_parameter": 10,
            "alpha": 0.8, "slab_sigma": 0.2,
            "proposal_sigmas": {"alpha_sigma": "beta", "sigma_sigma": 0.02,
                                "phi_sigma": 0.005, "indicator_sampling": "gibbs"}}
yaml.safe_dump(p, open(f"{work}/params.yaml", "w"))
PYEOF

( cd "$LEGACY" && /usr/bin/time -f "LEGACY  wall=%e s  peak_rss=%M kB" \
    "$PY" baymer/run_baymer.py -c "$WORK/k9/pointer.yaml" \
    -p "$WORK/k9/params.yaml" -r 0 -t "$NUMBA_NUM_THREADS" \
    > "$WORK/k9/legacy.log" 2> "$WORK/k9/legacy.time" ) || true
tail -1 "$WORK/k9/legacy.time"

/usr/bin/time -f "BAYMER2 wall=%e s  peak_rss=%M kB" \
  env PYTHONPATH="$HERE/src" "$PY" - "$WORK/k9" <<'PYEOF'
import json, sys
from baymer2.contexts import ContextTree
from baymer2.counts import CountTable
from baymer2.backends.numba_mh import LayerConfig
from baymer2.model import ModelConfig, fit
work = sys.argv[1]
tree = ContextTree(max_mer=9, alternation="right")
cd = json.load(open(f"{work}/9mer.ALL.SIM.bench.folded_count_dict.json"))
fit(CountTable.from_count_dict(cd, tree, name="SIM"),
    ModelConfig(max_mer=9, seed=14, default_layer=LayerConfig(
        n_iterations=400, burnin=200, thin=10, set_sigma=0.8540486,
        c=0.04472136, phi_proposal_sd=0.005, alpha_init=0.8)),
    trace_dir=f"{work}/b2_traces")
PYEOF

#!/usr/bin/env bash
# The hybrid-vs-gpu-only benchmark on one GPU node, as one SLURM job: for each
# dataset, first a sweep of the hybrid's --gpu-min-rows (no depth limit, CART
# and C4.5 unpruned), then the depth runs, hybrid at the sweep's fastest value
# and GPU only, each through job.sh (the same GPU and threads, every CPU of the node).
# Submit it from the repository root:
#
#   sbatch -p hopper -w hopper --exclusive -t 8:00:00 -o bench/results/hybrid-vs-gpu-only/hopper.log \
#       --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty)" \
#       bench/cluster/hybrid_vs_gpu_only.sh hopper
#
# Results: bench/results/$DIR/{cuda,gpuonly}-<node>-<dataset>/ (the depth runs),
# $DIR/sweep/cuda-<node>-<dataset>-mr<N>/ (the sweep, "default" = ./tree's own),
# $DIR/sweep/<node>-<dataset>.json (the value taken). The value taken has the
# smallest mean, over CART and C4.5, of its time / that algorithm's fastest.
# For the datasets in PROFILE_DATASETS (default mnist8m) the depth runs are then
# made once more with ./tree --profile, into $DIR/profile/{cuda,gpuonly}-<node>-<dataset>/
# (results.jsonl + profiles/*.json; chart: bench/hybrid_profile_chart.py).
# Environment (for a quick check of the script): DIR, DATASETS, VALUES, RUNS,
# SWEEP_RUNS, PROTOCOLS, PROFILE_DATASETS ("" for none).

set -uo pipefail
NODE="$1"
cd "$SLURM_SUBMIT_DIR"
DIR="${DIR:-hybrid-vs-gpu-only}"
DATASETS="${DATASETS:-higgs mnist8m sorel_4m}"
VALUES="${VALUES:-default 256 512 1024 2048 4096 8192 16384 32768}"
RUNS="${RUNS:-3}" SWEEP_RUNS="${SWEEP_RUNS:-3}"
PROTOCOLS="${PROTOCOLS:-cart_depth5,cart_depth10,cart_depth15,cart_depth20,cart_depth25,cart_depth30,cart_full_nodepth,c45_depth5_unpruned,c45_depth10_unpruned,c45_depth15_unpruned,c45_depth20_unpruned,c45_depth25_unpruned,c45_depth30_unpruned,c45_unpruned}"
PROFILE_DATASETS="${PROFILE_DATASETS-mnist8m}"
SWEEP_PROTOCOLS=cart_full_nodepth,c45_unpruned
PY=bench/.venv/bin/python
failed=()
run() {  # <dir> <name> <dataset> <impls> <runs> <protocols> [RUN_EXTRA]
  mkdir -p "bench/results/$1/$2"
  echo "$(date '+%F %T')  $1/$2${7:+ ($7)}"
  RUN_EXTRA="${7:-}" bench/cluster/job.sh "$1" "$2" "$3" "$4" "$5" "$6" all 3600 \
    > "bench/results/$1/$2/run.log" 2>&1 < /dev/null ||
    { echo "  failed (exit $?): bench/results/$1/$2/run.log"; failed+=("$2"); }
}
for dataset in $DATASETS; do
  for value in $VALUES; do
    extra=; [ "$value" = default ] || extra="--gpu-min-rows $value"
    run "$DIR/sweep" "cuda-$NODE-$dataset-mr$value" "$dataset" tree_cuda "$SWEEP_RUNS" "$SWEEP_PROTOCOLS" "$extra"
  done
  best="$("$PY" - "bench/results/$DIR/sweep" "$NODE" "$dataset" $VALUES <<'EOF'
import json, statistics, sys
from pathlib import Path
folder, node, dataset, values = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
times = {}  # value -> {protocol: mean training seconds}
for value in values:
    results = folder / f"cuda-{node}-{dataset}-mr{value}" / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines()] if results.exists() else []
    by = {}
    for row in rows:
        if row.get("kind") == "run" and row.get("status") == "ok":
            by.setdefault(row["protocol"], []).append(row["train_seconds"])
    if len(by) == 2:
        times[value] = {p: statistics.mean(v) for p, v in by.items()}
fastest = {p: min(t[p] for t in times.values()) for p in next(iter(times.values()))}
score = {v: statistics.mean(t[p] / fastest[p] for p in t) for v, t in times.items()}
numeric = [v for v in score if v != "default"]
best = min(numeric, key=score.get)
(folder / f"{node}-{dataset}.json").write_text(json.dumps(
    {"min_rows": int(best), "times": times, "score": score}, indent=1) + "\n")
print(best)
EOF
)" || { echo "  no sweep result for $dataset, skipping its depth runs"; failed+=("$dataset"); continue; }
  echo "$(date '+%F %T')  $dataset: --gpu-min-rows $best"
  run "$DIR" "cuda-$NODE-$dataset" "$dataset" tree_cuda "$RUNS" "$PROTOCOLS" "--gpu-min-rows $best"
  run "$DIR" "gpuonly-$NODE-$dataset" "$dataset" tree_gpu_only "$RUNS" "$PROTOCOLS"
  if [[ " $PROFILE_DATASETS " == *" $dataset "* ]]; then
    run "$DIR/profile" "cuda-$NODE-$dataset" "$dataset" tree_cuda "$RUNS" "$PROTOCOLS" "--gpu-min-rows $best --profile"
    run "$DIR/profile" "gpuonly-$NODE-$dataset" "$dataset" tree_gpu_only "$RUNS" "$PROTOCOLS" "--profile"
  fi
done
echo "$(date '+%F %T')  done${failed[*]:+, failed: ${failed[*]}}"
[ ${#failed[@]} -eq 0 ]

#!/usr/bin/env bash
# The best --handoff-rows for each dataset on one GPU node, and the --hybrid
# runs of the other benchmarks at it, as one SLURM job. For each dataset: a
# sweep of --handoff-rows with the benchmarks' own protocols (depth <= 10, CART
# and C4.5 unpruned), the value taken by pick_handoff.py, then at that value
# the test-set run (cluster-<dataset>/cuda-<node>-<dataset>, as final.sh's gpu
# kind) and, for PROFILE_DATASETS, the profiling runs (cluster-profiling/ and
# cluster-profiling/plain/, cuda-<node>-<dataset>, as final.sh's profiling kind).
# Submit it from the repository root (the build to measure), e.g.
#
#   TREE_GIT_DESCRIBE="$(git describe --always --abbrev=12)" sbatch -p hopper -w hopper --exclusive \
#       -t 12:00:00 -o bench/results/handoff-sweep/hopper.log bench/cluster/handoff_sweep.sh hopper
#
# Results: bench/results/handoff-sweep/cuda-<node>-<dataset>-mr<N>/ (the sweep,
# "default" = ./tree's own, for reference) and <node>-<dataset>.json (the value taken).
# Environment: DATASETS (test-set runs), PROFILE_DATASETS (profiling runs; a
# dataset may be in either or both), VALUES, RUNS (default 5), SWEEP_RUNS (default 3).
# Chess is not swept: its depth-10 tree has 17 nodes of over 100K rows, so no
# value up to 32768 hands anything off.

set -uo pipefail
NODE="$1"
cd "$SLURM_SUBMIT_DIR"
DIR=handoff-sweep
DATASETS="${DATASETS-susy higgs mnist8m ecbdl_15m sorel_4m}"
PROFILE_DATASETS="${PROFILE_DATASETS-}"
VALUES="${VALUES:-default 256 512 1024 2048 4096 8192 16384 32768}"
RUNS="${RUNS:-5}" SWEEP_RUNS="${SWEEP_RUNS:-3}"
P=cart_depth10,c45_depth10_unpruned
# As batch.sh: an idle node runs its first seconds of work slow.
for _ in $(seq "$(nproc)"); do timeout 30 sh -c 'while :; do :; done' & done
wait
failed=()
run() {  # <dir> <name> <dataset> <runs> <protocols> <timeout> [VAR=value ...]
  local dir="$1" name="$2" dataset="$3" runs="$4" protocols="$5" timeout="$6"
  shift 6
  if [ -f "bench/results/$dir/$name/report.md" ]; then  # report.md is written last
    echo "$(date '+%F %T')  $dir/$name kept"; return
  fi
  mkdir -p "bench/results/$dir/$name"
  echo "$(date '+%F %T')  $dir/$name${*:+ ($*)}"
  env "$@" bench/cluster/job.sh "$dir" "$name" "$dataset" tree_cuda "$runs" "$protocols" all "$timeout" \
    > "bench/results/$dir/$name/run.log" 2>&1 < /dev/null ||
    { echo "  failed (exit $?): bench/results/$dir/$name/run.log"; failed+=("$name"); }
}
dir_of() {
  case "$1" in
    ecbdl*) echo cluster-ecbdl ;;
    sorel*) echo cluster-sorel ;;
    *) echo "cluster-$1" ;;
  esac
}
for dataset in $(echo $DATASETS $PROFILE_DATASETS | tr ' ' '\n' | awk '!seen[$0]++'); do
  for value in $VALUES; do
    extra=; [ "$value" = default ] || extra="--handoff-rows $value"
    run "$DIR" "cuda-$NODE-$dataset-mr$value" "$dataset" "$SWEEP_RUNS" "$P" 14400 RUN_EXTRA="$extra"
  done
  best="$(bench/.venv/bin/python bench/cluster/pick_handoff.py "bench/results/$DIR" "$NODE" "$dataset" $VALUES)" ||
    { echo "  no sweep result for $dataset, skipping its runs"; failed+=("$dataset"); continue; }
  echo "$(date '+%F %T')  $dataset: --handoff-rows $best"
  if [[ " $DATASETS " == *" $dataset "* ]]; then
    run "$(dir_of "$dataset")" "cuda-$NODE-$dataset" "$dataset" "$RUNS" "$P" 14400 \
      ACCURACY=--test-accuracy RUN_EXTRA="--hash-trees --handoff-rows $best"
  fi
  if [[ " $PROFILE_DATASETS " == *" $dataset "* ]]; then
    run cluster-profiling "cuda-$NODE-$dataset" "$dataset" "$RUNS" "$P" 18000 RUN_EXTRA="--profile --handoff-rows $best"
    run cluster-profiling/plain "cuda-$NODE-$dataset" "$dataset" "$RUNS" "$P" 18000 RUN_EXTRA="--handoff-rows $best"
  fi
done
echo "$(date '+%F %T')  done${failed[*]:+, failed: ${failed[*]}}"
[ ${#failed[@]} -eq 0 ]

#!/usr/bin/env bash
# The best --hybrid --handoff-rows for each dataset on one GPU node, as one
# SLURM job: a sweep of the value with the benchmarks' own protocols (CART and
# C4.5 unpruned, depth <= DEPTH, or no limit with DEPTH=none), the value taken
# by pick_handoff.py and stored in bench/tools/handoff/handoff_rows.json, from which
# bench/tools/run.py takes it for every --hybrid run on that node, dataset and depth
# limit. Run it again only when a dataset, the hardware or the hybrid's code
# changes. Submit it from the repository root (the build to measure), e.g.
#
#   TREE_GIT_DESCRIBE="$(git describe --always --abbrev=12)" sbatch -p hopper -w hopper --exclusive \
#       -t 12:00:00 -o bench/tools/handoff/hopper.log bench/tools/handoff/handoff_sweep.sh hopper
#
# Results: bench/tools/handoff/depth<DEPTH>/cuda-<node>-<dataset>-mr<N>/
# ("default" = ./tree's own, for reference) and <node>-<dataset>.json.
# Environment: DATASETS, DEPTH (default 10; "none": no limit), VALUES,
# SWEEP_RUNS (default 5). A value whose folder has a report.md is kept, so a
# cut-short sweep is finished by resubmitting. Chess is not swept: its
# depth-10 tree has no node small enough to hand off.

set -uo pipefail
NODE="$1"
cd "$SLURM_SUBMIT_DIR"
DATASETS="${DATASETS-susy higgs ecbdl_15m mnist8m sorel_4m}"
DEPTH="${DEPTH:-10}"
VALUES="${VALUES:-default 256 512 1024 2048 4096 8192 16384 32768}"
SWEEP_RUNS="${SWEEP_RUNS:-5}"
if [ "$DEPTH" = none ]; then KEY=nodepth P=cart_full_nodepth,c45_unpruned
else KEY="depth$DEPTH" P="cart_depth$DEPTH,c45_depth${DEPTH}_unpruned"; fi
# job.sh writes to bench/$DIR (run.py --run-id).
DIR="tools/handoff/$KEY" RESULTS="bench/tools/handoff/$KEY"
# As batch.sh: an idle node runs its first seconds of work slow.
for _ in $(seq "$(nproc)"); do timeout 30 sh -c 'while :; do :; done' & done
wait
failed=()
for dataset in $DATASETS; do
  for value in $VALUES; do
    name="cuda-$NODE-$dataset-mr$value"
    if [ -f "$RESULTS/$name/report.md" ]; then  # report.md is written last
      echo "$(date '+%F %T')  $RESULTS/$name kept"; continue
    fi
    mkdir -p "$RESULTS/$name"
    echo "$(date '+%F %T')  $RESULTS/$name"
    RUN_EXTRA="--handoff-rows $value" bench/tools/cluster/job.sh "$DIR" "$name" "$dataset" tree_cuda \
      "$SWEEP_RUNS" "$P" all 14400 > "$RESULTS/$name/run.log" 2>&1 < /dev/null ||
      { echo "  failed (exit $?): $RESULTS/$name/run.log"; failed+=("$name"); }
  done
  best="$(bench/tools/.venv/bin/python bench/tools/handoff/pick_handoff.py "$RESULTS" "$NODE" "$dataset" "$KEY" $VALUES)" ||
    { echo "  no sweep result for $dataset"; failed+=("$dataset"); continue; }
  echo "$(date '+%F %T')  $dataset ($KEY): --handoff-rows $best"
done
echo "$(date '+%F %T')  done${failed[*]:+, failed: ${failed[*]}}"
[ ${#failed[@]} -eq 0 ]

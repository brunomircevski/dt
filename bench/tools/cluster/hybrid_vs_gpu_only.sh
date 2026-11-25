#!/usr/bin/env bash
# The hybrid-vs-gpu-only benchmark on one GPU node, as one SLURM job: for each
# dataset, the depth runs of ./tree --hybrid and ./tree --gpu, each through
# job.sh (the same GPU and threads, every CPU of the node). The hybrid's
# --handoff-rows: the value measured best for the node, dataset and depth limit
# (bench/tools/handoff/handoff_rows.json, by bench/tools/handoff/handoff_sweep.sh; run.py takes it).
# Submit it from the repository root:
#
#   sbatch -p hopper -w hopper --exclusive -t 8:00:00 -o bench/hybrid-vs-gpu-only/hopper.log \
#       --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty)" \
#       bench/tools/cluster/hybrid_vs_gpu_only.sh hopper
#
# Results: bench/$DIR/{cuda,gpuonly}-<node>-<dataset>/ (the depth runs).
# For the datasets in PROFILE_DATASETS (default mnist8m) the depth runs are then
# made once more with ./tree --profile, into $DIR/profile/{cuda,gpuonly}-<node>-<dataset>/
# (results.jsonl + profiles/*.json; chart: bench/tools/hybrid_profile_chart.py).
# $DIR/sweep/ holds the no-limit --handoff-rows sweep of f53347a9 (the
# "nodepth" values in bench/tools/handoff/handoff_rows.json, and ./tree's default). Environment (for a quick check of the script): DIR,
# DATASETS, RUNS, PROTOCOLS, PROFILE_DATASETS ("" for none), KINDS (default
# "cuda gpuonly"; "cuda": the hybrid only). A rerun of some cases goes to a new
# DIR, then into the results with bench/tools/merge_runs.py --replace, e.g.
#   DIR=hybrid-vs-gpu-only-new KINDS=cuda DATASETS=higgs PROTOCOLS=cart_depth30 PROFILE_DATASETS= ...

set -uo pipefail
NODE="$1"
cd "$SLURM_SUBMIT_DIR"
DIR="${DIR:-hybrid-vs-gpu-only}"
DATASETS="${DATASETS:-higgs mnist8m sorel_4m}"
RUNS="${RUNS:-3}"
PROTOCOLS="${PROTOCOLS:-cart_depth5,cart_depth10,cart_depth15,cart_depth20,cart_depth25,cart_depth30,cart_full_nodepth,c45_depth5_unpruned,c45_depth10_unpruned,c45_depth15_unpruned,c45_depth20_unpruned,c45_depth25_unpruned,c45_depth30_unpruned,c45_unpruned}"
PROFILE_DATASETS="${PROFILE_DATASETS-mnist8m}"
KINDS="${KINDS:-cuda gpuonly}"
failed=()
run() {  # <dir> <name> <dataset> <impls> <runs> <protocols> [RUN_EXTRA]
  mkdir -p "bench/$1/$2"
  echo "$(date '+%F %T')  $1/$2${7:+ ($7)}"
  RUN_EXTRA="${7:-}" bench/tools/cluster/job.sh "$1" "$2" "$3" "$4" "$5" "$6" all 3600 \
    > "bench/$1/$2/run.log" 2>&1 < /dev/null ||
    { echo "  failed (exit $?): bench/$1/$2/run.log"; failed+=("$2"); }
}
for dataset in $DATASETS; do
  for kind in $KINDS; do
    case "$kind" in cuda) impl=tree_cuda ;; gpuonly) impl=tree_gpu_only ;; *) echo "unknown kind $kind" >&2; exit 2 ;; esac
    run "$DIR" "$kind-$NODE-$dataset" "$dataset" "$impl" "$RUNS" "$PROTOCOLS"
  done
  if [[ " $PROFILE_DATASETS " == *" $dataset "* ]]; then
    for kind in $KINDS; do
      case "$kind" in cuda) impl=tree_cuda ;; gpuonly) impl=tree_gpu_only ;; esac
      run "$DIR/profile" "$kind-$NODE-$dataset" "$dataset" "$impl" "$RUNS" "$PROTOCOLS" "--profile"
    done
  fi
done
echo "$(date '+%F %T')  done${failed[*]:+, failed: ${failed[*]}}"
[ ${#failed[@]} -eq 0 ]

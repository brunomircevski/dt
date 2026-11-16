#!/usr/bin/env bash
# The hybrid-vs-gpu-only benchmark on one GPU node, as one SLURM job: for each
# dataset, the depth runs of ./tree --hybrid (at ./tree's default
# --handoff-rows, 10 x features between 512 and 16384) and ./tree --gpu, each
# through job.sh (the same GPU and threads, every CPU of the node).
# Submit it from the repository root:
#
#   sbatch -p hopper -w hopper --exclusive -t 8:00:00 -o bench/results/hybrid-vs-gpu-only/hopper.log \
#       --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty)" \
#       bench/cluster/hybrid_vs_gpu_only.sh hopper
#
# Results: bench/results/$DIR/{cuda,gpuonly}-<node>-<dataset>/ (the depth runs).
# For the datasets in PROFILE_DATASETS (default mnist8m) the depth runs are then
# made once more with ./tree --profile, into $DIR/profile/{cuda,gpuonly}-<node>-<dataset>/
# (results.jsonl + profiles/*.json; chart: bench/hybrid_profile_chart.py).
# $DIR/sweep/ holds the --handoff-rows sweep (f53347a9) the default was chosen
# from; it is not rerun. Environment (for a quick check of the script): DIR,
# DATASETS, RUNS, PROTOCOLS, PROFILE_DATASETS ("" for none).

set -uo pipefail
NODE="$1"
cd "$SLURM_SUBMIT_DIR"
DIR="${DIR:-hybrid-vs-gpu-only}"
DATASETS="${DATASETS:-higgs mnist8m sorel_4m}"
RUNS="${RUNS:-3}"
PROTOCOLS="${PROTOCOLS:-cart_depth5,cart_depth10,cart_depth15,cart_depth20,cart_depth25,cart_depth30,cart_full_nodepth,c45_depth5_unpruned,c45_depth10_unpruned,c45_depth15_unpruned,c45_depth20_unpruned,c45_depth25_unpruned,c45_depth30_unpruned,c45_unpruned}"
PROFILE_DATASETS="${PROFILE_DATASETS-mnist8m}"
failed=()
run() {  # <dir> <name> <dataset> <impls> <runs> <protocols> [RUN_EXTRA]
  mkdir -p "bench/results/$1/$2"
  echo "$(date '+%F %T')  $1/$2${7:+ ($7)}"
  RUN_EXTRA="${7:-}" bench/cluster/job.sh "$1" "$2" "$3" "$4" "$5" "$6" all 3600 \
    > "bench/results/$1/$2/run.log" 2>&1 < /dev/null ||
    { echo "  failed (exit $?): bench/results/$1/$2/run.log"; failed+=("$2"); }
}
for dataset in $DATASETS; do
  run "$DIR" "cuda-$NODE-$dataset" "$dataset" tree_cuda "$RUNS" "$PROTOCOLS"
  run "$DIR" "gpuonly-$NODE-$dataset" "$dataset" tree_gpu_only "$RUNS" "$PROTOCOLS"
  if [[ " $PROFILE_DATASETS " == *" $dataset "* ]]; then
    run "$DIR/profile" "cuda-$NODE-$dataset" "$dataset" tree_cuda "$RUNS" "$PROTOCOLS" "--profile"
    run "$DIR/profile" "gpuonly-$NODE-$dataset" "$dataset" tree_gpu_only "$RUNS" "$PROTOCOLS" "--profile"
  fi
done
echo "$(date '+%F %T')  done${failed[*]:+, failed: ${failed[*]}}"
[ ${#failed[@]} -eq 0 ]

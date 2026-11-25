#!/usr/bin/env bash
# Every job of the pruned-accuracy benchmark (2026-11-24): HIGGS, MNIST8M,
# ECBDL'14 15M and SOREL 4M (SUSY ran first, by hand). Run from the repository
# root on the login node. Long CPU work on epyc2 and two dell nodes (dell2,
# dell3: dell-long), short CPU work (under an hour a job) on dell1 and dell4,
# GPU work only on GPU nodes: turing (HIGGS, 11 GB is enough) at once, hopper
# and ampere after the jobs already queued there (AFTER_HOPPER, AFTER_AMPERE).
# A case already done (its cases/*.json) is skipped: rerunning a job resumes it.
set -euo pipefail
cd "$(dirname "$0")/../../.."
# the commit ./tree and ./tree_cpu were built from (build.sh's log)
export TREE_GIT_DESCRIBE="$(sed -n '1s/.*commit //p' bench/tools/cluster/build.log)"
LOGS=bench/cluster-accuracy-pruned/logs
mkdir -p "$LOGS"
J=bench/tools/accuracy/job.sh
sub() {  # sub <name> <sbatch options...> -- <job.sh arguments...>
  local name="$1"; shift
  local opts=()
  while [ "$1" != -- ]; do opts+=("$1"); shift; done; shift
  sbatch -J "acc-$name" -o "$LOGS/%j-$name.out" "${opts[@]}" "$J" "$@"
}
AFTER_HOPPER="${AFTER_HOPPER:-}" AFTER_AMPERE="${AFTER_AMPERE:-}"
dep() { [ -n "$1" ] && echo "--dependency=afterany:$1" || true; }

# long CPU: J48 (one fit, both trees), scikit-learn, rpart
sub mnist8m-long -p epyc -w epyc2 -c 16 -t 7-00:00:00 -- mnist8m cpu --tools j48,sklearn --mem-gb 330 --jobs 8
sub higgs-long -p epyc -w epyc2 -c 16 -t 2-00:00:00 -- higgs cpu --tools j48,rpart,sklearn --mem-gb 90 --jobs 12
sub ecbdl-long -p dell-long -w dell2 -c 24 -t 7-00:00:00 -- ecbdl_15m cpu --tools j48,sklearn --mem-gb 460 --jobs 8
sub sorel-long -p dell-long -w dell3 -c 24 -t 7-00:00:00 -- sorel_4m cpu --tools j48,sklearn --mem-gb 460 --jobs 8
# short CPU: ./tree (multicore, serial), YaDT
sub higgs-short -p dell -w dell4 -c 24 -t 1:00:00 -- higgs cpu --tools tree,yadt --mem-gb 200 --jobs 12
sub mnist8m-short -p dell -w dell4 -c 24 -t 1:00:00 -- mnist8m cpu --tools tree,yadt --mem-gb 280 --jobs 6
sub ecbdl-short -p dell -w dell1 -c 24 -t 1:00:00 -- ecbdl_15m cpu --tools tree --mem-gb 350 --jobs 4
sub sorel-short -p dell -w dell1 -c 24 -t 1:00:00 -- sorel_4m cpu --tools tree --mem-gb 350 --jobs 4
# GPU: cuML, XGBoost, ./tree --gpu and --hybrid; one GPU per job
sub higgs-gpu -p turing -c 8 --oversubscribe -t 1:00:00 -- higgs gpu:0 --jobs 2
sub mnist8m-gpu -p hopper -c 8 --oversubscribe -t 12:00:00 $(dep "$AFTER_HOPPER") -- mnist8m gpu:0 --jobs 1 --mem-gb 400
sub ecbdl-gpu-a -p hopper -c 8 --oversubscribe -t 12:00:00 $(dep "$AFTER_HOPPER") -- ecbdl_15m gpu:1 --tools cuml,tree --jobs 1
sub ecbdl-gpu-b -p hopper -c 8 --oversubscribe -t 12:00:00 $(dep "$AFTER_HOPPER") -- ecbdl_15m gpu:2 --tools xgboost --jobs 1
sub sorel-gpu-a -p hopper -c 8 --oversubscribe -t 12:00:00 $(dep "$AFTER_HOPPER") -- sorel_4m gpu:3 --tools cuml,tree --jobs 1
sub sorel-gpu-b -p ampere -c 8 --oversubscribe -t 12:00:00 $(dep "$AFTER_AMPERE") -- sorel_4m gpu:0 --tools xgboost --jobs 1

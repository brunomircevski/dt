#!/usr/bin/env bash
# The final benchmark (the paper's data, 2026-11): every result in
# bench/results/cluster-* (except cluster-profiling, hybrid-vs-gpu-only and the
# power benchmark) from one build, in one go. Run it on the mordor3 login node
# in a clean clone at the commit to measure (~/dt-final), after build.sh:
#
#   bench/cluster/final.sh -t      # the quick check: one run of each kind of job, into cluster-test/
#   bench/cluster/final.sh -n      # write the plan files only (bench/cluster/plans/final-*.txt)
#   bench/cluster/final.sh         # write them and submit
#
# One chain of SLURM jobs per node (batch.sh, one plan file each, run one after
# the other with afterany, so a failed or killed job does not stop the next),
# the most important runs first, so the paper's core numbers come in first:
#
#   hopper, ampere  CUDA hybrid and --gpu-only on every dataset that fits in
#                   80 GB, then cuML and XGBoost (cluster-gpu-vs-others, the
#                   harness in ~/gpu-dt-bench, copied from its scripts/)
#   epyc1           parallel (48 threads), thread scaling, --serial, then the
#                   shorter --simple runs
#   epyc2           (the same hardware as epyc1) scikit-learn and Weka J48
#                   (cluster-cpu-vs-others), then the long --simple runs
#   dell1           parallel, thread scaling, --serial, --simple (dell-long:
#                   14 days; dell2-4 have half its memory channels)
#
# Parallel runs use both sockets (every CPU); one-thread runs the node's first
# CPU, memory first-touch; CUDA runs one GPU and every CPU, its socket's first
# (job.sh). Runs per case: 10, and 3 where one process takes over an hour
# (--simple on full ECBDL'14 and SOREL-20M, J48 on MNIST8M). Depth <= 10, no
# pruning, CART and C4.5, test accuracy on each dataset's test set.

set -euo pipefail
TEST= SUBMIT=1
while getopts "tnh" option; do
  case "$option" in
    t) TEST=1 ;;
    n) SUBMIT= ;;
    *) sed -n '2,30p' "$0"; exit 2 ;;
  esac
done
cd "$(dirname "$0")/../.."  # repository root: the jobs start here (SLURM_SUBMIT_DIR)
for f in tree tree_cpu bench/.tools/rusage bench/.venv/bin/python; do
  [ -e "$f" ] || { echo "$f missing: build first (bench/cluster/README.md)" >&2; exit 1; }
done
git diff --quiet HEAD -- src bench Makefile ||
  { echo "uncommitted changes in src/, bench/ or Makefile: measure a commit" >&2; exit 1; }
export TREE_GIT_DESCRIBE="$(git describe --always --dirty --abbrev=12)"
PLANS=bench/cluster/plans
mkdir -p "$PLANS"
P=cart_depth10,c45_depth10_unpruned
GPU_DATA="susy higgs mnist8m ecbdl_15m sorel_4m chess_1m chess_10m chess_100m chess_1b"
CPU_DATA="susy higgs mnist8m ecbdl_15m ecbdl sorel_4m sorel chess_1m chess_10m chess_100m chess_1b"
GPULIBS_DATA="higgs mnist8m"
EXTRA=ACCURACY=--test-accuracy,RUN_EXTRA=--hash-trees  # job.sh: test accuracy, every tree hashed
OUT=  # results directory: cluster-<dataset>, or cluster-test with -t
if [ -n "$TEST" ]; then
  GPU_DATA=susy CPU_DATA=susy GPULIBS_DATA=mnist8m OUT=cluster-test
fi

dir_of() {
  [ -z "$OUT" ] || { echo "$OUT"; return; }
  case "$1" in
    ecbdl*) echo cluster-ecbdl ;;
    sorel*) echo cluster-sorel ;;
    chess_*) echo cluster-chess ;;
    *) echo "cluster-$1" ;;
  esac
}

# Runs per case: 3 where one process takes over an hour (2026-10 runs on
# ec77727: --simple 1.7-2 h on full ECBDL'14 and 3.6-3.9 h on SOREL-20M; J48
# took 6.7 h on ECBDL'14 15M, so hours on MNIST8M), else 10.
runs() {
  [ -z "$TEST" ] || { echo 1; return; }
  case "$1-$2" in
    simple-ecbdl|simple-sorel|j48-mnist8m) echo 3 ;;
    *) echo 10 ;;
  esac
}

# plan <file> <backend> <node> <dataset>: one line of a batch.sh plan (the
# first field, a time limit, is not used: the batch job has its own).
plan() {
  local file="$PLANS/final-$1.txt" backend="$2" node="$3" dataset="$4" impls threads timeout=14400
  case "$backend" in
    cuda) impls=tree_cuda threads=all ;;
    gpuonly) impls=tree_gpu_only threads=all ;;
    cpu) impls=tree threads=all ;;
    serial) impls=tree threads=1 ;;
    simple) impls=tree_simple threads=1 timeout=36000 ;;
  esac
  echo "- $(dir_of "$dataset") $backend-$node-$dataset $dataset $impls $(runs "$backend" "$dataset") $P $threads $timeout $EXTRA" >> "$file"
}
# scaling <file> <node> <dataset>: 1 (--serial), 2, 4, then every 4 threads to 48
scaling() {
  echo "- ${OUT:-cluster-cpu-scaling} cpu-$2-$3 $3 tree $(runs scaling "$3") $P scaling 14400 RUN_EXTRA=--hash-trees" \
    >> "$PLANS/final-$1.txt"
}
# vs <file> <dataset> <tool> <algorithm>: scikit-learn (CART) or Weka J48 (C4.5), one thread
vs() {
  local protocol=cart_depth10; [ "$4" = cart ] || protocol=c45_depth10_unpruned
  echo "- ${OUT:-cluster-cpu-vs-others} vs-epyc-$2-$3-$4 $2 $3 $(runs "$3" "$2") $protocol 1 86400 $EXTRA" \
    >> "$PLANS/final-$1.txt"
}

rm -f "$PLANS"/final-*.txt
for gpu in hopper ampere; do
  for d in $GPU_DATA; do plan "$gpu" cuda "$gpu" "$d"; plan "$gpu" gpuonly "$gpu" "$d"; done
done
for node in epyc dell; do
  for d in $CPU_DATA; do plan "$node-1" cpu "$node" "$d"; done
  for d in higgs mnist8m; do [ -n "$TEST" ] && [ "$d" = mnist8m ] || scaling "$node-1" "$node" "$d"; done
  for d in $CPU_DATA; do plan "$node-2" serial "$node" "$d"; done
done
# --simple: the short ones on epyc1 and dell1, the long ones (hours a run) on
# epyc2 after the libraries, and on dell1 last.
for d in $CPU_DATA; do
  case "$d" in
    ecbdl|sorel|sorel_4m) plan epyc2-2 simple epyc "$d"; plan dell-4 simple dell "$d" ;;
    *) plan epyc-3 simple epyc "$d"; plan dell-3 simple dell "$d" ;;
  esac
done
if [ -n "$TEST" ]; then
  vs epyc2-1 susy sklearn cart; vs epyc2-1 covertype j48 c45
else
  for d in higgs mnist8m; do vs epyc2-1 "$d" sklearn cart; vs epyc2-1 "$d" j48 c45; done
fi

# Never mix two runs in one directory: run.py appends to results.jsonl.
clash=$(cut -d' ' -f2,3 "$PLANS"/final-*.txt | tr ' ' / | while read -r run; do
  [ ! -e "bench/results/$run/results.jsonl" ] || echo "bench/results/$run"; done)
[ -z "$clash" ] || { echo "already has results (move it away first):"; echo "$clash"; exit 1; } >&2
wc -l "$PLANS"/final-*.txt
[ -n "$SUBMIT" ] || exit 0

# chain <partition> <node> <time limit> <plan>...: the plans one job each, in order
chain() {
  local partition="$1" node="$2" limit="$3" after= id
  shift 3
  for p in "$@"; do
    [ -s "$PLANS/final-$p.txt" ] || continue
    id=$(sbatch --parsable -J "dt-final-$p" -p "$partition" -w "$node" --exclusive -t "$limit" \
      ${after:+--dependency=afterany:$after} -o "$PLANS/final-$p.log" \
      bench/cluster/batch.sh "$PLANS/final-$p.txt")
    echo "final-$p: job $id on $node${after:+ after $after}" >&2
    after=$id
  done
  echo "$after"
}
limit=4-00:00:00; [ -z "$TEST" ] || limit=04:00:00
hopper=$(chain hopper hopper "$limit" hopper)
ampere=$(chain ampere ampere "$limit" ampere)
chain epyc epyc1 "$limit" epyc-1 epyc-2 epyc-3 > /dev/null
chain epyc epyc2 "$limit" epyc2-1 epyc2-2 > /dev/null
chain dell-long dell1 "$limit" dell-1 dell-2 dell-3 dell-4 > /dev/null

# cuML and XGBoost after ./tree on each GPU node: one tree (10 fits), then each
# library's default forest of 10 trees (10 fits), each in one process after an
# untimed full-size warm-up fit.
G=$HOME/gpu-dt-bench
cp bench/results/cluster-gpu-vs-others/scripts/{gpubench.py,job.sh} "$G/"
results=$G/results-final; [ -z "$TEST" ] || results=$G/results-test
mkdir -p "$results"
export IMPLS="cuml xgboost" DATASETS="$GPULIBS_DATA" REPS=10 DEPTH=10 CUML_WARMUP_ROWS=0 RESULTS=$results
[ -z "$TEST" ] || REPS=1
for gpu in hopper ampere; do
  after=${!gpu}
  for trees in 1 10; do
    forest=identical; [ "$trees" = 1 ] || forest=default
    after=$(cd "$G" && TREES=$trees FOREST=$forest sbatch --parsable -J "gdb-final-$gpu" -p "$gpu" -w "$gpu" \
      --exclusive -t 1-00:00:00 ${after:+--dependency=afterany:$after} -o "$results/$gpu-trees$trees.log" \
      job.sh "$gpu")
    echo "cuML and XGBoost, $trees tree(s), $forest: job $after on $gpu"
  done
done
squeue -a -u "$USER" -o "%.8i %.10P %.22j %.2t %.10M %.12l %R"

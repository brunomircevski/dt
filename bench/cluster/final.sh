#!/usr/bin/env bash
# The final benchmark (the paper's data): every result in bench/results from one
# build, in one go. Run it on the mordor3 login node in a clean clone at the
# commit to measure (~/dt-final), after build.sh:
#
#   bench/cluster/final.sh -t      # the quick check: one run of each kind of job, into cluster-test/
#   bench/cluster/final.sh -n      # write the plan files only (bench/cluster/plans/final-*.txt)
#   bench/cluster/final.sh         # write them and submit
#   bench/cluster/final.sh -m 5 -k gpu,cpu,scaling,serial,profiling,hybrid
#                                  # 5 runs a case, only those kinds (the 2026-11-15 rerun)
#
# -m RUNS: runs per case for every kind (default: 10, and 3 where one process
#   takes over an hour: --simple on full ECBDL'14 and SOREL-20M, J48 on MNIST8M).
# -k KINDS: comma-separated, default all of
#   gpu        --hybrid and --gpu on hopper and ampere, every dataset that fits in 80 GB
#   cpu        --multicore, 48 threads on both sockets, epyc1 and dell1
#   scaling    the same at 1 (--serial), 2, 4, 8, ..., 48 threads, HIGGS and MNIST8M (cluster-cpu-scaling)
#   serial     --serial, one thread
#   simple     --simple, one thread
#   vs         scikit-learn and Weka J48 on epyc2 (cluster-cpu-vs-others)
#   gpulibs    cuML and XGBoost on hopper and ampere (cluster-gpu-vs-others, ~/gpu-dt-bench)
#   profiling  ./tree --profile, then the same without (plain/), epyc1, dell1, hopper (cluster-profiling)
#   hybrid     hybrid_vs_gpu_only.sh on hopper and ampere (hybrid-vs-gpu-only)
#
# One chain of SLURM jobs per node (batch.sh, one plan file each, and
# hybrid_vs_gpu_only.sh; run one after the other with afterany, so a failed or
# killed job does not stop the next), the most important runs first:
#
#   hopper   gpu, hybrid, profiling, gpulibs      ampere   gpu, hybrid, gpulibs
#   epyc1    cpu, scaling, profiling, serial (MNIST8M, SOREL 4M), simple (short ones)
#   epyc2    vs, serial (the other datasets), simple (ECBDL'14, SOREL, SOREL 4M)
#   dell1    cpu, scaling, profiling, serial (full ECBDL'14 and SOREL: 226 and
#            354 GiB, more than dell2's NUMA node), simple (full SOREL)
#   dell2    serial (ECBDL'14 15M, SOREL 4M), simple (ECBDL'14, SOREL 4M)
#   dell3    serial (MNIST8M, chess, HIGGS, SUSY), simple (the others)
#
# epyc2 is epyc1's hardware. dell2 and dell3 have half dell1's memory channels:
# their one-thread runs are within ~1% of dell1's (bench/cluster/README.md), so
# only one-thread runs go there. Parallel runs use every CPU; one-thread runs
# the node's first CPU, memory first-touch; CUDA runs one GPU and every CPU, its
# socket's first (job.sh). Depth <= 10, no pruning, CART and C4.5, test accuracy
# on each dataset's test set (profiling: training accuracy only).

set -euo pipefail
ALL_KINDS=gpu,cpu,scaling,serial,simple,vs,gpulibs,profiling,hybrid
TEST= SUBMIT=1 RUNS= KINDS=$ALL_KINDS
while getopts "tnm:k:h" option; do
  case "$option" in
    t) TEST=1 ;;
    n) SUBMIT= ;;
    m) RUNS="$OPTARG" ;;
    k) KINDS="$OPTARG" ;;
    *) sed -n '2,44p' "$0"; exit 2 ;;
  esac
done
[ -z "$RUNS" ] || [[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }
for k in ${KINDS//,/ }; do
  [[ ",$ALL_KINDS," == *",$k,"* ]] || { echo "-k: unknown kind $k (kinds: $ALL_KINDS)" >&2; exit 2; }
done
want() { [[ ",$KINDS," == *",$1,"* ]]; }

cd "$(dirname "$0")/../.."  # repository root: the jobs start here (SLURM_SUBMIT_DIR)
for f in tree tree_cpu bench/.tools/rusage bench/.venv/bin/python; do
  [ -e "$f" ] || { echo "$f missing: build first (bench/cluster/README.md)" >&2; exit 1; }
done
# The code must be a commit (the old run folders may be moved out of bench/results).
CODE=(src Makefile bench/*.py bench/adapters bench/cluster bench/datasets.toml bench/results/cluster-gpu-vs-others/scripts)
git diff --quiet HEAD -- "${CODE[@]}" ||
  { echo "uncommitted changes in ${CODE[*]}: measure a commit" >&2; exit 1; }
export TREE_GIT_DESCRIBE="$(git describe --always --abbrev=12)"
PLANS=bench/cluster/plans
mkdir -p "$PLANS"
P=cart_depth10,c45_depth10_unpruned
GPU_DATA="susy higgs mnist8m ecbdl_15m sorel_4m chess_1m chess_10m chess_100m chess_1b"
CPU_DATA="susy higgs mnist8m ecbdl_15m ecbdl sorel_4m sorel chess_1m chess_10m chess_100m chess_1b"
SCALING_DATA="higgs mnist8m"
GPULIBS_DATA="higgs mnist8m"
VS_DATA="higgs mnist8m"
PROFILE_DATA="higgs airline ecbdl_15m chess_10m chess_50m chess_100m chess_500m chess_1b chess_2b"
HYBRID_DATA="higgs mnist8m sorel_4m"
EXTRA=ACCURACY=--test-accuracy,RUN_EXTRA=--hash-trees  # job.sh: test accuracy, every tree hashed
OUT=  # results directory: cluster-<dataset>, or cluster-test with -t
if [ -n "$TEST" ]; then
  GPU_DATA=susy CPU_DATA=susy SCALING_DATA=higgs GPULIBS_DATA=mnist8m VS_DATA=susy PROFILE_DATA=susy
  HYBRID_DATA=susy OUT=cluster-test RUNS=1
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

# runs <kind> <dataset>: -m, else 3 where one process takes over an hour
# (--simple 1.7-2 h on full ECBDL'14 and 3.6-3.9 h on SOREL-20M; J48 5.3 h on
# MNIST8M), else 10.
runs() {
  [ -z "$RUNS" ] || { echo "$RUNS"; return; }
  case "$1-$2" in
    simple-ecbdl|simple-sorel|j48-mnist8m) echo 3 ;;
    *) echo 10 ;;
  esac
}

# line <plan> <dir> <name> <dataset> <impls> <runs> <protocols> <threads> <timeout> [environment]:
# one line of a batch.sh plan (the first field, a time limit, is not used:
# the batch job has its own).
line() { local plan="$1"; shift; echo "- $*" >> "$PLANS/final-$plan.txt"; }
# plan <plan> <backend> <node> <dataset>: a test-set run
plan() {
  local backend="$2" node="$3" dataset="$4" impls threads timeout=14400
  case "$backend" in
    cuda) impls=tree_cuda threads=all ;;
    gpuonly) impls=tree_gpu_only threads=all ;;
    cpu) impls=tree threads=all ;;
    serial) impls=tree threads=1 ;;
    simple) impls=tree_simple threads=1 timeout=36000 ;;
  esac
  line "$1" "$(dir_of "$dataset")" "$backend-$node-$dataset" "$dataset" "$impls" "$(runs "$backend" "$dataset")" \
    "$P" "$threads" "$timeout" "$EXTRA"
}
# scaling <plan> <node> <dataset>: 1 (--serial), 2, 4, then every 4 threads to 48
scaling() {
  line "$1" "${OUT:-cluster-cpu-scaling}" "cpu-$2-$3" "$3" tree "$(runs scaling "$3")" "$P" scaling 14400 \
    RUN_EXTRA=--hash-trees
}
# vs <plan> <dataset> <tool> <algorithm>: scikit-learn (CART) or Weka J48 (C4.5), one thread
vs() {
  local protocol=cart_depth10; [ "$4" = cart ] || protocol=c45_depth10_unpruned
  line "$1" "${OUT:-cluster-cpu-vs-others}" "vs-epyc-$2-$3-$4" "$2" "$3" "$(runs "$3" "$2")" "$protocol" 1 86400 "$EXTRA"
}
# profile <plan> <backend> <node>: every PROFILE_DATA dataset with --profile, then
# all of them again without it (into plain/), to see what profiling costs
profile() {
  local impls=tree d dir="${OUT:+$OUT/}cluster-profiling"
  case "$2" in cuda) impls=tree_cuda ;; gpuonly) impls=tree_gpu_only ;; esac
  for d in $PROFILE_DATA; do
    line "$1" "$dir" "$2-$3-$d" "$d" "$impls" "$(runs profiling "$d")" "$P" all 18000 RUN_EXTRA=--profile
  done
  for d in $PROFILE_DATA; do
    line "$1" "$dir/plain" "$2-$3-$d" "$d" "$impls" "$(runs profiling "$d")" "$P" all 18000
  done
}

rm -f "$PLANS"/final-*.txt
for gpu in hopper ampere; do
  if want gpu; then
    for d in $GPU_DATA; do plan "$gpu" cuda "$gpu" "$d"; plan "$gpu" gpuonly "$gpu" "$d"; done
  fi
done
if want profiling; then profile hopper-2 cuda hopper; profile hopper-2 gpuonly hopper; fi
for node in epyc dell; do
  if want cpu; then for d in $CPU_DATA; do plan "$node-1" cpu "$node" "$d"; done; fi
  if want scaling; then for d in $SCALING_DATA; do scaling "$node-1" "$node" "$d"; done; fi
  if want profiling; then profile "$node-1" cpu "$node"; fi
done
# --serial: the big ones where the memory is, the rest spread over epyc1/epyc2 and dell1-3
if want serial; then
  for d in $CPU_DATA; do
    case "$d" in
      mnist8m|sorel_4m) plan epyc-2 serial epyc "$d" ;;
      *) plan epyc2-2 serial epyc "$d" ;;
    esac
    case "$d" in
      ecbdl|sorel) plan dell-2 serial dell "$d" ;;
      ecbdl_15m|sorel_4m) plan dell2-1 serial dell "$d" ;;
      *) plan dell3-1 serial dell "$d" ;;
    esac
  done
fi
# --simple: the short ones on epyc1 and dell3, the long ones (hours a run) on
# epyc2 after the libraries, on dell2, and full SOREL on dell1 last.
if want simple; then
  for d in $CPU_DATA; do
    case "$d" in
      ecbdl|sorel|sorel_4m) plan epyc2-3 simple epyc "$d" ;;
      *) plan epyc-3 simple epyc "$d" ;;
    esac
    case "$d" in
      sorel) plan dell-3 simple dell "$d" ;;
      ecbdl|ecbdl_15m|sorel_4m) plan dell2-2 simple dell "$d" ;;
      *) plan dell3-2 simple dell "$d" ;;
    esac
  done
fi
if want vs; then
  for d in $VS_DATA; do
    if [ -n "$TEST" ]; then vs epyc2-1 "$d" sklearn cart
    else vs epyc2-1 "$d" sklearn cart; vs epyc2-1 "$d" j48 c45; fi
  done
fi

# hybrid_vs_gpu_only.sh's folders, for the clash check
HYBRID_DIR="${OUT:+$OUT/}hybrid-vs-gpu-only"
hybrid_runs() {
  local gpu d
  for gpu in hopper ampere; do for d in $HYBRID_DATA; do
    echo "$HYBRID_DIR/cuda-$gpu-$d $HYBRID_DIR/gpuonly-$gpu-$d $HYBRID_DIR/profile/cuda-$gpu-$d"
    echo "$HYBRID_DIR/profile/gpuonly-$gpu-$d"
  done; done
}

# Never mix two runs in one directory: run.py appends to results.jsonl.
clash=$({ cat "$PLANS"/final-*.txt 2>/dev/null | cut -d' ' -f2,3 | tr ' ' /
          ! want hybrid || hybrid_runs | tr ' ' '\n'; } | while read -r run; do
  [ ! -e "bench/results/$run/results.jsonl" ] || echo "bench/results/$run"; done)
[ -z "$clash" ] || { echo "already has results (move it away first):"; echo "$clash"; exit 1; } >&2
ls "$PLANS"/final-*.txt > /dev/null 2>&1 && wc -l "$PLANS"/final-*.txt
! want hybrid || echo "hybrid_vs_gpu_only.sh on hopper and ampere: $HYBRID_DATA, $(runs hybrid x) runs a case"
[ -n "$SUBMIT" ] || exit 0

# chain <partition> <node> <step>...: the steps one job each, in order. A step
# is a plan (final-<step>.txt, skipped when empty) or hybrid (hybrid_vs_gpu_only.sh).
limit=4-00:00:00; [ -z "$TEST" ] || limit=04:00:00
chain() {
  local partition="$1" node="$2" after= id step
  shift 2
  for step in "$@"; do
    if [ "$step" = hybrid ]; then
      want hybrid || continue
      mkdir -p "bench/results/$HYBRID_DIR"
      local hybrid_env=(DIR="$HYBRID_DIR" DATASETS="$HYBRID_DATA" RUNS="$(runs hybrid x)")
      [ -z "$TEST" ] || hybrid_env+=(PROTOCOLS=cart_depth10,c45_unpruned PROFILE_DATASETS=susy)
      id=$(env "${hybrid_env[@]}" sbatch --parsable -J "dt-final-hybrid-$node" -p "$partition" -w "$node" \
        --exclusive -t "$limit" ${after:+--dependency=afterany:$after} -o "bench/results/$HYBRID_DIR/$node.log" \
        bench/cluster/hybrid_vs_gpu_only.sh "$node")
    else
      [ -s "$PLANS/final-$step.txt" ] || continue
      id=$(sbatch --parsable -J "dt-final-$step" -p "$partition" -w "$node" --exclusive -t "$limit" \
        ${after:+--dependency=afterany:$after} -o "$PLANS/final-$step.log" \
        bench/cluster/batch.sh "$PLANS/final-$step.txt")
    fi
    echo "$step: job $id on $node${after:+ after $after}" >&2
    after=$id
  done
  echo "$after"
}
hopper=$(chain hopper hopper hopper hybrid hopper-2)
ampere=$(chain ampere ampere ampere hybrid)
chain epyc epyc1 epyc-1 epyc-2 epyc-3 > /dev/null
chain epyc epyc2 epyc2-1 epyc2-2 epyc2-3 > /dev/null
chain dell-long dell1 dell-1 dell-2 dell-3 > /dev/null
chain dell-long dell2 dell2-1 dell2-2 > /dev/null
chain dell-long dell3 dell3-1 dell3-2 > /dev/null
want gpulibs || { squeue -a -u "$USER" -o "%.8i %.10P %.22j %.2t %.10M %.12l %R"; exit 0; }

# cuML and XGBoost after ./tree on each GPU node: one tree (-m fits, default
# 10), then each library's default forest of 10 trees, each in one process
# after an untimed full-size warm-up fit.
G=$HOME/gpu-dt-bench
cp bench/results/cluster-gpu-vs-others/scripts/{gpubench.py,job.sh} "$G/"
results=$G/results-final; [ -z "$TEST" ] || results=$G/results-test
mkdir -p "$results"
export IMPLS="cuml xgboost" DATASETS="$GPULIBS_DATA" REPS="$(runs gpulibs x)" DEPTH=10 CUML_WARMUP_ROWS=0 RESULTS=$results
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

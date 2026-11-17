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
#   bench/cluster/final.sh -N epyc1,epyc2,dell2,dell3,dell4
#                                  # only those nodes' chains (the GPU nodes later)
#   bench/cluster/final.sh -D higgs,susy -k gpu,cpu,scaling,serial,simple,vs,gpulibs,e2e
#                                  # only those datasets' runs (a first part of the rerun)
#
# Every run is one fresh process that trains one tree, no warm-up
# (BENCHMARK-CHANGES.MD, section 1).
# -m RUNS: runs per case for every kind (default: 10, and 3 where one process
#   trains for over an hour: --simple on full ECBDL'14 and SOREL-20M; run.py also
#   stops any case at 3 runs whose first run took over an hour, e.g. J48 and
#   maybe rpart on MNIST8M).
# -N NODES: comma-separated, only these nodes' chains are checked and submitted
#   (default all: hopper ampere epyc1 epyc2 dell2 dell3 dell4).
# -D DATASETS: comma-separated, only these datasets' runs (plan lines, gpulibs,
#   hybrid); a kind left without a dataset is skipped.
# -k KINDS: comma-separated, default all of
#   gpu        --hybrid and --gpu on hopper and ampere, every dataset that fits in 80 GB
#   cpu        --multicore, 48 threads on both sockets, epyc1 and dell4
#   scaling    the same at 1 (--serial), 2, 4, 8, ..., 48 threads, HIGGS and MNIST8M (cluster-cpu-scaling)
#   serial     --serial, one thread
#   simple     --simple, one thread
#   vs         scikit-learn, rpart, YaDT (1 and 48 threads), Weka J48 on epyc2 (cluster-cpu-vs-others):
#              one run.py per dataset with every tool, so their runs interleave
#   gpulibs    cuML and XGBoost on hopper and ampere (cluster-gpu-vs-others, ~/gpu-dt-bench):
#              one tree and a default forest of 10, one gpubench.py process per run, interleaved;
#              and the warm variant: one CART tree of cuML, XGBoost (WARM=1) and ./tree
#              (--warm-up, cluster-gpu-vs-others/warm/, in the hopper and ampere plans)
#   profiling  ./tree --profile, then the same without (plain/), epyc1, dell4, hopper (cluster-profiling)
#   hybrid     hybrid_vs_gpu_only.sh on hopper and ampere (hybrid-vs-gpu-only)
#   e2e        end to end, HIGGS, every tool (cluster-end-to-end): epyc1 (CPU tools) and
#              hopper (./tree's GPU backends, cuML, XGBoost), run.py --no-eval --prefetch
#
# One chain of SLURM jobs per node (batch.sh, one plan file each, and
# hybrid_vs_gpu_only.sh; run one after the other with afterany, so a failed or
# killed job does not stop the next), the most important runs first:
#
#   hopper   gpu, hybrid, profiling, gpulibs, e2e      ampere   gpu, hybrid, gpulibs
#   epyc1    cpu, scaling, profiling, serial (MNIST8M, SOREL 4M), simple (short ones), e2e
#   epyc2    vs, serial (the other datasets), simple (ECBDL'14, SOREL, SOREL 4M)
#   dell4    cpu and scaling, then profiling (dell-512GB: one day a job at most)
#   dell2    serial (ECBDL'14 15M, SOREL 4M), simple (ECBDL'14, SOREL 4M)
#   dell3    serial (MNIST8M, chess, HIGGS, SUSY; then full ECBDL'14 and SOREL),
#            simple (the others; then full SOREL)
#
# epyc2 is epyc1's hardware. dell2-4 are the same (512 GB, 8 of 16 memory
# channels; triad 210-212 GB/s on 2026-11-17). dell1 (1 TB, every channel,
# 379 GB/s) is not used: the paper's dell is a 512 GB node. Parallel runs use every CPU; one-thread runs
# the node's first CPU, memory first-touch; CUDA runs one GPU and every CPU, its
# socket's first (job.sh). Depth <= 10, no pruning, CART and C4.5, test accuracy
# on each dataset's test set (profiling: training accuracy only).

set -euo pipefail
ALL_KINDS=gpu,cpu,scaling,serial,simple,vs,gpulibs,profiling,hybrid,e2e
ALL_NODES=hopper,ampere,epyc1,epyc2,dell2,dell3,dell4
TEST= SUBMIT=1 RUNS= KINDS=$ALL_KINDS NODES=$ALL_NODES ONLY_DATA=
while getopts "tnm:k:N:D:h" option; do
  case "$option" in
    t) TEST=1 ;;
    n) SUBMIT= ;;
    m) RUNS="$OPTARG" ;;
    k) KINDS="$OPTARG" ;;
    N) NODES="$OPTARG" ;;
    D) ONLY_DATA="$OPTARG" ;;
    *) sed -n '2,60p' "$0"; exit 2 ;;
  esac
done
for n in ${NODES//,/ }; do
  [[ ",$ALL_NODES," == *",$n,"* ]] || { echo "-N: unknown node $n (nodes: $ALL_NODES)" >&2; exit 2; }
done
node_wanted() { [[ ",$NODES," == *",$1,"* ]]; }
[ -z "$RUNS" ] || [[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }
for k in ${KINDS//,/ }; do
  [[ ",$ALL_KINDS," == *",$k,"* ]] || { echo "-k: unknown kind $k (kinds: $ALL_KINDS)" >&2; exit 2; }
done
want() { [[ ",$KINDS," == *",$1,"* ]]; }

cd "$(dirname "$0")/../.."  # repository root: the jobs start here (SLURM_SUBMIT_DIR)
REPO=$PWD
needs=(tree_cpu bench/.tools/rusage bench/.venv/bin/python)
{ node_wanted hopper || node_wanted ampere; } && needs+=(tree)  # the CUDA build: only on a GPU node
{ want vs || want e2e; } && needs+=(bench/.tools/yadt/dTcmd bench/.tools/weka.jar bench/.tools/classes/J48Fit.class)
{ want gpulibs || want e2e; } && { node_wanted hopper || node_wanted ampere; } &&
  needs+=("$HOME/gpu-dt-bench/venv/bin/python")
for f in "${needs[@]}"; do
  [ -e "$f" ] || { echo "$f missing: build first (bench/cluster/README.md)" >&2; exit 1; }
done
# The code must be a commit (the old run folders may be moved out of bench/results).
CODE=(src Makefile bench/*.py bench/adapters bench/cluster bench/datasets.toml bench/results/cluster-gpu-vs-others/scripts)
git diff --quiet HEAD -- "${CODE[@]}" ||
  { echo "uncommitted changes in ${CODE[*]}: measure a commit" >&2; exit 1; }
export TREE_GIT_DESCRIBE="$(git describe --always --abbrev=12)"
# The binaries must be built from this code (build.sh writes build/COMMIT).
BUILT="$(cat build/COMMIT 2>/dev/null)"; BUILT="${BUILT%-dirty}"
{ [ -n "$BUILT" ] && git diff --quiet "${BUILT##*-g}" HEAD -- src Makefile; } ||
  { echo "./tree is built from ${BUILT:-an unknown commit}, not this src/: rebuild (bench/cluster/build.sh)" >&2; exit 1; }
PLANS=bench/cluster/plans
mkdir -p "$PLANS"
P=cart_depth10,c45_depth10_unpruned
GPU_DATA="susy higgs mnist8m ecbdl_15m sorel_4m chess_1m chess_10m chess_100m chess_1b"
CPU_DATA="susy higgs mnist8m ecbdl_15m ecbdl sorel_4m sorel chess_1m chess_10m chess_100m chess_1b"
SCALING_DATA="higgs mnist8m"
GPULIBS_DATA="higgs mnist8m"
VS_DATA="higgs mnist8m"
PROFILE_DATA="higgs ecbdl_15m chess_10m chess_50m chess_100m chess_500m chess_1b chess_2b"
HYBRID_DATA="higgs mnist8m sorel_4m"
E2E_DATA=higgs
EXTRA=ACCURACY=--test-accuracy,RUN_EXTRA=--hash-trees  # job.sh: test accuracy, every tree hashed
OUT=  # results directory: cluster-<dataset>, or cluster-test with -t
if [ -n "$TEST" ]; then
  GPU_DATA=susy CPU_DATA=susy SCALING_DATA=higgs GPULIBS_DATA=mnist8m VS_DATA=susy PROFILE_DATA=susy
  HYBRID_DATA=susy E2E_DATA=susy OUT=cluster-test RUNS=1
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
# (--simple 1.7-2 h on full ECBDL'14 and 3.6-3.9 h on SOREL-20M), else 10.
# Within a run.py, a case whose first run took over an hour stops at 3 runs
# (run.py --long-runs; J48 5.3 h on MNIST8M, rpart maybe).
runs() {
  [ -z "$RUNS" ] || { echo "$RUNS"; return; }
  case "$1-$2" in
    simple-ecbdl|simple-sorel) echo 3 ;;
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
# vs <plan> <dataset> <tools>: every library in one run.py, so their runs
# interleave (each repetition a new shuffled order): scikit-learn and rpart
# (CART), Weka J48 and YaDT (C4.5) on one thread, YaDT also on every CPU (48).
vs() {
  line "$1" "${OUT:-cluster-cpu-vs-others}" "vs-epyc-$2" "$2" "$3" "$(runs vs "$2")" \
    "$P" 1,all 86400 "$EXTRA"
}
# e2e <plan> <node> <dataset> <impls> <threads>: the end-to-end runs of one node,
# every tool interleaved (job.sh E2E=1: run.py --no-eval --prefetch --no-baseline)
e2e() {
  line "$1" "${OUT:-cluster-end-to-end}" "e2e-$2-$3" "$3" "$4" "$(runs e2e "$3")" "$P" "$5" 86400 E2E=1
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
# The warm variant of cluster-gpu-vs-others (with gpulibs): ./tree's CART tree
# trained second in one process (--warm-up), as gpubench.py WARM=1.
if want gpulibs; then
  for gpu in hopper ampere; do for d in $GPULIBS_DATA; do for backend in cuda gpuonly; do
    impl=tree_cuda; [ "$backend" = gpuonly ] && impl=tree_gpu_only
    line "$gpu" "${OUT:-cluster-gpu-vs-others/warm}" "$backend-$gpu-$d" "$d" "$impl" "$(runs gpulibs "$d")" \
      cart_depth10 all 14400 "$EXTRA,WARM_UP=1"
  done; done; done
fi
if want profiling; then profile hopper-2 cuda hopper; profile hopper-2 gpuonly hopper; fi
for node in epyc dell; do
  if want cpu; then for d in $CPU_DATA; do plan "$node-1" cpu "$node" "$d"; done; fi
  if want scaling; then for d in $SCALING_DATA; do scaling "$node-1" "$node" "$d"; done; fi
  if want profiling; then profile "$node-1b" cpu "$node"; fi  # its own job: dell4's are one day at most
done
# --serial: the big ones where the memory is, the rest spread over epyc1/epyc2 and dell2-3
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
# epyc2 after the libraries, on dell2, and full SOREL on dell3 last.
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
  for d in $VS_DATA; do vs epyc2-1 "$d" sklearn,rpart,yadt,j48; done
fi
if want e2e; then
  for d in $E2E_DATA; do
    e2e epyc-4 epyc "$d" tree,tree_simple,sklearn,rpart,j48,yadt 1,all
    e2e hopper-4 hopper "$d" tree_cuda,tree_gpu_only,cuml,xgboost all
  done
fi

# hybrid_vs_gpu_only.sh's folders, for the clash check
HYBRID_DIR="${OUT:+$OUT/}hybrid-vs-gpu-only"
hybrid_runs() {
  local gpu d
  for gpu in ${GPU_NODES[@]+"${GPU_NODES[@]}"}; do for d in $HYBRID_DATA; do
    echo "$HYBRID_DIR/cuda-$gpu-$d $HYBRID_DIR/gpuonly-$gpu-$d $HYBRID_DIR/profile/cuda-$gpu-$d"
    echo "$HYBRID_DIR/profile/gpuonly-$gpu-$d"
  done; done
}

# The chain of each node: its steps one job each, in order (a plan file's name
# without final-, or hybrid, gpulibs).
declare -A CHAIN=(
  [hopper]="hopper hybrid hopper-2 gpulibs hopper-4"
  [ampere]="ampere hybrid gpulibs"
  [epyc1]="epyc-1 epyc-1b epyc-2 epyc-3 epyc-4"
  [epyc2]="epyc2-1 epyc2-2 epyc2-3"
  [dell4]="dell-1 dell-1b"
  [dell2]="dell2-1 dell2-2"
  [dell3]="dell3-1 dell3-2 dell-2 dell-3"
)
# -D: only those datasets' plan lines, gpulibs and hybrid runs
if [ -n "$ONLY_DATA" ]; then
  only() { local d out=; for d in $1; do [[ ",$ONLY_DATA," == *",$d,"* ]] && out+="$d "; done; echo "${out% }"; }
  for f in "$PLANS"/final-*.txt; do
    [ -e "$f" ] || continue
    awk -v only=",$ONLY_DATA," 'index(only, "," $4 ",")' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  done
  GPULIBS_DATA=$(only "$GPULIBS_DATA") HYBRID_DATA=$(only "$HYBRID_DATA")
  [ -n "$GPULIBS_DATA" ] || KINDS=$(echo ",$KINDS," | sed 's/,gpulibs,/,/; s/^,//; s/,$//')
  [ -n "$HYBRID_DATA" ] || KINDS=$(echo ",$KINDS," | sed 's/,hybrid,/,/; s/^,//; s/,$//')
fi
# The plan files of the chosen nodes' chains (-N); the others are removed.
chosen=()
for node in ${NODES//,/ }; do for step in ${CHAIN[$node]}; do
  [ -s "$PLANS/final-$step.txt" ] && chosen+=("$PLANS/final-$step.txt")
done; done
for f in "$PLANS"/final-*.txt; do
  [ -e "$f" ] || continue
  [[ " ${chosen[*]} " == *" $f "* ]] || rm "$f"
done
GPU_NODES=(); for gpu in hopper ampere; do node_wanted "$gpu" && GPU_NODES+=("$gpu"); done
G=$HOME/gpu-dt-bench
GPULIBS_RESULTS=$G/results-final; [ -z "$TEST" ] || GPULIBS_RESULTS=$G/results-test

# Never mix two runs in one directory: run.py appends to results.jsonl.
clash=$({ cat "$PLANS"/final-*.txt 2>/dev/null | cut -d' ' -f2,3 | tr ' ' /
          ! want hybrid || [ ${#GPU_NODES[@]} -eq 0 ] || hybrid_runs | tr ' ' '\n'; } | while read -r run; do
  [ ! -e "bench/results/$run/results.jsonl" ] || echo "bench/results/$run"; done)
if want gpulibs; then
  for gpu in ${GPU_NODES[@]+"${GPU_NODES[@]}"}; do for d in $GPULIBS_DATA; do
    [ ! -e "$GPULIBS_RESULTS/$d/$gpu/results.jsonl" ] || clash+=$'\n'"$GPULIBS_RESULTS/$d/$gpu"
  done; done
fi
clash=$(echo "$clash" | sed '/^$/d')
[ -z "$clash" ] || { echo "already has results (move it away first):"; echo "$clash"; exit 1; } >&2
ls "$PLANS"/final-*.txt > /dev/null 2>&1 && wc -l "$PLANS"/final-*.txt
! want hybrid || [ ${#GPU_NODES[@]} -eq 0 ] ||
  echo "hybrid_vs_gpu_only.sh on ${GPU_NODES[*]}: $HYBRID_DATA, $(runs hybrid x) runs a case"
! want gpulibs || [ ${#GPU_NODES[@]} -eq 0 ] ||
  echo "cuML and XGBoost on ${GPU_NODES[*]}: $GPULIBS_DATA, 1 tree and a forest of 10, $(runs gpulibs x) runs a case"
[ -n "$SUBMIT" ] || exit 0

# chain <partition> <node> <step>...: the steps one job each, in order. A step
# is a plan (final-<step>.txt, skipped when empty), hybrid
# (hybrid_vs_gpu_only.sh) or gpulibs (cluster-gpu-vs-others' job.sh in ~/gpu-dt-bench).
limit=4-00:00:00; [ -z "$TEST" ] || limit=04:00:00
chain() {
  local partition="$1" node="$2" after= id step limit="$limit"
  shift 2
  [ "$partition" != dell-512GB ] || [ -n "$TEST" ] || limit=1-00:00:00  # dell4: not in dell-long
  for step in "$@"; do
    if [ "$step" = hybrid ]; then
      want hybrid || continue
      mkdir -p "bench/results/$HYBRID_DIR"
      local hybrid_env=(DIR="$HYBRID_DIR" DATASETS="$HYBRID_DATA" RUNS="$(runs hybrid x)")
      [ -z "$TEST" ] || hybrid_env+=(PROTOCOLS=cart_depth10,c45_unpruned PROFILE_DATASETS=susy)
      id=$(env "${hybrid_env[@]}" sbatch --parsable -J "dt-final-hybrid-$node" -p "$partition" -w "$node" \
        --exclusive -t "$limit" ${after:+--dependency=afterany:$after} -o "bench/results/$HYBRID_DIR/$node.log" \
        bench/cluster/hybrid_vs_gpu_only.sh "$node")
    elif [ "$step" = gpulibs ]; then
      # cuML and XGBoost: one tree and each library's default forest of 10,
      # REPS rounds of one fresh gpubench.py process per case (job.sh).
      want gpulibs || continue
      cp bench/results/cluster-gpu-vs-others/scripts/{gpubench.py,job.sh} "$G/"
      mkdir -p "$GPULIBS_RESULTS"
      id=$(cd "$G" && IMPLS="cuml xgboost" DATASETS="$GPULIBS_DATA" REPS="$(runs gpulibs x)" DEPTH=10 \
        FORESTS="1:identical 10:default" WARM_FORESTS="1:identical" RESULTS="$GPULIBS_RESULTS" RUSAGE="$REPO/bench/.tools/rusage" \
        sbatch --parsable -J "gdb-final-$node" -p "$partition" -w "$node" --exclusive -t "$limit" \
        ${after:+--dependency=afterany:$after} -o "$GPULIBS_RESULTS/$node.log" job.sh "$node")
    else
      [ -s "$PLANS/final-$step.txt" ] || continue
      id=$(sbatch --parsable -J "dt-final-$step" -p "$partition" -w "$node" --exclusive -t "$limit" \
        ${after:+--dependency=afterany:$after} -o "$PLANS/final-$step.log" \
        bench/cluster/batch.sh "$PLANS/final-$step.txt")
    fi
    echo "$step: job $id on $node${after:+ after $after}" >&2
    after=$id
  done
}
declare -A PARTITION=([hopper]=hopper [ampere]=ampere [epyc1]=epyc [epyc2]=epyc
                      [dell2]=dell-long [dell3]=dell-long [dell4]=dell-512GB)
for node in ${NODES//,/ }; do
  # shellcheck disable=SC2086  # the steps
  chain "${PARTITION[$node]}" "$node" ${CHAIN[$node]}
done
squeue -a -u "$USER" -o "%.8i %.10P %.22j %.2t %.10M %.12l %R"

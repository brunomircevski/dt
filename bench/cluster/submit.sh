#!/usr/bin/env bash
# One benchmark run per job on mordor3: one SLURM job per node, one whole node
# each, held until all are queued and then released together (run it on the
# login node; it only calls sbatch). For the full set of runs use final.sh;
# this is for single reruns. Results go to bench/results/cluster-<dataset>/
# (ecbdl and ecbdl_15m in cluster-ecbdl, sorel and sorel_4m in cluster-sorel,
# chess_* in cluster-chess):
#   cuda-hopper-<data>     H100 PCIe, 2x Xeon Gold 6444Y     --cuda, every CPU
#   cuda-ampere-<data>     A100 80GB PCIe, 2x EPYC 7402      --cuda, every CPU
#   cpu-epyc-<data>        epyc1: 2x EPYC 9255               --parallel, 48 threads
#   cpu-dell-<data>        dell1: 2x Xeon Gold 6542Y         --parallel, 48 threads
#   (only when named:)
#   gpuonly-hopper-<data>, gpuonly-ampere-<data>             --cuda --gpu-only, every CPU
#   serial-epyc-<data>, serial-dell-<data>                   --serial, one thread
#   simple-epyc-<data>, simple-dell-<data>                   --simple, one thread
#   vs-epyc-<data>-<tool>-<algorithm>   scikit-learn (sklearn-cart) or Weka J48
#                          (j48-c45), one thread, every tree hashed; into
#                          cluster-cpu-vs-others with -o
# The GPU nodes run CUDA only, the CPU nodes the CPU backends only: a CPU
# benchmark on a GPU node would measure a CPU that is only there to feed the GPU.
# Build and prepare first (bench/cluster/README.md).
#
#   bench/cluster/submit.sh -m 10 -e -d higgs -p cart_depth10,c45_depth10_unpruned cuda-hopper-higgs
#   bench/cluster/submit.sh -m 3 -e -d mnist8m -o cluster-cpu-vs-others -w epyc2 vs-epyc-mnist8m-j48-c45
# -P <file>: no jobs: append each run to that plan file instead, one line each
# (time limit, then job.sh's arguments and environment), for batch.sh: several
# runs, of any datasets, one after another in one job on one node.
# -m: runs per case; -w: every job on that node instead (e.g. epyc2, the same
# hardware as epyc1); -p: run.py's protocols (default cart_depth10,
# c45_depth10_unpruned); -o: the directory under bench/results/; -a: start only
# after that job (e.g. prepare.sh) ended well; -e: test accuracy too (run.py
# --test-accuracy: the test rows predicted from each dumped tree); -n: the NUMA
# node of the GPU, or of the threads and memory of a CPU run (job.sh NUMA_NODE).

set -euo pipefail
RUNS=1
DATASET=higgs
PROTOCOLS=cart_depth10,c45_depth10_unpruned
OUTDIR=
NUMA=
AFTER=
NODE=
ACCURACY=
PLAN=
while getopts "m:d:p:o:a:n:w:P:eh" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    p) PROTOCOLS="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    a) AFTER="$OPTARG" ;;
    n) NUMA="$OPTARG" ;;
    w) NODE="$OPTARG" ;;
    e) ACCURACY=--test-accuracy ;;
    P) PLAN="$OPTARG" ;;
    *) sed -n '2,50p' "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }
case "$DATASET" in
  susy|higgs|mnist8m) DIR=cluster-$DATASET; LIMIT=1-00:00:00 TIMEOUT=14400 ;;
  ecbdl|ecbdl_15m) DIR=cluster-ecbdl; LIMIT=4-00:00:00 TIMEOUT=36000 ;;
  sorel|sorel_4m) DIR=cluster-sorel; LIMIT=4-00:00:00 TIMEOUT=36000 ;;
  chess_1m|chess_10m|chess_100m|chess_1b) DIR=cluster-chess; LIMIT=1-00:00:00 TIMEOUT=14400 ;;
  covertype) DIR=cluster-test; LIMIT=02:00:00 TIMEOUT=3600 ;;
  *) echo "-d: susy, higgs, mnist8m, ecbdl_15m, ecbdl, sorel_4m, sorel, chess_1m, chess_10m, chess_100m, chess_1b or covertype" >&2
     exit 2 ;;
esac
DIR="${OUTDIR:-$DIR}"

cd "$(dirname "$0")/../.."  # repository root: the jobs start here (SLURM_SUBMIT_DIR)
for f in tree tree_cpu bench/.tools/rusage bench/.venv/bin/python "bench/data/$DATASET/meta.json"; do
  # With -a the data may still be in the making.
  [ -e "$f" ] || [ -n "$AFTER" -a "${f#bench/data/}" != "$f" ] ||
    { echo "$f missing: build and prepare first (bench/cluster/README.md)" >&2; exit 1; }
done

# dell1 is in the dell (12 hours at most) and dell-long (14 days) partitions.
#        name                    partition  node    impls          threads  [protocols  time limit  timeout]
JOBS=("cuda-hopper-$DATASET     hopper     hopper  tree_cuda      all"
      "cuda-ampere-$DATASET     ampere     ampere  tree_cuda      all"
      "cpu-epyc-$DATASET        epyc       epyc1   tree           all"
      "cpu-dell-$DATASET        dell-long  dell1   tree           all")
# Full ECBDL and SOREL do not fit in 80 GB of GPU memory: CPU nodes only.
GPU_JOBS=()
if [ "$DATASET" = ecbdl ] || [ "$DATASET" = sorel ]; then
  GPU_JOBS=("${JOBS[@]:0:2}")
  JOBS=("${JOBS[@]:2}")
fi
OPTIONAL=(${GPU_JOBS[@]+"${GPU_JOBS[@]}"}
          "gpuonly-hopper-$DATASET  hopper     hopper  tree_gpu_only  all"
          "gpuonly-ampere-$DATASET  ampere     ampere  tree_gpu_only  all"
          "serial-epyc-$DATASET     epyc       epyc1   tree           1"
          "serial-dell-$DATASET     dell-long  dell1   tree           1"
          "simple-epyc-$DATASET     epyc       epyc1   tree_simple    1"
          "simple-dell-$DATASET     dell-long  dell1   tree_simple    1"
          "vs-epyc-$DATASET-sklearn-cart  epyc  epyc1  sklearn  1  cart_depth10          7-00:00:00  86400"
          "vs-epyc-$DATASET-j48-c45       epyc  epyc1  j48      1  c45_depth10_unpruned  7-00:00:00  86400")

# Only the runs named on the command line, if any.
if [ $# -gt 0 ]; then
  selected=()
  for job in "${JOBS[@]}" "${OPTIONAL[@]}"; do
    read -r name _ _ <<<"$job"
    for wanted in "$@"; do [ "$wanted" = "$name" ] && selected+=("$job"); done
  done
  [ ${#selected[@]} -eq $# ] || { echo "unknown run name in: $*" >&2; exit 2; }
  JOBS=("${selected[@]}")
fi

# run.py appends to results.jsonl: never mix two runs in one directory.
for job in "${JOBS[@]}"; do
  read -r name _ _ <<<"$job"
  if [ -e "bench/results/$DIR/$name/results.jsonl" ]; then
    echo "bench/results/$DIR/$name already has results: move them away before running again" >&2
    exit 1
  fi
done

# The commit, for machine.json (hopper has no git): the jobs inherit it.
git diff --quiet HEAD -- src bench Makefile ||
  echo "warning: uncommitted changes in src/, bench/ or Makefile: machine.json will say -dirty"
export TREE_GIT_DESCRIBE="$(git describe --always --dirty --abbrev=12)"
echo "code: $TREE_GIT_DESCRIBE, $DATASET, $PROTOCOLS -> bench/results/$DIR, runs per case: $RUNS${NUMA:+, NUMA node $NUMA}${ACCURACY:+, test accuracy}"

ids=()
for job in "${JOBS[@]}"; do
  read -r name partition node impls threads protocols limit timeout <<<"$job"
  node="${NODE:-$node}"
  mkdir -p "bench/results/$DIR/$name"
  # The library runs check that they grew the same tree on every run.
  extra=; [[ "$name" != vs-* ]] || extra=",RUN_EXTRA=--hash-trees"
  if [ -n "$PLAN" ]; then
    environment="${NUMA:+,NUMA_NODE=$NUMA}${ACCURACY:+,ACCURACY=$ACCURACY}$extra"
    echo "${limit:-$LIMIT} $DIR $name $DATASET $impls $RUNS ${protocols:-$PROTOCOLS} $threads ${timeout:-$TIMEOUT}" \
      "${environment:+${environment#,}}" >> "$PLAN"
    echo "$name: planned in $PLAN"
    continue
  fi
  ids+=("$(sbatch --parsable --hold -J "dt-$name" -p "$partition" -w "$node" --exclusive -t "${limit:-$LIMIT}" \
    --export=ALL${NUMA:+,NUMA_NODE=$NUMA}${ACCURACY:+,ACCURACY=$ACCURACY}$extra \
    ${AFTER:+--dependency=afterok:$AFTER} -o "bench/results/$DIR/$name/run.log" \
    bench/cluster/job.sh "$DIR" "$name" "$DATASET" "$impls" "$RUNS" "${protocols:-$PROTOCOLS}" "$threads" \
    "${timeout:-$TIMEOUT}")")
  echo "$name: job ${ids[-1]} on $node"
done
[ ${#ids[@]} -gt 0 ] || exit 0
scontrol release "${ids[@]}"
squeue -a -u "$USER" -o "%.8i %.10P %.20j %.2t %.10M %R"

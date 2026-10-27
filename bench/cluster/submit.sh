#!/usr/bin/env bash
# The cluster benchmark on mordor3: one SLURM job per node, one whole node
# each, held until all are queued and then released together (run it on the
# login node; it only calls sbatch). Results go to
# bench/results/cluster-<dataset>/ (cluster-higgs, -airline, -susy, -ecbdl, -chess):
#   cuda-hopper-<data>  H100 PCIe (GPU 0 of 4), 2x Xeon Gold 6444Y   CUDA
#   cuda-ampere-<data>  A100 80GB PCIe (GPU 0), 2x EPYC 7402          CUDA
#   cpu-epyc-<data>     epyc1: 2x EPYC 9255                           parallel
#   cpu-dell-<data>     dell1: 2x Xeon Gold 6542Y                     parallel
#   cpu-dell512-<data>  dell2: the same CPUs, 512 GB instead of 1 TB  parallel
#                       (only when named, e.g. for the memory pressure of full ECBDL)
#   socket-epyc-<data>  epyc1, one socket: its 24 CPUs and memory     parallel
#   socket-dell-<data>  dell1, the same (job.sh SOCKET=auto)          (only when named)
#   serial-epyc-<data>  epyc1, one thread                             serial
#   serial-dell-<data>  dell1, one thread                             serial
#                       (only when named; the job still owns the whole node)
#   gpuonly-hopper-<data>  hopper, --cuda --gpu-only, every CPU       CUDA, GPU only
#   gpuonly-ampere-<data>  ampere, the same                           (only when named)
#   simple-epyc-<data>  epyc1, one socket: ./tree --simple, --serial, parallel,
#                       scikit-learn and Weka J48, every tree hashed
#                       (only when named; -o cluster-cpu-vs-others, the "simple" benchmark)
#   simple-epyc-<data>-<tool>-<protocol>  the same, one tool and protocol per job
#                       (tool: simple, serial, parallel, dual (parallel on both
#                       sockets, every CPU), sklearn, j48; protocol:
#                       cart, c45; J48 has no CART), each with its own time limit,
#                       so a tool that fails or runs out of time stops only its own
#                       job (HIGGS: hours of J48)
# The GPU nodes run CUDA only, the CPU nodes parallel only: a CPU benchmark
# on a GPU node would measure a CPU that is only there to feed the GPU.
# The epyc and dell partitions have more nodes: the runs always take epyc1
# and dell1, so runs of both datasets on one partition wait for each other.
# Build and prepare first (bench/cluster/README.md).
#
#   bench/cluster/submit.sh                  # HIGGS, each case once, all four nodes
#   bench/cluster/submit.sh -m 3             # each case 3 times
#   bench/cluster/submit.sh -m 3 -d airline  # the airline dataset
#   bench/cluster/submit.sh -m 3 cuda-hopper-higgs   # only the named runs
#   bench/cluster/submit.sh -m 3 -d airline -p cart_full_nodepth,c45_unpruned \
#       -o cluster-airline/noprunning                # unpruned trees
#   bench/cluster/submit.sh -m 3 -e -d higgs -p cart_depth10,cart_depth10_alpha,c45_depth10_unpruned,c45_depth10 \
#       cuda-hopper-higgs gpuonly-hopper-higgs cuda-ampere-higgs gpuonly-ampere-higgs \
#       cpu-epyc-higgs cpu-dell-higgs            # the test-set benchmark (cluster-higgs/README.md)
#   bench/cluster/submit.sh -d ecbdl_15m -a 1234 # ECBDL'14, 15M rows, after job 1234
#   bench/cluster/submit.sh -d ecbdl             # full ECBDL'14: CPU nodes only (no GPU
#                                                # has the 160 GB it needs)
#   bench/cluster/submit.sh -m 3 -d chess_1b -p cart_full_nodepth,c45_unpruned \
#       cuda-hopper-chess_1b cuda-ampere-chess_1b cpu-epyc-chess_1b cpu-dell-chess_1b \
#       cpu-dell512-chess_1b                     # chess: 1m, 10m, 100m, 1b; with dell2
#   bench/cluster/submit.sh -m 3 -d chess_1b -p cart_full_nodepth,c45_unpruned \
#       serial-epyc-chess_1b serial-dell-chess_1b    # one thread (--serial)
#   bench/cluster/submit.sh -m 3 -d chess_1b -p cart_full_nodepth,c45_unpruned \
#       gpuonly-hopper-chess_1b gpuonly-ampere-chess_1b  # every node on the GPU
#   bench/cluster/submit.sh -m 3 socket-epyc-higgs socket-dell-higgs  # one socket
#   bench/cluster/submit.sh -d covertype -o cluster-cpu-vs-others simple-epyc-covertype
#   bench/cluster/submit.sh -m 3 -d higgs -o cluster-cpu-vs-others $(for t in parallel serial \
#       sklearn simple; do echo simple-epyc-higgs-$t-cart simple-epyc-higgs-$t-c45; done) \
#       simple-epyc-higgs-j48-c45                # one job per tool and protocol
# -w: every job on that node instead (e.g. epyc2, the same hardware as epyc1);
# -p: run.py's protocols (default cart_alpha_nodepth,c45: no depth limit, as
# C4.5 has none); -o: the directory under bench/results/; -a: start only after
# that job (e.g. prepare.sh) ended well; -e: test accuracy too (run.py
# --test-accuracy: the test rows predicted from each dumped tree); -n: threads, memory and GPU on that
# NUMA node (job.sh NUMA_NODE). The CUDA runs (cuda-*, gpuonly-*) stay on one
# socket, their GPU's, without it too: job.sh takes the NUMA node of a GPU
# with the most free memory (hopper: node 1, as another user's /dev/shm filled
# node 0 in 2026-10). Only the parallel CPU runs use both sockets, and the
# socket-* runs measure the parallel backend on one of them.

set -euo pipefail
RUNS=1
DATASET=higgs
PROTOCOLS=cart_alpha_nodepth,c45
OUTDIR=
NUMA=
AFTER=
NODE=
ACCURACY=
while getopts "m:d:p:o:a:n:w:eh" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    p) PROTOCOLS="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    a) AFTER="$OPTARG" ;;
    n) NUMA="$OPTARG" ;;
    w) NODE="$OPTARG" ;;
    e) ACCURACY=--test-accuracy ;;
    *) sed -n '2,50p' "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }
case "$DATASET" in
  higgs) DIR=cluster-higgs; LIMIT=01:00:00 TIMEOUT=3600 ;;
  airline) DIR=cluster-airline; LIMIT=04:00:00 TIMEOUT=3600 ;;
  # ECBDL: the dell partition allows at most 12 hours.
  ecbdl_15m) DIR=cluster-ecbdl; LIMIT=08:00:00 TIMEOUT=10800 ;;
  ecbdl) DIR=cluster-ecbdl; LIMIT=12:00:00 TIMEOUT=18000 ;;
  chess_1m|chess_10m|chess_100m|chess_1b) DIR=cluster-chess; LIMIT=02:00:00 TIMEOUT=3600 ;;
  covertype) DIR=cluster-cpu-vs-others; LIMIT=02:00:00 TIMEOUT=3600 ;;
  # SUSY (4.5M rows) for the simple benchmark: hours of --simple, scikit-learn and J48.
  susy) DIR=cluster-cpu-vs-others; LIMIT=12:00:00 TIMEOUT=10800 ;;
  *) echo "-d: higgs, airline, ecbdl_15m, ecbdl, covertype, susy or chess_1m, chess_10m, chess_100m, chess_1b" >&2
     exit 2 ;;
esac
DIR="${OUTDIR:-$DIR}"

cd "$(dirname "$0")/../.."  # repository root: the jobs start here (SLURM_SUBMIT_DIR)
for f in tree tree_cpu bench/.tools/rusage bench/.venv/bin/python "bench/data/$DATASET/train.csv"; do
  # With -a the data may still be in the making.
  [ -e "$f" ] || [ -n "$AFTER" -a "${f#bench/data/}" != "$f" ] ||
    { echo "$f missing: build and prepare first (bench/cluster/README.md)" >&2; exit 1; }
done

#        name                    partition  node    impls      threads  [protocols  time limit  timeout]
JOBS=("cuda-hopper-$DATASET  hopper     hopper  tree_cuda  all"
      "cuda-ampere-$DATASET  ampere     ampere  tree_cuda  all"
      "cpu-epyc-$DATASET     epyc       epyc1   tree       all"
      "cpu-dell-$DATASET     dell       dell1   tree       all")

# Full ECBDL needs about 160 GB of GPU memory: CPU nodes only.
if [ "$DATASET" = ecbdl ]; then
  JOBS=("${JOBS[@]:2}")
fi
OPTIONAL=("cpu-dell512-$DATASET  dell-512GB dell2   tree       all"
          "socket-epyc-$DATASET  epyc       epyc1   tree       all"
          "socket-dell-$DATASET  dell       dell1   tree       all"
          "serial-epyc-$DATASET  epyc       epyc1   tree       1"
          "serial-dell-$DATASET  dell       dell1   tree       1"
          "gpuonly-hopper-$DATASET  hopper  hopper  tree_gpu_only  all"
          "gpuonly-ampere-$DATASET  ampere  ampere  tree_gpu_only  all"
          "simple-epyc-$DATASET  epyc       epyc1   tree,tree_simple,sklearn,j48  1,all")
# The simple benchmark, one job per tool and protocol: the time limit of the
# job, and the timeout of each process, fit the tool (HIGGS, one thread:
# ./tree --serial about 40 s, --simple about 10x that, J48 hours).
#     tool      impls        threads  time limit   timeout (s)
for tool in "parallel  tree         all  02:00:00    3600" \
            "dual      tree         all  02:00:00    3600" \
            "serial    tree         1    04:00:00    3600" \
            "sklearn   sklearn      1    1-00:00:00  21600" \
            "simple    tree_simple  1    1-00:00:00  21600" \
            "j48       j48          1    3-00:00:00  86400"; do
  read -r t impls threads limit timeout <<<"$tool"
  OPTIONAL+=("simple-epyc-$DATASET-$t-c45  epyc  epyc1  $impls  $threads  c45  $limit  $timeout")
  [ "$t" = j48 ] ||  # Weka has no CART
    OPTIONAL+=("simple-epyc-$DATASET-$t-cart  epyc  epyc1  $impls  $threads  cart_alpha_nodepth  $limit  $timeout")
done

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
  socket=; [[ "$name" != socket-* && "$name" != simple-* ]] || socket=,SOCKET=auto
  [[ "$name" != simple-*-dual-* ]] || socket=  # both sockets
  # The simple runs check that every backend grew the same tree.
  extra=; [[ "$name" != simple-* ]] || extra=",RUN_EXTRA=--hash-trees"
  ids+=("$(sbatch --parsable --hold -J "dt-$name" -p "$partition" -w "$node" --exclusive -t "${limit:-$LIMIT}" \
    --export=ALL${NUMA:+,NUMA_NODE=$NUMA}${ACCURACY:+,ACCURACY=$ACCURACY}$socket$extra \
    ${AFTER:+--dependency=afterok:$AFTER} -o "bench/results/$DIR/$name/run.log" \
    bench/cluster/job.sh "$DIR" "$name" "$DATASET" "$impls" "$RUNS" "${protocols:-$PROTOCOLS}" "$threads" \
    "${timeout:-$TIMEOUT}")")
  echo "$name: job ${ids[-1]} on $node"
done
scontrol release "${ids[@]}"
squeue -a -u "$USER" -o "%.8i %.10P %.20j %.2t %.10M %R"

#!/usr/bin/env bash
# The cluster benchmark on mordor3: one SLURM job per node, one whole node
# each, held until all are queued and then released together (run it on the
# login node; it only calls sbatch). Results go to
# bench/results/cluster-higgs/ or bench/results/cluster-airlines/:
#   cuda-hopper-<data>  H100 PCIe (GPU 0 of 4), 2x Xeon Gold 6444Y   CUDA
#   cuda-ampere-<data>  A100 80GB PCIe (GPU 0), 2x EPYC 7402          CUDA
#   cpu-epyc-<data>     epyc1: 2x EPYC 9255                           parallel
#   cpu-dell-<data>     dell1: 2x Xeon Gold 6542Y                     parallel
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
#       -o cluster-airlines/noprunning               # unpruned trees
#   bench/cluster/submit.sh -d ecbdl_6m -a 1234  # ECBDL'14, 6M rows, after job 1234
#   bench/cluster/submit.sh -d ecbdl             # full ECBDL'14: CPU nodes only (no GPU
#                                                # has the 320 GB it needs)
# -p: run.py's protocols (default cart_alpha_nodepth,c45: no depth limit, as
# C4.5 has none); -o: the directory under bench/results/; -a: start only after
# that job (e.g. prepare.sh) ended well.

set -euo pipefail
RUNS=1
DATASET=higgs
PROTOCOLS=cart_alpha_nodepth,c45
OUTDIR=
AFTER=
while getopts "m:d:p:o:a:h" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    p) PROTOCOLS="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    a) AFTER="$OPTARG" ;;
    *) sed -n '2,28p' "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }
case "$DATASET" in
  higgs) DIR=cluster-higgs; LIMIT=01:00:00 TIMEOUT=3600 ;;
  airline) DIR=cluster-airlines; LIMIT=04:00:00 TIMEOUT=3600 ;;
  # ECBDL: the dell partition allows at most 12 hours.
  ecbdl_6m) DIR=cluster-ecbdl; LIMIT=08:00:00 TIMEOUT=10800 ;;
  ecbdl) DIR=cluster-ecbdl; LIMIT=12:00:00 TIMEOUT=18000 ;;
  *) echo "-d: higgs, airline, ecbdl_6m or ecbdl" >&2; exit 2 ;;
esac
DIR="${OUTDIR:-$DIR}"

cd "$(dirname "$0")/../.."  # repository root: the jobs start here (SLURM_SUBMIT_DIR)
for f in tree tree_cpu bench/.tools/rusage bench/.venv/bin/python "bench/data/$DATASET/train.csv"; do
  # With -a the data may still be in the making.
  [ -e "$f" ] || [ -n "$AFTER" -a "${f#bench/data/}" != "$f" ] ||
    { echo "$f missing: build and prepare first (bench/cluster/README.md)" >&2; exit 1; }
done

#        name                    partition  node    impls
JOBS=("cuda-hopper-$DATASET  hopper     hopper  tree_cuda"
      "cuda-ampere-$DATASET  ampere     ampere  tree_cuda"
      "cpu-epyc-$DATASET     epyc       epyc1   tree"
      "cpu-dell-$DATASET     dell       dell1   tree")

# Full ECBDL needs about 320 GB of GPU memory: CPU nodes only.
if [ "$DATASET" = ecbdl ]; then
  JOBS=("${JOBS[@]:2}")
fi

# Only the runs named on the command line, if any.
if [ $# -gt 0 ]; then
  selected=()
  for job in "${JOBS[@]}"; do
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
echo "code: $TREE_GIT_DESCRIBE, $DATASET, $PROTOCOLS -> bench/results/$DIR, runs per case: $RUNS"

ids=()
for job in "${JOBS[@]}"; do
  read -r name partition node impls <<<"$job"
  mkdir -p "bench/results/$DIR/$name"
  ids+=("$(sbatch --parsable --hold -J "dt-$name" -p "$partition" -w "$node" --exclusive -t "$LIMIT" \
    ${AFTER:+--dependency=afterok:$AFTER} -o "bench/results/$DIR/$name/run.log" \
    bench/cluster/job.sh "$DIR" "$name" "$DATASET" "$impls" "$RUNS" "$PROTOCOLS" all "$TIMEOUT")")
  echo "$name: job ${ids[-1]} on $node"
done
scontrol release "${ids[@]}"
squeue -a -u "$USER" -o "%.8i %.10P %.20j %.2t %.10M %R"

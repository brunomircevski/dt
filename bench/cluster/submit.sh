#!/usr/bin/env bash
# The cluster benchmark on mordor3: one SLURM job per node, one whole node
# each, held until all are queued and then released together (run it on the
# login node; it only calls sbatch). Results go to
# bench/results/cluster-higgs/ or bench/results/cluster-airlines/:
#   cuda-hopper-<data>  H100 PCIe (GPU 0 of 4), 2x Xeon Gold 6444Y   parallel + CUDA
#   cuda-ampere-<data>  A100 80GB PCIe (GPU 0), 2x EPYC 7402          parallel + CUDA
#   cuda-turing-<data>  RTX 2080 Ti (GPU 0), Threadripper 1950X       parallel + CUDA
#                       (airline: parallel only, it needs ~25 GB of GPU memory)
#   cpu-epyc-<data>     2x EPYC 9255                                  parallel
#   cpu-dell-<data>     2x Xeon Gold 6542Y                            parallel
# Build and prepare first (bench/cluster/README.md).
#
#   bench/cluster/submit.sh                  # HIGGS, each case once, all five nodes
#   bench/cluster/submit.sh -m 3             # each case 3 times
#   bench/cluster/submit.sh -m 3 -d airline  # the airline dataset
#   bench/cluster/submit.sh -m 3 cuda-turing-higgs   # only the named runs

set -euo pipefail
RUNS=1
DATASET=higgs
while getopts "m:d:h" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    *) sed -n '2,19p' "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }
case "$DATASET" in
  higgs) DIR=cluster-higgs; TURING=tree,tree_cuda; LIMIT=01:00:00 ;;
  airline) DIR=cluster-airlines; TURING=tree; LIMIT=04:00:00 ;;
  *) echo "-d: higgs or airline" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/../.."  # repository root: the jobs start here (SLURM_SUBMIT_DIR)
for f in tree tree_cpu bench/.tools/rusage bench/.venv/bin/python "bench/data/$DATASET/train.csv"; do
  [ -e "$f" ] || { echo "$f missing: build and prepare first (bench/cluster/README.md)" >&2; exit 1; }
done

#        name                    partition  impls
JOBS=("cuda-hopper-$DATASET  hopper     tree,tree_cuda"
      "cuda-ampere-$DATASET  ampere     tree,tree_cuda"
      "cuda-turing-$DATASET  turing     $TURING"
      "cpu-epyc-$DATASET     epyc       tree"
      "cpu-dell-$DATASET     dell       tree")

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
echo "code: $TREE_GIT_DESCRIBE, $DATASET, runs per case: $RUNS"

ids=()
for job in "${JOBS[@]}"; do
  read -r name partition impls <<<"$job"
  mkdir -p "bench/results/$DIR/$name"
  ids+=("$(sbatch --parsable --hold -J "dt-$name" -p "$partition" --exclusive -N 1 -t "$LIMIT" \
    -o "bench/results/$DIR/$name/run.log" bench/cluster/job.sh "$DIR" "$name" "$DATASET" "$impls" "$RUNS")")
  echo "$name: job ${ids[-1]} on partition $partition"
done
scontrol release "${ids[@]}"
squeue -a -u "$USER" -o "%.8i %.10P %.20j %.2t %.10M %R"

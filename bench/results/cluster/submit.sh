#!/usr/bin/env bash
# HIGGS on the mordor3 cluster: submits four SLURM jobs that start together,
# one whole node each (run on the login node; it only calls sbatch):
#   cuda-hopper-higgs  4x H100 PCIe (GPU 0 used), 2x Xeon Gold 6444Y   parallel + CUDA
#   cuda-ampere-higgs  A100 80GB PCIe (GPU 0), 2x EPYC 7402           parallel + CUDA
#   cpu-epyc-higgs     2x EPYC 9255                                   parallel
#   cpu-dell-higgs     2x Xeon Gold 6542Y                             parallel
# Run bench/results/cluster/prepare.sh (sbatch) first.
#
#   bench/results/cluster/submit.sh          # each case once
#   bench/results/cluster/submit.sh -m 3     # each case 3 times

set -euo pipefail
RUNS=1
while getopts "m:h" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    *) sed -n '2,12p' "$0"; exit 2 ;;
  esac
done
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../.."  # repository root: the jobs start here (SLURM_SUBMIT_DIR)
for f in tree tree_cpu bench/.tools/rusage bench/.venv/bin/python bench/data/higgs/train.csv; do
  [ -e "$f" ] || { echo "$f missing: run sbatch bench/results/cluster/prepare.sh first" >&2; exit 1; }
done

#        name               partition  impls
JOBS=("cuda-hopper-higgs  hopper     tree,tree_cuda"
      "cuda-ampere-higgs  ampere     tree,tree_cuda"
      "cpu-epyc-higgs     epyc       tree"
      "cpu-dell-higgs     dell       tree")

# run.py appends to results.jsonl: never mix two runs in one directory.
for job in "${JOBS[@]}"; do
  read -r name _ _ <<<"$job"
  if [ -e "$HERE/$name/results.jsonl" ]; then
    echo "$HERE/$name already has results: move them away before running again" >&2
    exit 1
  fi
done

# The commit, for machine.json (hopper has no git): the jobs inherit it.
git diff --quiet HEAD -- src bench Makefile ||
  echo "warning: uncommitted changes in src/, bench/ or Makefile: machine.json will say -dirty"
export TREE_GIT_DESCRIBE="$(git describe --always --dirty --abbrev=12)"
echo "code: $TREE_GIT_DESCRIBE, runs per case: $RUNS"

# Held until all four are queued, then released together: on idle nodes they
# start at the same time.
ids=()
for job in "${JOBS[@]}"; do
  read -r name partition impls <<<"$job"
  mkdir -p "$HERE/$name"
  ids+=("$(sbatch --parsable --hold -J "dt-$name" -p "$partition" --exclusive -N 1 -t 01:00:00 \
    -o "$HERE/$name/run.log" "$HERE/job.sh" "$name" "$impls" "$RUNS")")
  echo "$name: job ${ids[-1]} on partition $partition"
done
scontrol release "${ids[@]}"
squeue -u "$USER" -o "%.8i %.10P %.20j %.2t %.10M %R"

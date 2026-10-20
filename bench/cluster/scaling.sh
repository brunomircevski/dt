#!/usr/bin/env bash
# Thread scaling of ./tree on the two CPU nodes of mordor3, as the scaling
# benchmarks of the desktop and laptop (consumer/cpu-pc-susy-scaling): pruned CART
# without a depth limit (cart_alpha_nodepth) and C4.5 (c45), --parallel with
# 1, 2, 4, then every 4 threads up to the node's 48 CPUs (1 thread: --serial).
# One whole node per job (job.sh), both released together. Results:
# bench/results/cluster-cpu-scaling/cpu-epyc-<data>/ and cpu-dell-<data>/.
#
#   bench/cluster/scaling.sh -m 3            # HIGGS, each case 3 times
#   bench/cluster/scaling.sh -m 3 -d airline

set -euo pipefail
RUNS=1
DATASET=higgs
while getopts "m:d:h" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    *) sed -n '2,10p' "$0"; exit 2 ;;
  esac
done
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }
DIR=cluster-cpu-scaling

cd "$(dirname "$0")/../.."  # repository root: the jobs start here (SLURM_SUBMIT_DIR)
for f in tree_cpu bench/.tools/rusage bench/.venv/bin/python "bench/data/$DATASET/train.csv"; do
  [ -e "$f" ] || { echo "$f missing: build and prepare first (bench/cluster/README.md)" >&2; exit 1; }
done
git diff --quiet HEAD -- src bench Makefile ||
  echo "warning: uncommitted changes in src/, bench/ or Makefile: machine.json will say -dirty"
export TREE_GIT_DESCRIBE="$(git describe --always --dirty --abbrev=12)"
echo "code: $TREE_GIT_DESCRIBE, $DATASET -> bench/results/$DIR, runs per case: $RUNS"

ids=()
for job in "cpu-epyc-$DATASET epyc epyc1" "cpu-dell-$DATASET dell dell1"; do
  read -r name partition node <<<"$job"
  # run.py appends to results.jsonl: never mix two runs in one directory.
  if [ -e "bench/results/$DIR/$name/results.jsonl" ]; then
    echo "bench/results/$DIR/$name already has results: move them away before running again" >&2
    exit 1
  fi
  mkdir -p "bench/results/$DIR/$name"
  ids+=("$(sbatch --parsable --hold -J "dt-scaling-$name" -p "$partition" -w "$node" --exclusive \
    -t 04:00:00 -o "bench/results/$DIR/$name/run.log" \
    bench/cluster/job.sh "$DIR" "$name" "$DATASET" tree "$RUNS" cart_alpha_nodepth,c45 scaling)")
  echo "$name: job ${ids[-1]} on $node"
done
scontrol release "${ids[@]}"
squeue -a -u "$USER" -o "%.8i %.10P %.24j %.2t %.10M %R"

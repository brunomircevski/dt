#!/usr/bin/env bash
# Times tree building on every backend and prints a Markdown table.
#
#   bench/bench.sh                          # all cases whose dataset exists
#   bench/bench.sh covertype susy           # only these datasets
#   BIN=/path/to/tree REPEAT=3 bench/bench.sh
#
# Columns: "build" = growing the (main) tree, "train" = everything the
# trainer does (presort / GPU setup, build, cross-validation, pruning), best
# of REPEAT runs. Serial runs of the large datasets are done once.

set -euo pipefail
cd "$(dirname "$0")/.."

BIN=${BIN:-./tree}
REPEAT=${REPEAT:-2}

declare -A path=([covertype]=datasets/covertype.csv
                 [susy]=datasets/supersymmetry.csv
                 [higgs]=datasets/higgs.csv)

# dataset | name | flags | backends
cases=(
  "covertype|CART|--cart --no-prune|serial parallel cuda"
  "covertype|CART + 10-fold CV|--cart --cv 10|serial parallel cuda"
  "covertype|C4.5|--c45|serial parallel cuda"
  "susy|CART|--cart --no-prune|serial parallel cuda"
  "susy|CART + 10-fold CV|--cart --cv 10|parallel cuda"
  "susy|C4.5|--c45|serial parallel cuda"
  "higgs|CART|--cart --no-prune|serial parallel cuda"
  "higgs|CART + 10-fold CV|--cart --cv 10|parallel cuda"
  "higgs|C4.5|--c45|serial parallel cuda"
)

selected=("$@")
wanted() {
  ((${#selected[@]} == 0)) && return 0
  local name
  for name in "${selected[@]}"; do [[ $name == "$1" ]] && return 0; done
  return 1
}

# Prints "<build ms> <train ms> <nodes>" (best build of the repeats).
measure() {
  local file=$1 flags=$2 repeats=$3 best_build="" best_train="" nodes="" out
  for ((run = 0; run < repeats; ++run)); do
    # shellcheck disable=SC2086
    out=$("$BIN" $flags "$file")
    local build train
    build=$(awk '$1 == "build" {print $2}' <<< "$out")
    train=$(awk '$1 == "train" && $2 == "total" {print $3}' <<< "$out")
    nodes=$(awk '$1 == "nodes" {print $2}' <<< "$out" | tr -d ,)
    if [[ -z $best_build ]] || awk -v a="$build" -v b="$best_build" 'BEGIN {exit !(a < b)}'; then
      best_build=$build
      best_train=$train
    fi
  done
  echo "$best_build $best_train $nodes"
}

echo "| Dataset | Algorithm | Backend | Build (ms) | Train total (ms) | Nodes |"
echo "|---|---|---|---:|---:|---:|"
for entry in "${cases[@]}"; do
  IFS='|' read -r dataset name flags backends <<< "$entry"
  wanted "$dataset" || continue
  file=${path[$dataset]}
  [[ -f $file ]] || continue
  for backend in $backends; do
    repeats=$REPEAT
    [[ $backend == serial && $dataset != covertype ]] && repeats=1
    read -r build train nodes < <(measure "$file" "--$backend $flags" "$repeats")
    printf '| %s | %s | %s | %s | %s | %s |\n' "$dataset" "$name" "$backend" "$build" "$train" "$nodes"
  done
done

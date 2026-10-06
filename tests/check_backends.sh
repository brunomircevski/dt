#!/usr/bin/env bash
# Checks that every backend grows exactly the same tree (byte-identical dump)
# for CART and C4.5 on the datasets in datasets/ (plus samples of the big
# ones). The Cuda backend is tested with several hand-off sizes, including one
# that keeps almost every node on the GPU.
#
#   tests/check_backends.sh            # uses ./tree (make)
#   TREE_BIN=./tree_cpu tests/check_backends.sh   # CPU backends only

set -euo pipefail
cd "$(dirname "$0")/.."

TREE_BIN=${TREE_BIN:-./tree}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

datasets=()
for name in iris diabetes; do
  [[ -f datasets/$name.csv ]] && datasets+=("datasets/$name.csv")
done
# Samples of the large datasets keep the run short.
for name in covertype supersymmetry; do
  if [[ -f datasets/$name.csv ]]; then
    head -n 60001 "datasets/$name.csv" > "$WORK/${name}_60k.csv"
    datasets+=("$WORK/${name}_60k.csv")
  fi
done

backends=(--parallel)
if "$TREE_BIN" --help | grep -q -- --cuda && "$TREE_BIN" --cuda -d 0 datasets/iris.csv > /dev/null 2>&1; then
  backends+=("--cuda --gpu-min-rows 2" "--cuda --gpu-min-rows 5000" "--cuda")
fi

configs=("--cart" "--cart --criterion entropy --min-leaf 3" "--cart -d 8 --alpha 0.001"
         "--c45" "--c45 --cf 0.1 --min-objs 5" "--c45 --no-prune")

failures=0
for dataset in "${datasets[@]}"; do
  for config in "${configs[@]}"; do
    # shellcheck disable=SC2086
    "$TREE_BIN" --serial $config "$dataset" --dump "$WORK/reference.txt" > /dev/null
    for backend in "${backends[@]}"; do
      # shellcheck disable=SC2086
      "$TREE_BIN" $backend $config "$dataset" --dump "$WORK/other.txt" > /dev/null
      if cmp -s "$WORK/reference.txt" "$WORK/other.txt"; then
        result=ok
      else
        result=DIFFERENT
        failures=$((failures + 1))
      fi
      printf '%-8s %-26s %-40s %s\n' "$result" "$(basename "$dataset")" "$config" "$backend"
    done
  done
done

if ((failures > 0)); then
  echo "$failures configuration(s) differ from --serial"
  exit 1
fi
echo "All backends agree."

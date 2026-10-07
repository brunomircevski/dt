#!/usr/bin/env bash
# Regression test: every backend must grow exactly the tree stored in
# tests/golden/ (byte-identical --dump), for CART and C4.5 with several
# settings. The golden trees come from the version that was verified against
# scikit-learn and Quinlan's c4.5 (see docs/ALGORITHMS.md).
#
#   tests/check.sh                    # ./tree, all backends it supports
#   TREE_BIN=./tree_cpu tests/check.sh
#   tests/check.sh --update           # rewrite the golden files (serial backend)
#
# iris and diabetes are part of the repository; covertype and supersymmetry
# samples are tested when datasets/ has them.

set -euo pipefail
cd "$(dirname "$0")/.."

TREE_BIN=${TREE_BIN:-./tree}
UPDATE=0
[[ "${1:-}" == "--update" ]] && UPDATE=1
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

datasets=(tests/data/iris.csv tests/data/diabetes.csv)
for name in covertype supersymmetry; do
  if [[ -f datasets/$name.csv ]]; then
    head -n 20001 "datasets/$name.csv" > "$WORK/${name}_20k.csv"
    datasets+=("$WORK/${name}_20k.csv")
  fi
done

configs=("--cart"
         "--cart --test-sample 0.25 --holdout 0.3"
         "--cart --no-prune"
         "--cart --min-leaf 3 --no-prune"
         "--cart -d 8 --alpha 0.001"
         "--cart --cv 5"
         "--cart --cv 5 --holdout 0.3"
         "--c45"
         "--c45 --cf 0.1 --min-leaf 5"
         "--c45 --no-prune"
         "--c45 -d 6 --holdout 0.3")

backends=(--serial --parallel "--parallel --threads 3")
if "$TREE_BIN" --cuda -d 0 tests/data/iris.csv > /dev/null 2>&1; then
  backends+=("--cuda --gpu-min-rows 2" "--cuda --gpu-min-rows 5000" "--cuda"
             "--cuda --gpu-sweep one-pass --gpu-min-rows 2" "--cuda --gpu-sweep two-pass")
fi

slug() { echo "$*" | tr -s ' -' '_' | sed 's/^_//'; }

failures=0
for dataset in "${datasets[@]}"; do
  base=$(basename "$dataset" .csv)
  for config in "${configs[@]}"; do
    golden="tests/golden/${base}__$(slug "$config").txt.gz"
    if ((UPDATE)); then
      # shellcheck disable=SC2086
      "$TREE_BIN" --serial $config "$dataset" --dump "$WORK/tree.txt" > /dev/null
      gzip -9n -c "$WORK/tree.txt" > "$golden"
      echo "updated  $golden"
      continue
    fi
    if [[ ! -f "$golden" ]]; then
      echo "missing  $golden"
      failures=$((failures + 1))
      continue
    fi
    gunzip -c "$golden" > "$WORK/golden.txt"
    for backend in "${backends[@]}"; do
      # shellcheck disable=SC2086
      if "$TREE_BIN" $backend $config "$dataset" --dump "$WORK/tree.txt" > "$WORK/log.txt" 2>&1 &&
         cmp -s "$WORK/golden.txt" "$WORK/tree.txt"; then
        result=ok
      else
        result=DIFFERENT
        failures=$((failures + 1))
      fi
      printf '%-9s %-22s %-52s %s\n' "$result" "$base" "$config" "$backend"
    done
  done
done

((UPDATE)) && exit 0
if ((failures > 0)); then
  echo "$failures check(s) failed"
  exit 1
fi
echo "All backends match the golden trees."

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
# iris, diabetes and zero_gain are part of the repository; covertype and
# supersymmetry samples are tested when datasets/ has them. zero_gain is one
# node of a covertype C4.5 tree (24:3 rows) whose only allowed cut keeps the
# class proportions (16:2 and 8:1): its gain is exactly 0, so C4.5 must make
# a leaf, even though rounding computes the gain as +4e-16.

set -euo pipefail
cd "$(dirname "$0")/.."

TREE_BIN=${TREE_BIN:-./tree}
UPDATE=0
[[ "${1:-}" == "--update" ]] && UPDATE=1
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

datasets=(tests/data/iris.csv tests/data/diabetes.csv tests/data/zero_gain.csv)
# The supersymmetry sample is the desktop copy's first 20,000 rows: since
# 2026-10-23 datasets/supersymmetry.csv is in UCI's order, so make_datasets.py
# writes the sample as bench/data/supersymmetry_20k.csv (older copies: their head).
for name in covertype supersymmetry; do
  if [[ $name == supersymmetry && -f bench/data/supersymmetry_20k.csv ]]; then
    cp bench/data/supersymmetry_20k.csv "$WORK/${name}_20k.csv"
    datasets+=("$WORK/${name}_20k.csv")
  elif [[ -f datasets/$name.csv ]]; then
    head -n 20001 "datasets/$name.csv" > "$WORK/${name}_20k.csv"
    datasets+=("$WORK/${name}_20k.csv")
  fi
done

configs=("--cart --prune 1e-4"
         "--cart --prune 1e-5 --holdout 0.3"
         "--cart"
         "--cart --min-leaf 3"
         "--cart -d 8 --prune 0.001"
         "--c45 --prune 0.25"
         "--c45 --prune 0.1 --min-leaf 5"
         "--c45"
         "--c45 -d 6 --prune 0.25 --holdout 0.3")

backends=(--simple --serial --multicore "--multicore --threads 3")
if "$TREE_BIN" --hybrid -d 0 tests/data/iris.csv > /dev/null 2>&1; then
  backends+=("--hybrid --handoff-rows 2" "--hybrid --handoff-rows 5000" "--hybrid"
             "--gpu" "--gpu --threads 1")
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

# A separate test file (--test) must give what --test-last gives on the same
# rows: the same tree and the same test accuracy.
for dataset in "${datasets[@]}"; do
  base=$(basename "$dataset" .csv)
  rows=$(($(awk 'END { print NR }' "$dataset") - 1)) # (the last line may lack its newline)
  last=$((rows / 4))
  head -n $((rows - last + 1)) "$dataset" > "$WORK/train.csv"
  { head -n 1 "$dataset"; tail -n "$last" "$dataset"; } > "$WORK/test.csv"
  for algorithm in --cart --c45; do
    "$TREE_BIN" $algorithm --test-last "$last" "$dataset" --dump "$WORK/a.txt" |
      grep 'test accuracy' > "$WORK/a.log"
    "$TREE_BIN" $algorithm --test "$WORK/test.csv" "$WORK/train.csv" --dump "$WORK/b.txt" |
      grep 'test accuracy' > "$WORK/b.log"
    if cmp -s "$WORK/a.txt" "$WORK/b.txt" && cmp -s "$WORK/a.log" "$WORK/b.log"; then
      result=ok
    else
      result=DIFFERENT
      failures=$((failures + 1))
    fi
    printf '%-9s %-22s %-52s %s\n' "$result" "$base" "$algorithm --test FILE" "= --test-last $last"
  done
done

# --rows N must give what a file of the first N rows gives (here every
# backend's tree is the same, so one backend is enough).
for dataset in "${datasets[@]}"; do
  base=$(basename "$dataset" .csv)
  rows=$(($(awk 'END { print NR }' "$dataset") - 1))
  half=$(((rows + 1) / 2))
  head -n $((half + 1)) "$dataset" > "$WORK/head.csv"
  for algorithm in --cart --c45; do
    "$TREE_BIN" --multicore --threads 3 $algorithm --rows "$half" "$dataset" --dump "$WORK/a.txt" |
      grep 'rows ' > "$WORK/a.log"
    "$TREE_BIN" --multicore --threads 3 $algorithm "$WORK/head.csv" --dump "$WORK/b.txt" |
      grep 'rows ' > "$WORK/b.log"
    if cmp -s "$WORK/a.txt" "$WORK/b.txt" && cmp -s "$WORK/a.log" "$WORK/b.log"; then
      result=ok
    else
      result=DIFFERENT
      failures=$((failures + 1))
    fi
    printf '%-9s %-22s %-52s %s\n' "$result" "$base" "$algorithm --rows $half" "= its first rows' file"
  done
done

if ((failures > 0)); then
  echo "$failures check(s) failed"
  exit 1
fi
echo "All backends match the golden trees."

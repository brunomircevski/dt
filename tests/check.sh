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
# Every dataset is also written as the binary input (tests/csv_to_f32.py:
# X.f32, X.y.i32, meta.json), which must give the same trees as its CSV.
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
# 2026-10-23 supersymmetry.csv is in UCI's order, so make_datasets.py writes
# the sample as datasets/susy/supersymmetry_20k.csv (older copies: their
# head). Each dataset's originals are in its folder, datasets/<dataset>/.
for name in covertype supersymmetry; do
  folder=datasets/$name; [[ $name == supersymmetry ]] && folder=datasets/susy
  if [[ $name == supersymmetry && -f $folder/supersymmetry_20k.csv ]]; then
    cp "$folder/supersymmetry_20k.csv" "$WORK/${name}_20k.csv"
    datasets+=("$WORK/${name}_20k.csv")
  elif [[ -f $folder/$name.csv ]]; then
    head -n 20001 "$folder/$name.csv" > "$WORK/${name}_20k.csv"
    datasets+=("$WORK/${name}_20k.csv")
  fi
done

# Each dataset's binary copy: $WORK/<name>/train.f32 (not written by --update).
binary() { echo "$WORK/$(basename "$1" .csv)/train.f32"; }
if ((!UPDATE)); then
  for dataset in "${datasets[@]}"; do
    python3 tests/csv_to_f32.py "$(dirname "$(binary "$dataset")")" "$dataset"
  done
fi

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
    for input in "$dataset" "$(binary "$dataset")"; do
      name=$base
      [[ $input == *.f32 ]] && name="$base (f32)"
      for backend in "${backends[@]}"; do
        # shellcheck disable=SC2086
        if "$TREE_BIN" $backend $config "$input" --dump "$WORK/tree.txt" > "$WORK/log.txt" 2>&1 &&
           cmp -s "$WORK/golden.txt" "$WORK/tree.txt"; then
          result=ok
        else
          result=DIFFERENT
          failures=$((failures + 1))
        fi
        printf '%-9s %-22s %-52s %s\n' "$result" "$name" "$config" "$backend"
      done
    done
  done
done

((UPDATE)) && exit 0

# A separate test file (--test) must give what --test-last gives on the same
# rows: the same tree and the same test accuracy. Also with binary files, and
# with a binary training file and a CSV test file and the other way round.
for dataset in "${datasets[@]}"; do
  base=$(basename "$dataset" .csv)
  rows=$(($(awk 'END { print NR }' "$dataset") - 1)) # (the last line may lack its newline)
  last=$((rows / 4))
  head -n $((rows - last + 1)) "$dataset" > "$WORK/train.csv"
  { head -n 1 "$dataset"; tail -n "$last" "$dataset"; } > "$WORK/test.csv"
  python3 tests/csv_to_f32.py "$WORK/split" "$WORK/train.csv" "$WORK/test.csv"
  for algorithm in --cart --c45; do
    "$TREE_BIN" $algorithm --test-last "$last" "$dataset" --dump "$WORK/a.txt" |
      grep 'test accuracy' > "$WORK/a.log"
    for files in "$WORK/test.csv $WORK/train.csv" "$WORK/split/test.f32 $WORK/split/train.f32" \
                 "$WORK/test.csv $WORK/split/train.f32" "$WORK/split/test.f32 $WORK/train.csv" \
                 "- $(binary "$dataset")"; do
      read -r test train <<< "$files"
      if [[ $test == - ]]; then # --test-last on the binary copy
        "$TREE_BIN" $algorithm --test-last "$last" "$train" --dump "$WORK/b.txt" |
          grep 'test accuracy' > "$WORK/b.log"
        description="$algorithm --test-last $last f32"
      else
        "$TREE_BIN" $algorithm --test "$test" "$train" --dump "$WORK/b.txt" |
          grep 'test accuracy' > "$WORK/b.log"
        description="$algorithm --test ${test##*.} ${train##*.}"
      fi
      if cmp -s "$WORK/a.txt" "$WORK/b.txt" && cmp -s "$WORK/a.log" "$WORK/b.log"; then
        result=ok
      else
        result=DIFFERENT
        failures=$((failures + 1))
      fi
      printf '%-9s %-22s %-52s %s\n' "$result" "$base" "$description" "= --test-last $last (csv)"
    done
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
    "$TREE_BIN" --multicore --threads 3 $algorithm "$WORK/head.csv" --dump "$WORK/b.txt" |
      grep 'rows ' > "$WORK/b.log"
    for input in "$dataset" "$(binary "$dataset")"; do
      "$TREE_BIN" --multicore --threads 3 $algorithm --rows "$half" "$input" --dump "$WORK/a.txt" |
        grep 'rows ' > "$WORK/a.log"
      if cmp -s "$WORK/a.txt" "$WORK/b.txt" && cmp -s "$WORK/a.log" "$WORK/b.log"; then
        result=ok
      else
        result=DIFFERENT
        failures=$((failures + 1))
      fi
      printf '%-9s %-22s %-52s %s\n' "$result" "$base" "$algorithm --rows $half ${input##*.}" \
        "= its first rows' file"
    done
  done
done

# A CSV as bench/prepare.py writes it with Arrow: the header quotes the names
# ("f0", which ./tree keeps), meta.json has no csv_features / csv_classes. 12
# classes: their ids follow the text order (c10 before c2), also when --rows
# leaves only some of them (9 rows: c2 to c10).
awk -F, 'NR == 1 { print "\"f0\",\"f1\",\"f2\",\"f3\",\"class\""; next }
         { print $2 "," $3 "," $4 "," $5 ",c" NR % 12 }' tests/data/iris.csv > "$WORK/arrow.csv"
python3 tests/csv_to_f32.py "$WORK/arrow" "$WORK/arrow.csv"
for rows in 150 9; do
  for algorithm in --cart --c45; do
    for input in "$WORK/arrow.csv" "$WORK/arrow/train.f32"; do
      log=$WORK/$([[ $input == *.f32 ]] && echo b || echo a)
      "$TREE_BIN" $algorithm --rows "$rows" "$input" --dump "$log.txt" | grep 'rows ' > "$log.log"
    done
    if cmp -s "$WORK/a.txt" "$WORK/b.txt" && cmp -s "$WORK/a.log" "$WORK/b.log"; then
      result=ok
    else
      result=DIFFERENT
      failures=$((failures + 1))
    fi
    printf '%-9s %-22s %-52s %s\n' "$result" "iris_arrow_12_classes" "$algorithm --rows $rows f32" \
      "= the CSV"
  done
done

if ((failures > 0)); then
  echo "$failures check(s) failed"
  exit 1
fi
echo "All backends match the golden trees."

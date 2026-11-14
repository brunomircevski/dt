#!/usr/bin/env bash
# Interleaved A/B runs of ./tree variants on one node (SLURM job), for tuning:
# every round runs every case with every variant, the variants' order rotating
# from round to round, each invocation training --repeat trees on one load.
#
#   sbatch -p ampere -w ampere --exclusive -t 4:00:00 -o ab-ampere.log \
#       bench/cluster/ab.sh <out-dir> <variants-file> <cases-file> [rounds] [repeat]
#
# variants file, one per line:  label|binary|extra args|prefix (e.g. numactl ...)
# cases file, one per line:     case|args|dataset csv
# Lines starting with # are skipped. Writes <out-dir>/times.csv (label, case,
# round, run, train ms, tree sha256) and each invocation's output.
# REPEAT_WARMUP (default 5) is --repeat-warmup; PROFILE=1 adds --profile
# (<out-dir>/out/<label>-<case>-r<round>.json).

set -uo pipefail
OUT="$1" VARIANTS="$2" CASES="$3" ROUNDS="${4:-3}" REPEAT="${5:-3}"
mkdir -p "$OUT/out"
mapfile -t variants < <(grep -v '^\s*#' "$VARIANTS" | grep -v '^\s*$')
mapfile -t cases < <(grep -v '^\s*#' "$CASES" | grep -v '^\s*$')
[ -f "$OUT/times.csv" ] || echo "label,case,round,run,train_ms,sha256" > "$OUT/times.csv"
echo "job ${SLURM_JOB_ID:-} on $(hostname) $(date '+%F %T'): ${#variants[@]} variants, ${#cases[@]} cases, $ROUNDS rounds x $REPEAT"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader 2>/dev/null
for ((round = 1; round <= ROUNDS; ++round)); do
  for c in "${cases[@]}"; do
    IFS='|' read -r cname cargs cdata <<< "$c"
    n=${#variants[@]}
    for ((k = 0; k < n; ++k)); do
      v="${variants[$(( (k + round - 1) % n ))]}"
      IFS='|' read -r label bin vargs prefix <<< "$v"
      args="$cargs"  # --handoff-rows is for --hybrid only
      [[ " $vargs " == *" --hybrid "* ]] || args="$(sed -E 's/--handoff-rows [0-9]+//' <<< "$cargs")"
      log="$OUT/out/$label-$cname-r$round.txt"
      dump="$OUT/out/$label-$cname-r$round.tree"
      # shellcheck disable=SC2086
      profile=(); [ -z "${PROFILE:-}" ] || profile=(--profile "$OUT/out/$label-$cname-r$round.json")
      timeout 3600 $prefix "$bin" $args $vargs "${profile[@]}" --repeat "$REPEAT" --repeat-warmup "${REPEAT_WARMUP:-5}" \
        --dump "$dump" "$cdata" > "$log" 2>&1
      status=$?
      sha=$(sha256sum "$dump" 2>/dev/null | cut -c1-16)
      rm -f "$dump"
      if [ $status -ne 0 ]; then
        echo "$(date '+%T') FAILED ($status) $label $cname r$round: $log"
        continue
      fi
      i=0
      grep -o 'repeat [0-9]*/[0-9]*: train [0-9.]*' "$log" | awk '{print $4}' | while read -r t; do
        i=$((i + 1)); echo "$label,$cname,$round,$i,$t,$sha" >> "$OUT/times.csv"
      done
      echo "$(date '+%T') $label $cname r$round: $(grep -o 'repeat [0-9]*/[0-9]*: train [0-9.]*' "$log" | awk '{printf "%s ", $4}') sha $sha"
    done
  done
done
echo "done $(date '+%F %T')"

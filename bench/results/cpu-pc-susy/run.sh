#!/usr/bin/env bash
# The paper benchmark on SUSY: ./tree vs scikit-learn and rpart (CART) and
# Weka J48 and YaDT (C4.5), 1 and 28 threads. Everything it writes goes into
# this directory; BENCHMARK.md explains every choice.
#
#   bench/results/cpu-pc-susy/run.sh          # each case once (~45 min)
#   bench/results/cpu-pc-susy/run.sh -m 5     # each case 5 times (~3 h)
#
# Every run measures training time, peak memory, tree size and accuracy in the
# same process.

set -euo pipefail
RUNS=1
while getopts "m:h" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    *) sed -n '2,10p' "$0"; exit 2 ;;
  esac
done
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$(basename "$HERE")"
cd "$HERE/../../.."  # repository root
PY=bench/.venv/bin/python
step() { echo; echo "=== $(date +%H:%M:%S)  $*"; }

# run.py appends to results.jsonl: never mix two runs in one directory.
if [ -e "$HERE/results.jsonl" ] || [ -e "$HERE/warmup-check" ]; then
  echo "$HERE already has results: move them away before running again" >&2
  exit 1
fi

step "1/5 machine check"
governors="$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u | tr '\n' ' ')"
[ "$governors" = "performance " ] && echo "CPU governor: performance" ||
  echo "warning: CPU governor is '$governors', not performance"
git diff --quiet HEAD -- src bench Makefile && echo "code: committed ($(git rev-parse --short HEAD))" ||
  echo "warning: uncommitted changes in src/, bench/ or Makefile: machine.json will say -dirty"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg) (close browsers, chat apps, IDE indexers)"
echo "runs per case: $RUNS"

step "2/5 build ./tree_cpu, the J48 adapter and the memory launcher; check the tools"
bench/setup.sh

step "3/5 warm-up check: is a 50,000-row warm-up enough for J48's JIT? (~5 min)"
# J48 on the 500k-row SUSY subset, 3 runs after a 50k-row warm-up and 3 after a
# full one. If the 50k warm-up leaves J48 more than 3% slower, the benchmark
# warms up on all rows instead (for all managed runtimes, so the rule stays the same).
for warmup in 50000 all; do
  $PY bench/run.py susy_500k --protocols c45 --impls j48 --threads 1 --cpus 2 --reps 3 \
    --no-baseline --warmup-rows "$warmup" --run-id "$RUN/warmup-check/$warmup"
done
WARMUP="$($PY - "$HERE/warmup-check" <<'EOF'
import json, statistics, sys
from pathlib import Path
median = {}
for warmup in ("50000", "all"):
    rows = [json.loads(line) for line in open(Path(sys.argv[1]) / warmup / "results.jsonl")]
    median[warmup] = statistics.median(r["train_seconds"] for r in rows if r["status"] == "ok")
ratio = median["50000"] / median["all"]
choice = "50000" if ratio <= 1.03 else "all"
summary = {"j48_median_after_50k_warmup": median["50000"],
           "j48_median_after_full_warmup": median["all"], "ratio": ratio, "chosen": choice}
json.dump(summary, open(Path(sys.argv[1]) / "decision.json", "w"), indent=1)
print(choice, file=sys.stdout)
print(f"warm-up check: J48 {median['50000']:.2f} s after 50k rows, {median['all']:.2f} s after "
      f"all rows (x{ratio:.3f}) -> --warmup-rows {choice}", file=sys.stderr)
EOF
)"

step "4/5 benchmark: SUSY, pruned CART and C4.5, 1 and 28 threads, $RUNS run(s) per case"
# 1 thread runs on CPU 2 (a performance core); 28 threads on CPUs 0-27.
$PY bench/run.py susy --protocols cart_alpha,c45 --threads 1,all --cpus 2,0,1,3-27 \
  --reps "$RUNS" --timeout 3600 --warmup-rows "$WARMUP" --run-id "$RUN"

step "5/5 tables, CSV files and the chart"
$PY bench/report.py "$HERE" --md "$HERE/report.md" --csv "$HERE/summary.csv" \
  --runs-csv "$HERE/runs.csv"
$PY bench/chart.py "$HERE"
echo "done: $HERE/report.md, summary.csv, runs.csv, chart.png"

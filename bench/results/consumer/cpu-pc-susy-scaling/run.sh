#!/usr/bin/env bash
# Thread scaling of ./tree on full SUSY: pruned CART (cart_alpha_nodepth) and C4.5
# (c45), neither with a depth limit, with 1, 2, 4, 8, 12, 16, 20, 24 and 28 threads. Everything it writes goes into
# this directory; BENCHMARK.md explains every choice.
#
#   bench/results/cpu-pc-susy-scaling/run.sh          # each case once
#   bench/results/cpu-pc-susy-scaling/run.sh -m 5     # each case 5 times
#
# A quick check of the script on a small dataset, results elsewhere:
#   bench/results/cpu-pc-susy-scaling/run.sh -d susy_50k -o /tmp/scaling-check

set -euo pipefail
RUNS=1
DATASET=susy
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE"
while getopts "m:d:o:h" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    o) OUT="$(mkdir -p "$OPTARG" && cd "$OPTARG" && pwd)" ;;
    *) sed -n '2,10p' "$0"; exit 2 ;;
  esac
done
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }

RUN="$(basename "$OUT")"
cd "$HERE/../../.."  # repository root
PY=bench/.venv/bin/python
step() { echo; echo "=== $(date +%H:%M:%S)  $*"; }

# Threads go to the CPUs in this order (i7-14700KF; lscpu --extended):
#   2,4,...,14,0   one thread on each of the 8 P-cores (CPU 2 first: the 1-thread
#                  case runs there, as in the other benchmarks; CPU 0 takes most interrupts)
#   16-27          the 12 E-cores
#   3,5,...,15,1   the SMT siblings of the P-cores
CPUS=2,4,6,8,10,12,14,0,16-27,3,5,7,9,11,13,15,1
THREADS=1,2,4,8,12,16,20,24,28

# run.py appends to results.jsonl: never mix two runs in one directory.
if [ -e "$OUT/results.jsonl" ]; then
  echo "$OUT already has results: move them away before running again" >&2
  exit 1
fi
[ -x "$PY" ] && [ -x bench/.tools/rusage ] || { echo "run bench/setup.sh first" >&2; exit 1; }

# Keep the PC from sleeping or shutting down when idle until the run ends, and
# log everything to run.log next to the results.
if [ -z "${BENCH_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
  export BENCH_INHIBITED=1
  exec systemd-inhibit --what=idle:sleep:shutdown --who="bench $RUN" \
    --why="benchmark running" "$HERE/run.sh" "$@"
fi
export PYTHONUNBUFFERED=1
exec > >(trap "" INT; tee -a "$OUT/run.log") 2>&1
echo "log: $OUT/run.log, started $(date '+%F %T')"

step "1/4 machine check"
governors="$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u | tr '\n' ' ')"
[ "$governors" = "performance " ] && echo "CPU governor: performance" ||
  echo "warning: CPU governor is '$governors', not performance"
git diff --quiet HEAD -- src bench Makefile && echo "code: committed ($(git rev-parse --short HEAD))" ||
  echo "warning: uncommitted changes in src/, bench/ or Makefile: machine.json will say -dirty"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg) (close browsers, chat apps, IDE indexers)"
echo "dataset: $DATASET, runs per case: $RUNS, threads: $THREADS"

step "2/4 build ./tree_cpu; prepare the data if it is missing"
make cpu
# Skips what is already prepared; also writes the 200-row _baseline dataset
# that measures ./tree's runtime footprint.
$PY bench/prepare.py "$DATASET"

step "3/4 benchmark: $DATASET, pruned CART and C4.5, no depth limit, ./tree only, $RUNS run(s) per case"
$PY bench/run.py "$DATASET" --protocols cart_alpha_nodepth,c45 --impls tree --threads "$THREADS" \
  --cpus "$CPUS" --reps "$RUNS" --timeout 3600 --run-id "$OUT"

step "4/4 tables, CSV files and the chart"
$PY bench/report.py "$OUT" --md "$OUT/report.md" --csv "$OUT/summary.csv" \
  --runs-csv "$OUT/runs.csv"
$PY bench/scaling_chart.py "$OUT"
echo "done: $OUT/report.md, summary.csv, runs.csv, scaling.csv, chart.png"

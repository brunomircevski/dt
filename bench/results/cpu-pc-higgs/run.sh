#!/usr/bin/env bash
# ./tree parallel (28 threads) on the desktop (i7-14700KF) on full HIGGS: pruned
# CART and C4.5 with the protocols of cpu-pc-susy, to compare with the laptop's
# parallel and CUDA runs in cuda-legion-higgs. Everything it writes goes into
# this directory; BENCHMARK.md explains every choice.
#
#   bench/results/cpu-pc-higgs/run.sh          # each case once
#   bench/results/cpu-pc-higgs/run.sh -m 5     # each case 5 times

set -euo pipefail
RUNS=1
while getopts "m:h" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    *) sed -n '2,8p' "$0"; exit 2 ;;
  esac
done
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$(basename "$HERE")"
cd "$HERE/../../.."  # repository root
PY=bench/.venv/bin/python
step() { echo; echo "=== $(date +%H:%M:%S)  $*"; }

# All 28 logical CPUs of the i7-14700KF, in the order of cpu-pc-susy.
CPUS=2,0,1,3-27

# run.py appends to results.jsonl: never mix two runs in one directory.
if [ -e "$HERE/results.jsonl" ]; then
  echo "$HERE already has results: move them away before running again" >&2
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
exec > >(trap "" INT; tee -a "$HERE/run.log") 2>&1
echo "log: $HERE/run.log, started $(date '+%F %T')"

step "1/4 machine check"
governors="$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u | tr '\n' ' ')"
[ "$governors" = "performance " ] && echo "CPU governor: performance" ||
  echo "warning: CPU governor is '$governors', not performance"
git diff --quiet HEAD -- src bench Makefile && echo "code: committed ($(git rev-parse --short HEAD))" ||
  echo "warning: uncommitted changes in src/, bench/ or Makefile: machine.json will say -dirty"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg) (close browsers, chat apps, IDE indexers)"
echo "runs per case: $RUNS"

step "2/4 build ./tree_cpu; prepare the data if it is missing"
make cpu
"$PY" bench/prepare.py higgs

step "3/4 benchmark: HIGGS, pruned CART and C4.5, parallel (28 threads), $RUNS run(s) per case"
"$PY" bench/run.py higgs --protocols cart_alpha,c45 --impls tree --threads all \
  --cpus "$CPUS" --reps "$RUNS" --timeout 3600 --run-id "$RUN"

step "4/4 tables, CSV files and the chart"
"$PY" bench/report.py "$HERE" --md "$HERE/report.md" --csv "$HERE/summary.csv" \
  --runs-csv "$HERE/runs.csv"
"$PY" bench/chart.py "$HERE"
echo "done: $HERE/report.md, summary.csv, runs.csv, chart.png"

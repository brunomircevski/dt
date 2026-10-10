#!/usr/bin/env bash
# ./tree on the Legion laptop (i7-13650HX, RTX 5070 Laptop GPU) on full SUSY:
# serial (1 thread), parallel (20 threads) and CUDA (GPU + 20 CPU threads),
# pruned CART and C4.5 with the protocols of cpu-pc-susy, so the times compare
# with the desktop's. Everything it writes goes into this directory;
# BENCHMARK.md explains every choice.
#
#   bench/results/cuda-legion-susy/run.sh          # each case once
#   bench/results/cuda-legion-susy/run.sh -m 5     # each case 5 times

set -euo pipefail
RUNS=1
while getopts "m:h" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    *) sed -n '2,9p' "$0"; exit 2 ;;
  esac
done
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$(basename "$HERE")"
cd "$HERE/../../.."  # repository root
PY=bench/.venv/bin/python
step() { echo; echo "=== $(date +%H:%M:%S)  $*"; }

# i7-13650HX (lscpu --extended): CPUs 0-11 are the 6 P-cores (siblings 0/1,
# 2/3, ...), 12-19 the 8 E-cores. The 1-thread case runs on CPU 4, a P-core
# that boosts to 4.9 GHz (CPU 0 takes most interrupts); 20 threads use them all.
CPUS=4,0-3,5-19

# run.py appends to results.jsonl: never mix two runs in one directory.
if [ -e "$HERE/results.jsonl" ]; then
  echo "$HERE already has results: move them away before running again" >&2
  exit 1
fi

# Keep the laptop from sleeping or shutting down when idle until the run ends,
# and log everything to run.log next to the results.
if [ -z "${BENCH_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
  export BENCH_INHIBITED=1
  exec systemd-inhibit --what=idle:sleep:shutdown:handle-lid-switch --who="bench $RUN" \
    --why="benchmark running" "$HERE/run.sh" "$@"
fi
export PYTHONUNBUFFERED=1
exec > >(trap "" INT; tee -a "$HERE/run.log") 2>&1
echo "log: $HERE/run.log, started $(date '+%F %T')"

step "1/5 machine check"
governors="$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u | tr '\n' ' ')"
[ "$governors" = "performance " ] && echo "CPU governor: performance" ||
  echo "warning: CPU governor is '$governors', not performance"
profile="$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)"
[ "$profile" = "max-power" ] && echo "platform profile: max-power" ||
  echo "warning: platform profile is '$profile', not max-power"
grep -qx 1 /sys/class/power_supply/ADP*/online 2>/dev/null && echo "power: on AC" ||
  echo "warning: not on AC power"
nvidia-smi --query-gpu=name,memory.used,memory.total,power.limit,pstate --format=csv,noheader |
  sed 's/^/GPU: /'
git diff --quiet HEAD -- src bench Makefile && echo "code: committed ($(git rev-parse --short HEAD))" ||
  echo "warning: uncommitted changes in src/, bench/ or Makefile: machine.json will say -dirty"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg) (close browsers, chat apps, IDE indexers)"
echo "runs per case: $RUNS"

step "2/5 build ./tree (CPU + CUDA) and ./tree_cpu, the memory launcher and the Python venv"
make tree tree_cpu
mkdir -p bench/.tools
[ -x bench/.tools/rusage ] ||
  cc -O2 -static -Wall -o bench/.tools/rusage bench/adapters/rusage.c 2>/dev/null ||
  cc -O2 -Wall -o bench/.tools/rusage bench/adapters/rusage.c
[ -x "$PY" ] || python3 -m venv bench/.venv
"$PY" -m pip install -q -r bench/requirements.txt

step "3/5 prepare the data if it is missing"
"$PY" bench/prepare.py susy

step "4/5 benchmark: SUSY, pruned CART and C4.5, serial, parallel and CUDA, $RUNS run(s) per case"
# ./tree_cpu --serial on CPU 4; ./tree_cpu --parallel and ./tree --cuda with 20 threads.
"$PY" bench/run.py susy --protocols cart_alpha,c45 --impls tree,tree_cuda --threads 1,all \
  --cpus "$CPUS" --reps "$RUNS" --timeout 3600 --run-id "$RUN"

step "5/5 tables, CSV files and the charts"
"$PY" bench/report.py "$HERE" --md "$HERE/report.md" --csv "$HERE/summary.csv" \
  --runs-csv "$HERE/runs.csv"
"$PY" bench/chart.py "$HERE"
echo "done: $HERE/report.md, summary.csv, runs.csv, chart.png"

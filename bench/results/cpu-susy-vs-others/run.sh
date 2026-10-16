#!/usr/bin/env bash
# The SUSY benchmark against other implementations: ./tree vs scikit-learn and
# rpart (CART) and Weka J48 and YaDT (C4.5), on one thread and on every CPU of
# three machines. Each machine writes into its own directory here; BENCHMARK.md
# explains every choice.
#
#   run.sh pc   [-m N]   the desktop (i7-14700KF): runs here
#   run.sh epyc [-m N]   mordor3's epyc1 (2x EPYC 9255): on the login node, submits a SLURM job
#   run.sh dell [-m N]   mordor3's dell1 (2x Xeon Gold 6542Y): the same
#
# -m: runs per case (default 1). Every run measures training time, peak memory,
# tree size and accuracy in the same process. Then bench/vs_others_chart.py draws
# all machines in one chart.

set -euo pipefail
MACHINE="${1:-}"
shift || true
RUNS=1
while getopts "m:h" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    *) sed -n '2,12p' "$0"; exit 2 ;;
  esac
done
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }
case "$MACHINE" in
  pc|epyc|dell) ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac

# A SLURM job runs a copy of this script: find the directory from where it was submitted.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  HERE="$SLURM_SUBMIT_DIR/bench/results/cpu-susy-vs-others"
else
  HERE="$(cd "$(dirname "$0")" && pwd)"
fi
RUN="$(basename "$HERE")/$MACHINE"
OUT="$HERE/$MACHINE"
SCRIPT="$HERE/run.sh"
cd "$HERE/../../.."  # repository root
PY=bench/.venv/bin/python
step() { echo; echo "=== $(date +%H:%M:%S)  $*"; }

# On mordor3 R, rpart and a JDK 17 come from a micromamba environment in the home
# directory (the nodes have no R): see BENCHMARK.md.
if [ "$MACHINE" != pc ]; then
  BENCH_ENV="${BENCH_ENV:-$HOME/micromamba/envs/dtbench}"
  [ -x "$BENCH_ENV/bin/Rscript" ] || { echo "$BENCH_ENV: no R there (BENCHMARK.md)" >&2; exit 1; }
  export PATH="$BENCH_ENV/bin:$PATH"
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
  # run.py appends to results.jsonl: never mix two runs in one directory.
  if [ -e "$OUT/results.jsonl" ] || [ -e "$OUT/warmup-check" ]; then
    echo "$OUT already has results: move them away before running again" >&2
    exit 1
  fi
  git diff --quiet HEAD -- src bench Makefile && echo "code: committed ($(git rev-parse --short HEAD))" ||
    echo "warning: uncommitted changes in src/, bench/ or Makefile: machine.json will say -dirty"
fi

if [ "$MACHINE" != pc ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  # The login node builds the tools once (light work), then a job takes the
  # whole node (--exclusive) and runs this script again there.
  step "build ./tree_cpu, the J48 adapter and the memory launcher; check the tools"
  bench/setup.sh
  mkdir -p "$OUT"
  partition="$MACHINE" node="${MACHINE}1"
  limit=12:00:00  # the dell partition's maximum
  job="$(sbatch --parsable -J "dt-vs-others-$MACHINE" -p "$partition" -w "$node" --exclusive \
    -t "$limit" -o "$OUT/run.log" \
    --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty --abbrev=12)" \
    "$SCRIPT" "$MACHINE" -m "$RUNS")"
  echo "$MACHINE: job $job on $node, log $OUT/run.log"
  exit 0
fi

if [ "$MACHINE" = pc ]; then
  # Keep the PC from sleeping or shutting down when idle (KDE's "when inactive"
  # action) until the run ends, and log everything to run.log.
  if [ -z "${BENCH_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export BENCH_INHIBITED=1
    exec systemd-inhibit --what=idle:sleep:shutdown --who="bench $RUN" \
      --why="benchmark running" "$SCRIPT" "$MACHINE" -m "$RUNS"
  fi
  mkdir -p "$OUT"
  exec > >(trap "" INT; tee -a "$OUT/run.log") 2>&1
  # 1 thread on CPU 2 (a performance core); 28 threads on CPUs 0-27.
  CPUS=2,0,1,3-27
else
  # Every CPU of the node, in its own order (socket 0 first); 1 thread on CPU 0.
  CPUS="0-$(($(nproc) - 1))"
fi
FIRST="${CPUS%%[,-]*}"
export PYTHONUNBUFFERED=1
echo "log: $OUT/run.log, $(hostname)${SLURM_JOB_ID:+, job $SLURM_JOB_ID}, started $(date '+%F %T')"

step "1/5 machine check"
governors="$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')"
[ "$governors" = "performance " ] && echo "CPU governor: performance" ||
  echo "warning: CPU governor is '$governors', not performance"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "CPUs: $CPUS ($(nproc) available), 1 thread on CPU $FIRST"
echo "runs per case: $RUNS"

if [ "$MACHINE" = pc ]; then
  step "2/5 build ./tree_cpu, the J48 adapter and the memory launcher; check the tools"
  bench/setup.sh
else
  step "2/5 tools (built on the login node)"
  echo "code: $TREE_GIT_DESCRIBE"
fi

step "3/5 warm-up check: is a 50,000-row warm-up enough for J48's JIT? (~5 min)"
# J48 on the 500k-row SUSY subset, 3 runs after a 50k-row warm-up and 3 after a
# full one. If the 50k warm-up leaves J48 more than 3% slower, the benchmark
# warms up on all rows instead (for all managed runtimes, so the rule stays the same).
for warmup in 50000 all; do
  $PY bench/run.py susy_500k --protocols c45 --impls j48 --threads 1 --cpus "$FIRST" --reps 3 \
    --no-baseline --warmup-rows "$warmup" --java-heap 15g --run-id "$RUN/warmup-check/$warmup"
done
WARMUP="$($PY - "$OUT/warmup-check" <<'EOF'
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

step "4/5 benchmark: SUSY, pruned CART and C4.5, 1 and all threads, $RUNS run(s) per case"
# J48 gets the same 15 GB heap everywhere: the default (half the RAM) would be
# 250-500 GB on the servers, and the JVM grows its heap before it collects.
$PY bench/run.py susy --protocols cart_alpha,c45 --threads 1,all --cpus "$CPUS" \
  --reps "$RUNS" --timeout 3600 --warmup-rows "$WARMUP" --java-heap 15g --run-id "$RUN"

step "5/5 tables and CSV files"
$PY bench/report.py "$OUT" --md "$OUT/report.md" --csv "$OUT/summary.csv" --runs-csv "$OUT/runs.csv"
echo "done $(date '+%F %T'): $OUT"
echo "when all machines are done: bench/.venv/bin/python bench/vs_others_chart.py $HERE"

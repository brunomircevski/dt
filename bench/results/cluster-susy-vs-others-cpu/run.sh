#!/usr/bin/env bash
# The SUSY benchmark against other implementations: ./tree vs scikit-learn and
# rpart (CART) and Weka J48 and YaDT (C4.5), on one thread and on every CPU of
# three machines. Each machine writes into its own directory here; BENCHMARK.md
# explains every choice.
#
#   run.sh pc   [-m N]   the desktop (i7-14700KF): runs here
#   run.sh epyc [-m N]   mordor3's epyc1 (2x EPYC 9255): on the login node, submits a SLURM job
#   run.sh dell [-m N]   mordor3's dell1 (2x Xeon Gold 6542Y): the same
#   run.sh pc -t -m 3    ./tree only, when only ./tree changed (the desktop since 2026-10-16)
#   run.sh epyc -s -m 3  ./tree only, parallel on one socket (its CPUs and its NUMA
#                        nodes' memory; the one with the most free memory) -> epyc-socket/
#
# -m: runs per case (default 1). -t: run ./tree alone into <machine>/tree-rerun,
# then replace the ./tree rows of <machine>/results.jsonl with its rows and keep
# the other tools' rows (machine.json -> tree_rerun records it). Every run
# measures training time, peak memory, tree size and accuracy in the same
# process. Then bench/vs_others_chart.py draws all machines in one chart.

set -euo pipefail
MACHINE="${1:-}"
shift || true
RUNS=1
TREE_ONLY=
SOCKET=
while getopts "m:tsh" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    t) TREE_ONLY=-t ;;
    s) SOCKET=-s ;;
    *) sed -n '2,18p' "$0"; exit 2 ;;
  esac
done
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }
case "$MACHINE" in
  pc|epyc|dell) ;;
  *) sed -n '2,18p' "$0"; exit 2 ;;
esac
[ -z "$SOCKET" ] || [ "$MACHINE" != pc ] || { echo "-s: epyc or dell (two sockets)" >&2; exit 2; }
[ -z "$SOCKET" ] || [ -z "$TREE_ONLY" ] || { echo "-s or -t, not both" >&2; exit 2; }

# A SLURM job runs a copy of this script: find the directory from where it was submitted.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  HERE="$SLURM_SUBMIT_DIR/bench/results/cluster-susy-vs-others-cpu"
else
  HERE="$(cd "$(dirname "$0")" && pwd)"
fi
RUN="$(basename "$HERE")/$MACHINE${SOCKET:+-socket}"
OUT="$HERE/$MACHINE${SOCKET:+-socket}"
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
  if [ -z "$TREE_ONLY" ] && [ -e "$OUT/results.jsonl" ]; then
    echo "$OUT already has results: move them away before running again" >&2
    exit 1
  fi
  if [ -n "$TREE_ONLY" ] && { [ ! -e "$OUT/results.jsonl" ] || [ -e "$OUT/tree-rerun" ]; }; then
    echo "-t needs $OUT/results.jsonl and no $OUT/tree-rerun" >&2
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
  job="$(sbatch --parsable -J "dt-vs-others-$MACHINE${SOCKET:+-socket}" -p "$partition" -w "$node" --exclusive \
    -t "$limit" -o "$OUT/run.log" \
    --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty --abbrev=12)" \
    "$SCRIPT" "$MACHINE" -m "$RUNS" $TREE_ONLY $SOCKET)"
  echo "$MACHINE: job $job on $node, log $OUT/run.log"
  exit 0
fi

if [ "$MACHINE" = pc ]; then
  # Keep the PC from sleeping or shutting down when idle (KDE's "when inactive"
  # action) until the run ends, and log everything to run.log.
  if [ -z "${BENCH_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export BENCH_INHIBITED=1
    exec systemd-inhibit --what=idle:sleep:shutdown --who="bench $RUN" \
      --why="benchmark running" "$SCRIPT" "$MACHINE" -m "$RUNS" $TREE_ONLY
  fi
  mkdir -p "$OUT"
  exec > >(trap "" INT; tee -a "$OUT/run.log") 2>&1
  # 1 thread on CPU 2 (a performance core); 28 threads on CPUs 0-27.
  CPUS=2,0,1,3-27
elif [ -n "$SOCKET" ]; then
  # One socket, the one with the most free memory: its CPUs, and memory only
  # from its NUMA nodes.
  free_mb() { numactl -H | awk -v n="$1" '$1 == "node" && $2 == n && $3 == "free:" { print $4 }'; }
  socket_nodes() { lscpu -p=NODE,SOCKET | grep -v '^#' | awk -F, -v s="$1" '$2 == s { print $1 }' | sort -un; }
  socket="$(for s in $(lscpu -p=SOCKET | grep -v '^#' | sort -un); do
    echo "$(for n in $(socket_nodes "$s"); do free_mb "$n"; done | awk '{ t += $1 } END { print t }') $s"
  done | sort -rn | head -1 | cut -d' ' -f2)"
  CPUS="$(lscpu -p=CPU,SOCKET | grep -v '^#' | awk -F, -v s="$socket" '$2 == s { print $1 }' | paste -sd,)"
  NODES="$(socket_nodes "$socket" | paste -sd,)"
else
  # Every CPU of the node, in its own order (socket 0 first); 1 thread on CPU 0.
  CPUS="0-$(($(nproc) - 1))"
fi
FIRST="${CPUS%%[,-]*}"  # the 1-thread runs (run.py pins them to the first CPU listed)
export PYTHONUNBUFFERED=1
echo "log: $OUT/run.log, $(uname -n)${SLURM_JOB_ID:+, job $SLURM_JOB_ID}, started $(date '+%F %T')"

step "1/4 machine check"
governors="$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')"
[ "$governors" = "performance " ] && echo "CPU governor: performance" ||
  echo "warning: CPU governor is '$governors', not performance"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "CPUs: $CPUS ($(nproc) available), 1 thread on CPU $FIRST"
[ -z "$SOCKET" ] || echo "socket $socket: CPUs $CPUS, memory bound to its NUMA nodes $NODES"
echo "runs per case: $RUNS"

if [ "$MACHINE" = pc ]; then
  step "2/4 build ./tree_cpu, the J48 adapter and the memory launcher; check the tools"
  bench/setup.sh
else
  step "2/4 tools (built on the login node)"
  echo "code: $TREE_GIT_DESCRIBE"
fi

step "3/4 benchmark: full SUSY, pruned CART and C4.5, ${SOCKET:+the threads of one socket}${SOCKET:-1 and all threads}, $RUNS run(s) per case${TREE_ONLY:+, ./tree only}${SOCKET:+, ./tree only}"
# Full SUSY only: no subset runs anywhere, not even a pilot. scikit-learn, rpart
# and J48 fit one untimed tree on the first 50,000 training rows first (code
# loading, JIT); a warm-up on all rows would add a whole tree to J48's peak RSS.
# CPU only: the implementations are named, so ./tree's CUDA build never runs.
IMPLS=tree,sklearn,rpart,j48,yadt
[ -z "$TREE_ONLY$SOCKET" ] || IMPLS=tree
THREADS=1,all; MEMBIND=()
[ -z "$SOCKET" ] || { THREADS=all; MEMBIND=(--membind "$NODES"); }
# J48 gets the same 15 GB heap everywhere: the default (half the RAM) would be
# 250-500 GB on the servers, and the JVM grows its heap before it collects.
$PY bench/run.py susy --protocols cart_alpha,c45 --threads "$THREADS" --cpus "$CPUS" ${MEMBIND[@]+"${MEMBIND[@]}"} \
  --reps "$RUNS" --timeout 3600 --warmup-rows 50000 --java-heap 15g \
  --impls "$IMPLS" --run-id "$RUN${TREE_ONLY:+/tree-rerun}"

if [ -n "$TREE_ONLY" ]; then
  # The ./tree rows (runs and footprint) of the rerun replace the old ones.
  $PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rerun = out / "tree-rerun"
old = [json.loads(line) for line in open(out / "results.jsonl")]
new = [json.loads(line) for line in open(rerun / "results.jsonl")]
with open(out / "results.jsonl", "w") as f:
    for row in [r for r in old if r["impl"] != "tree"] + new:
        f.write(json.dumps(row) + "\n")
machine = json.load(open(out / "machine.json"))
again = json.load(open(rerun / "machine.json"))
replaced = machine.get("tree_rerun", {}).get("tree_git") or machine["versions"]["tree_git"]
machine["tree_rerun"] = {
    "note": "./tree rows rerun on the newer code; the other tools' rows are from the first run",
    "started": again["started"], "command": again["command"],
    "tree_git": again["versions"]["tree_git"], "loadavg_at_start": again["loadavg_at_start"],
    "governors": again["governors"], "reps": again["settings"]["reps"],
    "replaced_tree_git": replaced}
json.dump(machine, open(out / "machine.json", "w"), indent=1)
print(f"replaced the ./tree rows: {sum(r['impl'] == 'tree' for r in old)} old, {len(new)} new")
EOF
  rm -r "$OUT/tree-rerun"
fi

step "4/4 tables and CSV files"
$PY bench/report.py "$OUT" --md "$OUT/report.md" --csv "$OUT/summary.csv" --runs-csv "$OUT/runs.csv"
echo "done $(date '+%F %T'): $OUT"
echo "when all machines are done: bench/.venv/bin/python bench/vs_others_chart.py $HERE"

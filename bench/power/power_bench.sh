#!/usr/bin/env bash
# Power benchmark job: ./tree's five backends train one tree (depth <= 10, no
# pruning) while bench/power/pwrlog.py samples every power source a user can
# read (bench/power/README.md). Submit from the repository root, one job per
# node, with the whole node:
#
#   sbatch -p ampere -w ampere --exclusive -t 1:30:00 -J pwr-ampere \
#     -o bench/results/power/ampere/job.log bench/power/power_bench.sh gpu bench/results/power/ampere
#   sbatch -p dell -w dell4 --exclusive -t 8:00:00 -J pwr-dell \
#     -o bench/results/power/dell/job.log bench/power/power_bench.sh cpu bench/results/power/dell
#
# gpu: Hybrid (--cuda) and GPU (--cuda --gpu-only) on GPU 0 with the 24 CPUs and
#      the memory of its NUMA node, as in the cluster benchmarks; RAPL is read with
#      perf (ampere allows it to users).
# cpu: Serial-Simple (--simple), Serial-Opt (--serial) and Parallel on all cores.
# all: all five on one node with readable CPU (RAPL) and GPU power (ampere), so
#      that every implementation gets the same CPU/GPU split:
#   sbatch -p ampere -w ampere --exclusive -c 48 -t 9:00:00 -J pwr-all \
#     --export=ALL,TARGET=30,REPS=5,REPS_BY="simple=1 serial=3" \
#     -o bench/results/power-ampere/ampere/job.log bench/power/power_bench.sh all bench/results/power-ampere/ampere
#
# The run: a warm-up run of each fast backend (Parallel, Hybrid, GPU) per
# algorithm (two trees), which also picks its repeat count K = ceil(TARGET /
# the second tree's training time),
# so that its training window lasts >= TARGET seconds (the PSU meter averages
# over 2-6 s); then IDLE seconds of idle; then REPS rounds of every case
# (round-robin), each process preceded by PRE seconds of idle; then IDLE seconds
# of idle again. ./tree --repeat K prints every tree's training window as Unix
# times; bench/power/analyze.py integrates the samples over those windows.
#
# Environment: DATA (CSV), REPS (3), REPS_BY (per-implementation rounds,
# e.g. "simple=1 serial=3"; the rest run REPS rounds), TARGET (120), IDLE (60),
# PRE (20), ALGOS ("cart c45"), IMPLS (gpu: "cuda gpuonly", cpu: "simple serial
# parallel", all: all five).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
KIND=$1
OUT=$2
DATA=${DATA:-bench/data/ecbdl_15m/train.csv}
REPS=${REPS:-3}
TARGET=${TARGET:-120}
IDLE=${IDLE:-60}
PRE=${PRE:-20}
ALGOS=${ALGOS:-cart c45}
PROTOCOL=(--no-prune -d 10)
mkdir -p "$OUT/out"
now() { date +%s.%N | cut -c1-17; }
mark() { echo "$(now) $*" >> "$OUT/markers.txt"; }

case $KIND in
  gpu) IMPLS=${IMPLS:-cuda gpuonly}; FAST="cuda gpuonly" ;;
  cpu) IMPLS=${IMPLS:-simple serial parallel}; FAST="parallel" ;;
  all) IMPLS=${IMPLS:-simple serial parallel cuda gpuonly}; FAST="parallel cuda gpuonly" ;;
  *) echo "usage: $0 gpu|cpu|all OUTDIR" >&2; exit 2 ;;
esac
declare -A NREPS
MAXREP=$REPS
for impl in $IMPLS; do NREPS[$impl]=$REPS; done
for kv in ${REPS_BY:-}; do
  NREPS[${kv%%=*}]=${kv#*=}
  [ "${kv#*=}" -gt "$MAXREP" ] && MAXREP=${kv#*=}
done

command_of() { # impl -> the command line before the algorithm flags
  case $1 in
    cuda) echo "env CUDA_VISIBLE_DEVICES=0 numactl --membind=0 taskset -c 0-23 ./tree --cuda --threads 24" ;;
    gpuonly) echo "env CUDA_VISIBLE_DEVICES=0 numactl --membind=0 taskset -c 0-23 ./tree --cuda --gpu-only --threads 24" ;;
    # --threads 0: loading and evaluation use every core; training is one thread
    simple) echo "./tree_cpu --simple --threads 0" ;;
    serial) echo "./tree_cpu --serial --threads 0" ;;
    parallel) echo "./tree_cpu --parallel --threads 0" ;;
  esac
}

{
  echo "job ${SLURM_JOB_ID:-none} on $(hostname), $(date -Is), kind $KIND"
  echo "commit ${TREE_GIT_DESCRIBE:-$(git describe --always --dirty 2>/dev/null)}"
  echo "data $DATA, reps $REPS ${REPS_BY:-}, target $TARGET s, idle $IDLE s, pre $PRE s, algos $ALGOS, impls $IMPLS"
  echo "protocol ${PROTOCOL[*]}"
  echo "kernel $(uname -r)"
  echo "dmi $(cat /sys/class/dmi/id/sys_vendor) $(cat /sys/class/dmi/id/product_name)"
  lscpu | grep -E 'Model name|Socket|Core\(s\)|NUMA node\(s\)'
  echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver) $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  echo "cpus available $(nproc)"
  echo "perf_event_paranoid $(cat /proc/sys/kernel/perf_event_paranoid)"
  for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$h/name")" = power_meter ] && for f in "$h"/power1_*; do echo "$(basename "$f") $(cat "$f")"; done
  done
  free -g | head -2
  command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=index,name,power.limit,driver_version --format=csv
} > "$OUT/machine.txt" 2>&1

python3 bench/power/pwrlog.py 0.1 "$OUT/samples.csv" &
SAMPLER=$!
PERF=
if [ "$KIND" != cpu ] && [ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ]; then
  mark perf_start
  perf stat -a --per-socket -x, -I 100 -e power/energy-pkg/ -o "$OUT/perf_rapl.csv" &
  PERF=$!
fi
cleanup() {
  kill "$SAMPLER" 2> /dev/null
  [ -n "$PERF" ] && kill -INT "$PERF" 2> /dev/null
  wait 2> /dev/null
}
trap cleanup EXIT
trap 'exit 143' TERM INT

run() { # id impl algo repeat
  local id=$1 impl=$2 algo=$3 k=$4
  mark "start $id impl=$impl algo=$algo repeat=$k"
  # shellcheck disable=SC2046
  timeout --foreground 3h $(command_of "$impl") "--$algo" "${PROTOCOL[@]}" --repeat "$k" "$DATA" \
    > "$OUT/out/$id.txt" 2>&1
  mark "end $id rc=$?"
}

declare -A REPEAT
mark job_start
for algo in $ALGOS; do
  for impl in $IMPLS; do
    REPEAT[$impl-$algo]=1
    case " $FAST " in *" $impl "*) ;; *) continue ;; esac
    # two trees: the first one in a process is slower (CUDA context, allocations)
    run "warmup-$impl-$algo" "$impl" "$algo" 2
    ms=$(sed -n 's/^  repeat 2\/2: train \([0-9.]*\) ms.*/\1/p' "$OUT/out/warmup-$impl-$algo.txt")
    if [ -n "$ms" ]; then
      REPEAT[$impl-$algo]=$(python3 -c "import math; print(max(1, math.ceil($TARGET / ($ms / 1000))))")
    fi
    echo "repeat $impl $algo ${REPEAT[$impl-$algo]} (warm-up train $ms ms)" >> "$OUT/machine.txt"
  done
done

sleep "$PRE"
mark idle_start
sleep "$IDLE"
mark idle_end
for rep in $(seq 1 "$MAXREP"); do
  for algo in $ALGOS; do
    for impl in $IMPLS; do
      [ "$rep" -le "${NREPS[$impl]}" ] || continue
      sleep "$PRE"
      run "rep$rep-$impl-$algo" "$impl" "$algo" "${REPEAT[$impl-$algo]}"
    done
  done
done
sleep "$PRE"
mark idle2_start
sleep "$IDLE"
mark idle2_end
mark job_end

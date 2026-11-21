#!/usr/bin/env bash
# Power benchmark job: ./tree's five backends train one tree (ECBDL'14 15M,
# depth <= 10, no pruning) while bench/tools/power/pwrlog.py samples every power
# source a user can read, every 0.1 s (bench/cluster-power/BENCHMARK.md). One job per
# node, with the whole node, from the repository root (submit.sh-style:
# held, then released, since sbatch --export can leave a job held):
#
#   bench/tools/power/submit.sh epyc1 hopper
#
# KIND (first argument): cpu = Serial-Simple (--simple), Serial-Opt (--serial)
# and Multicore (--multicore); all = those three plus GPU (--gpu) and Hybrid
# (--hybrid). OUT (second argument): the job's folder.
#
# The problem it solves: the whole-server (PSU) meters refresh every 1 s
# (hopper) or 4-10 s (epyc), while a tree takes 2.4 s on a GPU. So the job records, next to the
# meter, the fast sensors (RAPL package and DRAM per socket, every GPU's NVML
# energy counter; all read every 0.1 s), and gives the meter a wide range of
# loads to calibrate against: analyze.py fits the meter as a delayed moving
# average of  a + b * (CPU packages + DRAM) + c * (all GPUs)  over every
# reading of the job, and integrates that model over each tree's window.
#
# The job:
#   1. warm-up: each fast backend (Multicore, GPU, Hybrid) trains two trees
#      per algorithm; the second one's time sets K = ceil(TARGET / t) + 1
#      trees per measured process;
#   2. (GPU nodes, CHECK=1) the warm-up check: GPU and Hybrid, CART,
#      CHECK_TREES trees at each --repeat-warmup in CHECK_WARMUPS;
#   3. IDLE s idle; the calibration staircase: 1, 2, 4, ... and all CPUs
#      busy (spin loops), CAL s each; IDLE s idle;
#   4. REPS rounds of every case (round-robin), PRE s idle before each process;
#   5. IDLE s idle.
# ./tree --repeat K trains every tree in its own forked child (each tree is a
# fresh process's only tree) and prints its training window as Unix times; the
# GPU backends add --repeat-warmup WARMUP: the child keeps its CPU threads busy
# for WARMUP s before its timer, as a single run's data load does (without it, a
# GPU tree after idle CPUs set up 10-40% slower on hopper). 2 s reproduces the
# time benchmark's single runs on the final build (job 549: 0, 2, 5, 15 s).
#
# Placement as in the time benchmark (bench/tools/cluster/job.sh and bench/tools/run.py):
# Serial-Simple and Serial-Opt on CPU 0 (taskset; the load then has one
# thread too); Multicore on every CPU; GPU and Hybrid on the GPU of the NUMA node
# with the most free memory, with every CPU of the node (that node's first)
# and first-touch memory.
#
# Environment: DATA (.f32 or CSV; default ECBDL'14's train.f32 with ROWS=15000000, the
# time benchmark's ecbdl_15m), REPS (3), REPS_BY (per backend; default "simple=10
# serial=10": one tree per process, as many as the time benchmark's runs), TARGET (120 s of training per measured process), IDLE
# (300), CAL (60 s per staircase level; 0: no staircase), PRE (20), WARMUP (2),
# CHECK (0; 1: the warm-up check inside the job), CHECK_ONLY (1: the warm-up
# check alone), CHECK_WARMUPS ("0 2 5 15"), CHECK_TREES (20), ALGOS ("cart c45"), IMPLS (cpu: "simple serial parallel", all: all
# five), BIN (the folder of ./tree and ./tree_cpu; default the repository root),
# DEPTH (10; -1: no depth limit), HANDOFF (--hybrid's --handoff-rows; default
# ./tree's own rule; benchmarks take bench/tools/handoff/handoff_rows.json's),
# SHM (1: copy the .f32 dataset to /dev/shm first, removed at the end: ./tree
# drops its input from the page cache, so every process would read it again
# from NFS).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
KIND=$1
OUT=$2
if [ -z "${DATA:-}" ]; then  # ecbdl_15m: a prefix of ecbdl (bench/tools/datasets.toml)
  DATA=bench/tools/data/ecbdl/train.f32 ROWS=${ROWS:-15000000}
fi
ROWS=${ROWS:-}
REPS=${REPS:-3}
REPS_BY=${REPS_BY:-simple=10 serial=10}
TARGET=${TARGET:-120}
IDLE=${IDLE:-300}
CAL=${CAL:-60}
PRE=${PRE:-20}
WARMUP=${WARMUP:-2}
CHECK_ONLY=${CHECK_ONLY:-}
CHECK=${CHECK:-${CHECK_ONLY:-0}}
CHECK_WARMUPS=${CHECK_WARMUPS:-0 2 5 15}
ALGOS=${ALGOS:-cart c45}
BIN=${BIN:-.}
DEPTH=${DEPTH:-10}
PROTOCOL=(-d "$DEPTH")
HANDOFF=${HANDOFF:-}
mkdir -p "$OUT/out"
SHM_DIR=
if [ "${SHM:-}" = 1 ]; then  # X.f32, X.y.i32 and X.meta.json or meta.json, as ./tree reads them
  SHM_DIR=/dev/shm/power-bench-${SLURM_JOB_ID:-$$}
  stem=${DATA%.f32} meta=${DATA%.f32}.meta.json
  [ -e "$meta" ] || meta=$(dirname "$DATA")/meta.json
  mkdir -p "$SHM_DIR" && cp "$DATA" "$stem.y.i32" "$SHM_DIR/" && cp "$meta" "$SHM_DIR/$(basename "$stem").meta.json" ||
    { echo "copy to $SHM_DIR failed" >&2; rm -rf "$SHM_DIR"; exit 1; }
  DATA_SRC=$DATA DATA=$SHM_DIR/$(basename "$DATA")
fi
now() { date +%s.%N | cut -c1-17; }
mark() { echo "$(now) $*" >> "$OUT/markers.txt"; }

case $KIND in
  cpu) IMPLS=${IMPLS:-simple serial parallel}; FAST="parallel" ;;
  all) IMPLS=${IMPLS:-simple serial parallel gpuonly cuda}; FAST="parallel gpuonly cuda" ;;
  *) echo "usage: $0 cpu|all OUTDIR" >&2; exit 2 ;;
esac
declare -A NREPS
MAXREP=$REPS
for impl in $IMPLS; do NREPS[$impl]=$REPS; done
for kv in $REPS_BY; do
  NREPS[${kv%%=*}]=${kv#*=}
  [ "${kv#*=}" -gt "$MAXREP" ] && MAXREP=${kv#*=}
done

NCPU=$(nproc)
ALL_CPUS="0-$((NCPU - 1))"
# The GPU: as bench/tools/cluster/job.sh, a GPU with the most memory, on the NUMA node
# with the most free memory of those; its node's CPUs first, then the rest. A GPU
# under 80 GB stops the job (GPU_MIN_MIB, default 80000).
# Each GPU node's fixed GPU (the paper's placement; free memory alone is a tie
# on hopper and picked GPU 0 or 2 by chance): hopper GPU 2 on NUMA node 1.
# Another GPU stops the job unless
# GPU_ANY=1 (tuning only).
case "$(hostname -s)" in
  hopper) GPU_EXPECTED=2 GPU_NUMA=1 ;;
  *) GPU_EXPECTED= GPU_NUMA= ;;
esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # CUDA's device numbers = nvidia-smi's
GPU= GPU_NODE= GPU_CPUS=
if [ "$KIND" = all ]; then
  GPU_NODE="$GPU_NUMA"
  [ -n "$GPU_NODE" ] || GPU_NODE="$(nvidia-smi --query-gpu=memory.total,pci.bus_id --format=csv,noheader,nounits | tr A-Z a-z |
    while IFS=', ' read -r total bus; do
      node="$(cat "/sys/bus/pci/devices/${bus: -12}/numa_node")"
      echo "$total $(numactl -H | awk -v n="$node" '$1 == "node" && $2 == n && $3 == "free:" { print $4 }') $node"
    done | sort -k1,1nr -k2,2nr | head -1 | cut -d' ' -f3)"
  GPU="$(nvidia-smi --query-gpu=index,memory.total,pci.bus_id --format=csv,noheader,nounits | tr A-Z a-z |
    while IFS=', ' read -r index total bus; do
      [ "$(cat "/sys/bus/pci/devices/${bus: -12}/numa_node")" = "$GPU_NODE" ] && echo "$total $index"
    done | sort -s -k1,1nr | head -1 | cut -d' ' -f2 || true)"
  [ -n "$GPU" ] || { echo "no GPU found" >&2; exit 1; }
  gpu_mib="$(nvidia-smi --id="$GPU" --query-gpu=memory.total --format=csv,noheader,nounits)"
  [ "${gpu_mib:-0}" -ge "${GPU_MIN_MIB:-80000}" ] ||
    { echo "GPU $GPU has ${gpu_mib:-?} MiB, under ${GPU_MIN_MIB:-80000}: not an 80 GB card" >&2; exit 1; }
  [ -z "$GPU_EXPECTED" ] || [ "$GPU" = "$GPU_EXPECTED" ] || [ -n "${GPU_ANY:-}" ] ||
    { echo "GPU $GPU on $(hostname -s), not its fixed GPU $GPU_EXPECTED (GPU_ANY=1 allows it)" >&2; exit 1; }
  GPU_CPUS="$(cat "/sys/devices/system/node/node$GPU_NODE/cpulist")$(
    for n in /sys/devices/system/node/node[0-9]*; do
      [ "${n##*node}" = "$GPU_NODE" ] || printf ,%s "$(cat "$n/cpulist")"
    done)"
fi

command_of() { # impl -> the command line before the algorithm flags
  case $1 in
    simple) echo "taskset -c 0 $BIN/tree_cpu --simple" ;;
    serial) echo "taskset -c 0 $BIN/tree_cpu --serial" ;;
    parallel) echo "taskset -c $ALL_CPUS $BIN/tree_cpu --multicore --threads $NCPU" ;;
    gpuonly) echo "env CUDA_VISIBLE_DEVICES=$GPU taskset -c $GPU_CPUS $BIN/tree --gpu --threads $NCPU" ;;
    cuda) echo "env CUDA_VISIBLE_DEVICES=$GPU taskset -c $GPU_CPUS $BIN/tree --hybrid --threads $NCPU${HANDOFF:+ --handoff-rows $HANDOFF}" ;;
  esac
}
warmup_of() { case $1 in gpuonly|cuda) echo "--repeat-warmup ${2:-$WARMUP}" ;; esac; }

{
  echo "job ${SLURM_JOB_ID:-none} on $(hostname), $(date -Is), kind $KIND"
  echo "commit ${TREE_GIT_DESCRIBE:-$(git describe --always --dirty 2>/dev/null)}"
  echo "data ${DATA_SRC:-$DATA}${SHM_DIR:+ (copied to $DATA)}${ROWS:+ (first $ROWS rows)}, reps $REPS $REPS_BY, target $TARGET s, idle $IDLE s, cal $CAL s, pre $PRE s, warm-up $WARMUP s, algos $ALGOS, impls $IMPLS"
  [ -z "$GPU" ] || echo "used_gpu $GPU (NUMA node $GPU_NODE, cpus $GPU_CPUS, $NCPU threads, first-touch memory)"
  echo "protocol ${PROTOCOL[*]}"
  for impl in $IMPLS; do echo "command $impl $(command_of "$impl") $(warmup_of "$impl")"; done
  echo "kernel $(uname -r)"
  echo "dmi $(cat /sys/class/dmi/id/sys_vendor) $(cat /sys/class/dmi/id/product_name)"
  lscpu | grep -E 'Model name|Socket|Core\(s\)|NUMA node\(s\)'
  echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver) $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  echo "cpus available $NCPU"
  for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$h/name")" = power_meter ] && for f in "$h"/power1_*; do echo "$(basename "$f") $(cat "$f")"; done
  done
  for z in /sys/class/powercap/intel-rapl:*; do
    [ -r "$z/energy_uj" ] && echo "rapl_max $(basename "$z") $(cat "$z/name") $(cat "$z/max_energy_range_uj")"
  done
  [ -r /run/power/psu_W ] && echo "psu_file $(cat /run/power/psu_W)"
  free -g | head -2
  numactl -H | grep free
  if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=index,name,power.limit,driver_version,memory.used --format=csv
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/GPU process: /'
  fi
} > "$OUT/machine.txt" 2>&1

# the sampler on the last CPU: never on CPU 0 with the one-thread runs
taskset -c "$((NCPU - 1))" python3 bench/tools/power/pwrlog.py 0.1 "$OUT/samples.csv" &
SAMPLER=$!
BUSY=()
cleanup() {
  [ ${#BUSY[@]} -eq 0 ] || kill "${BUSY[@]}" 2> /dev/null
  kill "$SAMPLER" 2> /dev/null
  wait 2> /dev/null
  gzip -f "$OUT/samples.csv"
  [ -z "$SHM_DIR" ] || rm -rf "$SHM_DIR"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

run() { # id impl algo repeat [warm-up]
  local id=$1 impl=$2 algo=$3 k=$4
  mark "start $id impl=$impl algo=$algo repeat=$k"
  # shellcheck disable=SC2046
  timeout --foreground 4h $(command_of "$impl") "--$algo" "${PROTOCOL[@]}" --repeat "$k" $(warmup_of "$impl" "${5:-}") \
    ${ROWS:+--rows "$ROWS"} "$DATA" > "$OUT/out/$id.txt" 2>&1
  mark "end $id rc=$?"
}

# busy N SECONDS: spin loops on CPUs 0 .. N-1 (the calibration staircase)
busy() {
  BUSY=()
  for ((c = 0; c < $1; c++)); do
    taskset -c "$c" timeout "$2" sh -c 'while :; do :; done' & BUSY+=($!)
  done
  wait "${BUSY[@]}"  # not a bare wait: the sampler runs in the background too
  BUSY=()
}

declare -A REPEAT
mark job_start
[ -z "$CHECK_ONLY" ] || ALGOS=""
for algo in $ALGOS; do
  for impl in $IMPLS; do
    REPEAT[$impl-$algo]=1
    case " $FAST " in *" $impl "*) ;; *) continue ;; esac
    # two trees: the second one's time picks K (the data is in the page cache by then)
    run "warmup-$impl-$algo" "$impl" "$algo" 2
    ms=$(sed -n 's/^  repeat 2\/2: train \([0-9.]*\) ms.*/\1/p' "$OUT/out/warmup-$impl-$algo.txt")
    if [ -n "$ms" ]; then
      REPEAT[$impl-$algo]=$(python3 -c "import math; print(max(2, math.ceil($TARGET / ($ms / 1000))) + 1)")
    fi
    echo "repeat $impl $algo ${REPEAT[$impl-$algo]} (warm-up train $ms ms)" >> "$OUT/machine.txt"
  done
done
# The warm-up check: which WARMUP reproduces the time benchmark? (CART.)
if [ "$CHECK" = 1 ] && [ "$KIND" = all ]; then
  # one process first, so that the first check does not pay for the page cache
  [ -z "$CHECK_ONLY" ] || run "warmup-gpuonly-cart" gpuonly cart 2
  for w in $CHECK_WARMUPS; do
    for impl in gpuonly cuda; do
      sleep "$PRE"
      run "check-$impl-w$w" "$impl" cart "${CHECK_TREES:-20}" "$w"
    done
  done
fi
[ -z "$CHECK_ONLY" ] || { mark job_end; exit 0; }

sleep "$PRE"
mark idle_start
sleep "$IDLE"
mark idle_end
if [ "$CAL" -gt 0 ]; then
  levels=""
  for ((n = 1; n < NCPU; n *= 2)); do levels="$levels $n"; done
  for n in $levels $NCPU; do
    mark "cal_start n=$n"
    busy "$n" "$CAL"
    mark "cal_end n=$n"
  done
  mark step_end
  sleep "$IDLE"
  mark step_idle_end
fi
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

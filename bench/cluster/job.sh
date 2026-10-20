#!/usr/bin/env bash
# One benchmark on one whole node of mordor3, as a SLURM job; submit.sh submits
# it with the partition, --exclusive and the log file:
#
#   job.sh <results directory> <run name> <dataset> <impls> <runs per case> [protocols] [threads] [timeout]
#
# e.g. job.sh cluster-higgs cuda-hopper-higgs higgs tree,tree_cuda 3
#   -> bench/results/cluster-higgs/cuda-hopper-higgs/
# threads: run.py's --threads (default all); "scaling" = 1, 2, 4, then every
# 4 up to the node's CPU count (scaling.sh). timeout: seconds per ./tree process
# (default 3600). RUN_EXTRA (environment): more run.py arguments, e.g.
# --profile (profile.sh). NUMA_NODE (environment, submit.sh -n): the threads,
# the memory (numactl --membind) and the GPU all on that NUMA node (one
# socket): its CPUs in their order, and the first GPU attached to it instead of
# GPU 0. CUDA runs always get one (below); CPU runs only when it is given.

set -euo pipefail
DIR="$1" NAME="$2" DATASET="$3" IMPLS="$4" RUNS="$5" PROTOCOLS="${6:-cart_alpha_nodepth,c45}" THREADS="${7:-all}" TIMEOUT="${8:-3600}"
cd "$SLURM_SUBMIT_DIR"  # repository root
PY=bench/.venv/bin/python
OUT="bench/results/$DIR/$NAME"
step() { echo; echo "=== $(date +%H:%M:%S)  $*"; }
# Every temp file in one folder of the job, removed at the end with the
# datasets' page cache, also when the job fails or SLURM ends it (cleanup.sh).
export TMPDIR="$(mktemp -d "/tmp/dt-$SLURM_JOB_ID.XXXXXX")"
trap 'bench/cluster/cleanup.sh "$TMPDIR" bench/data' EXIT
trap 'exit 143' TERM INT
echo "log: $OUT/run.log, job $SLURM_JOB_ID on $(hostname), started $(date '+%F %T')"

# SLURM on mordor3 hands out cores, not GPUs (no GRES): the job owns the node
# through --exclusive, and ./tree uses GPU 0 in nvidia-smi's (PCI) order (with
# NUMA_NODE the first on that node), the GPU whose memory run.py samples.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
# CUDA runs stay on one socket, the GPU's: without NUMA_NODE they take the NUMA
# node of a GPU with the most free memory (hopper: node 1, node 0's memory was
# taken in 2026-10). Only the parallel CPU backend uses both sockets.
if [ -z "${NUMA_NODE:-}" ] && [[ "$IMPLS" == *tree_cuda* || "$IMPLS" == *tree_gpu_only* ]]; then
  NUMA_NODE="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | tr A-Z a-z |
    while read -r bus; do cat "/sys/bus/pci/devices/${bus: -12}/numa_node"; done | sort -u |
    while read -r node; do
      numactl -H | awk -v n="$node" '$1 == "node" && $2 == n && $3 == "free:" { print $4, n }'
    done | sort -rn | head -1 | cut -d' ' -f2)"
  [ -n "$NUMA_NODE" ] || { echo "no NUMA node with a GPU found" >&2; exit 1; }
fi
if [ -n "${NUMA_NODE:-}" ] &&
   [ "$(lscpu -p=NODE,SOCKET | grep -v '^#' | awk -F, -v n="$NUMA_NODE" '$1 == n { print $2 }' | sort -u | wc -l)" != 1 ]; then
  echo "NUMA node $NUMA_NODE spans more than one socket" >&2; exit 1
fi
MEMBIND=()
if [ -n "${NUMA_NODE:-}" ]; then
  MEMBIND=(--membind "$NUMA_NODE")
  # nvidia-smi's bus id (00000000:2A:00.0) ends in the sysfs one (0000:2a:00.0).
  gpu="$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | tr A-Z a-z |
    while IFS=', ' read -r index bus; do
      [ "$(cat "/sys/bus/pci/devices/${bus: -12}/numa_node")" = "$NUMA_NODE" ] && { echo "$index"; break; }
    done || true)"
  [ -n "$gpu" ] || { echo "no GPU on NUMA node $NUMA_NODE" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="$gpu"
fi
export NVCC=/usr/local/cuda/bin/nvcc LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONUNBUFFERED=1
CPUS="0-$(($(nproc) - 1))"  # every CPU of the node
[ -z "${NUMA_NODE:-}" ] || CPUS="$(cat "/sys/devices/system/node/node$NUMA_NODE/cpulist")"
[ "$THREADS" != scaling ] || THREADS="1,2,4,$(seq -s, 8 4 "$(nproc)")"

step "1/3 machine check"
echo "CPUs allocated: $(nproc) ($CPUS), SLURM_JOB_CPUS_PER_NODE=$SLURM_JOB_CPUS_PER_NODE"
governors="$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')"
echo "CPU governor: ${governors:-unknown}"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "free memory per NUMA node: $(numactl -H | awk '/ free:/ { printf "%snode %s %d GiB", (n++ ? ", " : ""), $2, $4 / 1024 }')"
[ -z "${NUMA_NODE:-}" ] || echo "NUMA node $NUMA_NODE: CPUs $CPUS, memory bound to it, GPU $CUDA_VISIBLE_DEVICES"
echo "CPU order (the threads take them from the first): CPU, NUMA node, socket, core"
lscpu -e=CPU,NODE,SOCKET,CORE | awk 'NR > 1 { printf "%s%s:%s/%s/%s", (NR > 2 ? " " : ""), $1, $2, $3, $4 } END { print "" }'
if [[ "$IMPLS" == *tree_cuda* || "$IMPLS" == *tree_gpu_only* ]]; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.limit,pstate \
    --format=csv,noheader | sed 's/^/GPU /'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
    sed 's/^/GPU process: /'
fi
echo "code: $TREE_GIT_DESCRIBE (from submit.sh: hopper has no git)"
echo "runs per case: $RUNS"

step "2/3 benchmark: $DATASET, $PROTOCOLS, $IMPLS, threads $THREADS (of $(nproc)), $RUNS run(s) per case"
"$PY" bench/run.py "$DATASET" --protocols "$PROTOCOLS" --impls "$IMPLS" --threads "$THREADS" \
  --cpus "$CPUS" ${MEMBIND[@]+"${MEMBIND[@]}"} --drop-cache --reps "$RUNS" --timeout "$TIMEOUT" --run-id "$DIR/$NAME" --train-accuracy-only \
  ${RUN_EXTRA:-}

step "3/3 tables and CSV files"
"$PY" bench/report.py "$OUT" --md "$OUT/report.md" --csv "$OUT/summary.csv" --runs-csv "$OUT/runs.csv"
echo "done $(date '+%F %T')"

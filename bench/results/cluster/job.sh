#!/usr/bin/env bash
# One benchmark on one whole node of mordor3, as a SLURM job; submit.sh submits
# it with the partition, --exclusive and the log file:
#
#   job.sh <run name> <impls> <runs per case>
#
# e.g. job.sh cuda-hopper-higgs tree,tree_cuda 3 -> bench/results/cluster/cuda-hopper-higgs/

set -euo pipefail
NAME="$1" IMPLS="$2" RUNS="$3"
cd "$SLURM_SUBMIT_DIR"  # repository root
PY=bench/.venv/bin/python
OUT="bench/results/cluster/$NAME"
step() { echo; echo "=== $(date +%H:%M:%S)  $*"; }
echo "log: $OUT/run.log, job $SLURM_JOB_ID on $(hostname), started $(date '+%F %T')"

# SLURM on mordor3 hands out cores, not GPUs (no GRES): the job owns the node
# through --exclusive, and ./tree uses GPU 0 in nvidia-smi's (PCI) order, the
# GPU whose memory run.py samples (nvidia-smi --id=0).
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export NVCC=/usr/local/cuda/bin/nvcc LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONUNBUFFERED=1
CPUS="0-$(($(nproc) - 1))"  # every core of the node (one thread per core on all four)

step "1/3 machine check"
echo "CPUs allocated: $(nproc) ($CPUS), SLURM_JOB_CPUS_PER_NODE=$SLURM_JOB_CPUS_PER_NODE"
governors="$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')"
echo "CPU governor: ${governors:-unknown}"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)"
if [[ "$IMPLS" == *tree_cuda* ]]; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.limit,pstate \
    --format=csv,noheader | sed 's/^/GPU /'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
    sed 's/^/GPU process: /'
fi
echo "code: $TREE_GIT_DESCRIBE (from submit.sh: hopper has no git)"
echo "runs per case: $RUNS"

step "2/3 benchmark: HIGGS, pruned CART and C4.5, $IMPLS, $(nproc) threads, $RUNS run(s) per case"
"$PY" bench/run.py higgs --protocols cart_alpha,c45 --impls "$IMPLS" --threads all \
  --cpus "$CPUS" --reps "$RUNS" --timeout 1800 --run-id "cluster/$NAME"

step "3/3 tables and CSV files"
"$PY" bench/report.py "$OUT" --md "$OUT/report.md" --csv "$OUT/summary.csv" --runs-csv "$OUT/runs.csv"
echo "done $(date '+%F %T')"

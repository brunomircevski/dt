#!/usr/bin/env bash
# One benchmark on one whole node of mordor3, as a SLURM job; submit.sh submits
# it with the partition, --exclusive and the log file:
#
#   job.sh <results directory> <run name> <dataset> <impls> <runs per case> [protocols] [threads]
#
# e.g. job.sh cluster-higgs cuda-hopper-higgs higgs tree,tree_cuda 3
#   -> bench/results/cluster-higgs/cuda-hopper-higgs/
# threads: run.py's --threads (default all); "scaling" = 1, 2, 4, then every
# 4 up to the node's CPU count (scaling.sh).

set -euo pipefail
DIR="$1" NAME="$2" DATASET="$3" IMPLS="$4" RUNS="$5" PROTOCOLS="${6:-cart_alpha_nodepth,c45}" THREADS="${7:-all}"
cd "$SLURM_SUBMIT_DIR"  # repository root
PY=bench/.venv/bin/python
OUT="bench/results/$DIR/$NAME"
step() { echo; echo "=== $(date +%H:%M:%S)  $*"; }
echo "log: $OUT/run.log, job $SLURM_JOB_ID on $(hostname), started $(date '+%F %T')"

# SLURM on mordor3 hands out cores, not GPUs (no GRES): the job owns the node
# through --exclusive, and ./tree uses GPU 0 in nvidia-smi's (PCI) order, the
# GPU whose memory run.py samples (nvidia-smi --id=0).
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export NVCC=/usr/local/cuda/bin/nvcc LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONUNBUFFERED=1
CPUS="0-$(($(nproc) - 1))"  # every CPU of the node (turing: 16 cores with SMT, 32 threads)
[ "$THREADS" != scaling ] || THREADS="1,2,4,$(seq -s, 8 4 "$(nproc)")"

step "1/3 machine check"
echo "CPUs allocated: $(nproc) ($CPUS), SLURM_JOB_CPUS_PER_NODE=$SLURM_JOB_CPUS_PER_NODE"
governors="$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')"
echo "CPU governor: ${governors:-unknown}"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "CPU order (the threads take them from the first): CPU, NUMA node, socket, core"
lscpu -e=CPU,NODE,SOCKET,CORE | awk 'NR > 1 { printf "%s%s:%s/%s/%s", (NR > 2 ? " " : ""), $1, $2, $3, $4 } END { print "" }'
if [[ "$IMPLS" == *tree_cuda* ]]; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.limit,pstate \
    --format=csv,noheader | sed 's/^/GPU /'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
    sed 's/^/GPU process: /'
fi
echo "code: $TREE_GIT_DESCRIBE (from submit.sh: hopper has no git)"
echo "runs per case: $RUNS"

step "2/3 benchmark: $DATASET, $PROTOCOLS, $IMPLS, threads $THREADS (of $(nproc)), $RUNS run(s) per case"
"$PY" bench/run.py "$DATASET" --protocols "$PROTOCOLS" --impls "$IMPLS" --threads "$THREADS" \
  --cpus "$CPUS" --reps "$RUNS" --timeout 3600 --run-id "$DIR/$NAME" --train-accuracy-only

step "3/3 tables and CSV files"
"$PY" bench/report.py "$OUT" --md "$OUT/report.md" --csv "$OUT/summary.csv" --runs-csv "$OUT/runs.csv"
echo "done $(date '+%F %T')"

#!/usr/bin/env bash
# SLURM job: the whole GPU comparison on one node (GPU 0), e.g.
#   sbatch -p hopper --exclusive -t 01:00:00 -o results/hopper/job.log job.sh hopper
#   IMPLS="cuml xgboost" sbatch ...   runs only those (default: tree cuml xgboost)
# Results: results/<node>/results.jsonl, test predictions in results/<node>/pred/.
#SBATCH -J gdb-bench
set -uo pipefail
NODE="$1" REPS="${REPS:-3}" IMPLS="${IMPLS:-tree cuml xgboost}"
B=$HOME/gpu-dt-bench
OUT=$B/results/$NODE
mkdir -p "$OUT/pred"
SCRATCH=$(mktemp -d /tmp/gdb.XXXXXX)  # node-local tmpfs, for the tree dumps
trap 'rm -rf "$SCRATCH"' EXIT
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
PY=$B/venv/bin/python
TREE=$B/dt/tree
echo "job ${SLURM_JOB_ID:-} on $(hostname), $(nproc) CPUs, $(date '+%F %T'), reps $REPS"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "CPU governor: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version,power.limit --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/GPU process: /'
{
  echo "{\"host\": \"$(hostname)\", \"cpu\": \"$(lscpu | sed -n 's/^Model name: *//p' | head -1)\","
  echo " \"cpus\": $(nproc), \"gpu\": \"$(nvidia-smi --id=0 --query-gpu=name,memory.total --format=csv,noheader)\","
  echo " \"driver\": \"$(nvidia-smi --id=0 --query-gpu=driver_version --format=csv,noheader)\","
  echo " \"cuda\": \"$(/usr/local/cuda/bin/nvcc --version | tail -2 | head -1)\","
  echo " \"tree_commit\": \"35cf315\", \"date\": \"$(date '+%F %T')\"}"
} > "$OUT/machine.json"

run() {  # timeout per process: a case that does not finish in 15 min is reported as such;
  # --foreground keeps the case in the job's process group, so scancel stops it
  echo; echo "=== $(date +%H:%M:%S) $*"
  timeout --foreground 900 "$@" || echo "{\"failed\": \"$*\", \"status\": $?}" | tee -a "$OUT/failed.jsonl"
}
[[ " $IMPLS " == *" tree "* ]] && for case in d12 d19; do
  run $PY $B/treebench.py "$TREE" $case "$REPS" "$OUT/results.jsonl" "$OUT/pred" "$SCRATCH"
done
for impl in $IMPLS; do
  [ "$impl" = tree ] && continue
  for case in d12 d19; do
    for bins in default 1024; do
      run $PY $B/gpubench.py $impl $case $bins "$REPS" "$OUT/results.jsonl" "$OUT/pred"
    done
  done
done
echo "done $(date '+%F %T')"

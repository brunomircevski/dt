#!/usr/bin/env bash
# SLURM job: the whole GPU comparison on one node (GPU 0), e.g.
#   sbatch -p hopper --exclusive -t 02:00:00 -o results/hopper.log job.sh hopper
#   IMPLS="cuml xgboost" DATASETS=airline sbatch ...   runs only those
#   (default: tree cuml xgboost, on higgs and airline; ECBDL: DATASETS=ecbdl_15m)
# Results: results/<dataset>/<node>/results.jsonl, test predictions in
# results/<dataset>/<node>/pred/.
#SBATCH -J gdb-bench
set -uo pipefail
NODE="$1" REPS="${REPS:-3}" IMPLS="${IMPLS:-tree cuml xgboost}" DATASETS="${DATASETS:-higgs airline}"
B=$HOME/gpu-dt-bench
TREE_COMMIT=$(cut -c1-7 $B/dt/.git/HEAD)  # detached at the measured commit; no git on the nodes
SCRATCH=$(mktemp -d /tmp/gdb.XXXXXX)  # node-local tmpfs, for the tree dumps
trap 'rm -rf "$SCRATCH"' EXIT
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
PY=$B/venv/bin/python
TREE=$B/dt/tree
echo "job ${SLURM_JOB_ID:-} on $(hostname), $(nproc) CPUs, $(date '+%F %T'), reps $REPS, ./tree $TREE_COMMIT"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "CPU governor: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version,power.limit --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/GPU process: /'

run() {  # timeout per process: a case that does not finish in 30 min is reported as such;
  # --foreground keeps the case in the job's process group, so scancel stops it
  echo; echo "=== $(date +%H:%M:%S) $*"
  timeout --foreground 1800 "$@" || echo "{\"failed\": \"$*\", \"status\": $?}" | tee -a "$OUT/failed.jsonl"
}
for dataset in $DATASETS; do
  OUT=$B/results/$dataset/$NODE
  mkdir -p "$OUT/pred"
  {
    echo "{\"host\": \"$(hostname)\", \"cpu\": \"$(lscpu | sed -n 's/^Model name: *//p' | head -1)\","
    echo " \"cpus\": $(nproc), \"gpu\": \"$(nvidia-smi --id=0 --query-gpu=name,memory.total --format=csv,noheader)\","
    echo " \"driver\": \"$(nvidia-smi --id=0 --query-gpu=driver_version --format=csv,noheader)\","
    echo " \"cuda\": \"$(/usr/local/cuda/bin/nvcc --version | tail -2 | head -1)\","
    echo " \"tree_commit\": \"$TREE_COMMIT\", \"dataset\": \"$dataset\", \"date\": \"$(date '+%F %T')\"}"
  } > "$OUT/machine.json"
  for impl in $IMPLS; do
    for case in cart c45; do
      [ "$impl" = xgboost ] && [ "$case" = c45 ] && continue  # no entropy criterion
      if [ "$impl" = tree ]; then
        run $PY $B/treebench.py "$TREE" $dataset $case "$REPS" "$OUT/results.jsonl" "$OUT/pred" "$SCRATCH"
      else
        run $PY $B/gpubench.py $impl $dataset $case "$REPS" "$OUT/results.jsonl" "$OUT/pred"
      fi
    done
  done
done
echo "done $(date '+%F %T')"

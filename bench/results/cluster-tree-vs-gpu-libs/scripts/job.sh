#!/usr/bin/env bash
# SLURM job: the whole GPU comparison on one node (GPU 0), e.g.
#   sbatch -p hopper --exclusive -t 02:00:00 -o results/hopper.log job.sh hopper
#   IMPLS="cuml xgboost" DATASETS=airline sbatch ...   runs only those
#   (default: tree cuml xgboost, on higgs and airline; ECBDL: DATASETS=ecbdl_15m;
#   chess: DATASETS=chess_1b; tree_gpu: ./tree --gpu-only)
#   NUMA_NODE=1 sbatch ...   the first GPU on NUMA node 1, and every case bound
#   to that node's CPUs and memory (numactl): hopper's node 0 had 40 GB free in
#   2026-10. Every library and ./tree runs on its default threads, all the CPUs
#   it may use: the same CPUs for everyone.
#   REPLACE=1 sbatch ...   first moves the IMPLS' earlier rows out of each
#   dataset's results.jsonl (to results.jsonl.before-<job id>)
# machine.json is written once per <dataset>/<node>; a later job that adds
# rows writes machine-<job id>.json next to it.
# Results: results/<dataset>/<node>/results.jsonl, test predictions in
# results/<dataset>/<node>/pred/.
#SBATCH -J gdb-bench
set -uo pipefail
NODE="$1" REPS="${REPS:-3}" IMPLS="${IMPLS:-tree cuml xgboost}" DATASETS="${DATASETS:-higgs airline}"
B=$HOME/gpu-dt-bench
TREE_COMMIT=$(cut -c1-7 $B/dt/.git/HEAD)  # detached at the measured commit; no git on the nodes
SCRATCH=$(mktemp -d /tmp/gdb.XXXXXX)  # node-local tmpfs, for the tree dumps
export TMPDIR=$SCRATCH  # every temp file of the job, removed with it
# Leave the node as we found it, also when the job fails or SLURM ends it:
# no temp files, and the datasets (only on the login node's disk, over NFS)
# out of the page cache (dd iflag=nocache count=0 = posix_fadvise DONTNEED).
trap 'rm -rf "$SCRATCH"; find -L "${DATA_ROOT:-$HOME/dt/bench/data}" -type f -exec dd if={} iflag=nocache count=0 status=none \; 2>/dev/null' EXIT
trap 'exit 143' TERM INT
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
BIND=()
if [ -n "${NUMA_NODE:-}" ]; then
  BIND=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
  # nvidia-smi's bus id (00000000:2A:00.0) ends in the sysfs one (0000:2a:00.0).
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | tr A-Z a-z |
    while IFS=', ' read -r index bus; do
      [ "$(cat "/sys/bus/pci/devices/${bus: -12}/numa_node")" = "$NUMA_NODE" ] && { echo "$index"; break; }
    done || true)"
  [ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "no GPU on NUMA node $NUMA_NODE" >&2; exit 1; }
fi
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
PY=$B/venv/bin/python
TREE=$B/dt/tree
echo "job ${SLURM_JOB_ID:-} on $(hostname), $(nproc) CPUs, $(date '+%F %T'), reps $REPS, ./tree $TREE_COMMIT"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "free memory per NUMA node: $(numactl -H | awk '/ free:/ { printf "%snode %s %d GiB", (n++ ? ", " : ""), $2, $4 / 1024 }')"
[ -z "${NUMA_NODE:-}" ] || echo "NUMA node $NUMA_NODE: GPU $CUDA_VISIBLE_DEVICES, ${BIND[*]}, CPUs $(cat /sys/devices/system/node/node$NUMA_NODE/cpulist)"
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
  MACHINE=$OUT/machine.json
  [ -e "$MACHINE" ] && MACHINE=$OUT/machine-${SLURM_JOB_ID:-$$}.json
  {
    echo "{\"host\": \"$(hostname)\", \"cpu\": \"$(lscpu | sed -n 's/^Model name: *//p' | head -1)\","
    echo " \"cpus\": $(nproc), \"gpu\": \"$(nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=name,memory.total --format=csv,noheader)\","
    echo " \"driver\": \"$(nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=driver_version --format=csv,noheader)\","
    echo " \"cuda\": \"$(/usr/local/cuda/bin/nvcc --version | tail -2 | head -1)\","
    echo " \"numa_node\": \"${NUMA_NODE:-}\", \"gpu_index\": $CUDA_VISIBLE_DEVICES,"
    echo " \"tree_commit\": \"$TREE_COMMIT\", \"dataset\": \"$dataset\", \"date\": \"$(date '+%F %T')\"}"
  } > "$MACHINE"
  if [ -n "${REPLACE:-}" ] && [ -e "$OUT/results.jsonl" ]; then
    mv "$OUT/results.jsonl" "$OUT/results.jsonl.before-${SLURM_JOB_ID:-$$}"
    IMPLS="$IMPLS" $PY -c 'import json, os, sys
drop = set(os.environ["IMPLS"].split())
for line in open(sys.argv[1]):
    if json.loads(line)["impl"] not in drop:
        sys.stdout.write(line)' "$OUT/results.jsonl.before-${SLURM_JOB_ID:-$$}" > "$OUT/results.jsonl"
  fi
  for impl in $IMPLS; do
    for case in cart c45; do
      [ "$impl" = xgboost ] && [ "$case" = c45 ] && continue  # no entropy criterion
      if [ "$impl" = tree ] || [ "$impl" = tree_gpu ]; then
        mode=split; [ "$impl" = tree_gpu ] && mode=gpu
        run ${BIND[@]+"${BIND[@]}"} $PY $B/treebench.py "$TREE" $dataset $case "$REPS" "$OUT/results.jsonl" "$OUT/pred" "$SCRATCH" $mode
      else
        run ${BIND[@]+"${BIND[@]}"} $PY $B/gpubench.py $impl $dataset $case "$REPS" "$OUT/results.jsonl" "$OUT/pred"
      fi
    done
  done
done
echo "done $(date '+%F %T')"

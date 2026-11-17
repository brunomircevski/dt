#!/usr/bin/env bash
# SLURM job: the GPU libraries' runs on one node (one GPU), e.g.
#   sbatch -p hopper -w hopper --exclusive -t 1-00:00:00 -o results/hopper.log job.sh hopper
#
# Every run is one fresh gpubench.py process that loads the data, does the
# library's start-up (import, CUDA context) and fits exactly one model under
# the timer; no warm-up fit (BENCHMARK-CHANGES.MD, section 1). The runs are
# interleaved: REPS rounds, each round one process per case in a new shuffled
# order, so slow drift spreads over every case. A case is (library, dataset,
# trees): FORESTS lists the trees per fit and the forest settings,
# "1:identical" one tree (default), "10:default" each library's default
# forest of 10 trees.
#
# Environment (all optional):
#   IMPLS="cuml xgboost"  DATASETS="higgs mnist8m"  REPS=10  DEPTH=10
#   FORESTS="1:identical 10:default"
#   WARM_FORESTS="1:identical"   also these cases warm (gpubench.py WARM=1: one
#     untimed identical fit first in the same process, then the timed one);
#     default none
#   RESULTS=$HOME/gpu-dt-bench/results   where results go
#   RUSAGE=<dt clone>/bench/.tools/rusage   the launcher run.py uses (peak RSS,
#     CPU time); final.sh sets it. wall_seconds is from just before the launch
#     to the exit, as run.py's.
#   NUMA_NODE=1   the first GPU on NUMA node 1. Without it the job takes the GPU
#     on the NUMA node with the most free memory, as ./tree's
#     bench/cluster/job.sh does (hopper: node 1). Nothing is bound: every
#     library runs on its default threads with every CPU of the node and
#     first-touch memory, as ./tree's CUDA runs.
#   REPLACE=1   first moves the IMPLS' earlier rows out of each dataset's
#     results.jsonl (to results.jsonl.before-<job id>)
# A forest that fails with cuML's default 4 streams (out of GPU memory) is run
# again with N_STREAMS=1. The datasets leave the page cache before every
# process (as run.py --drop-cache), and the job leaves none behind.
# machine.json is written once per <dataset>/<node>; a later job that adds
# rows writes machine-<job id>.json next to it.
# Results: results/<dataset>/<node>/results.jsonl (one row per process),
# test predictions in results/<dataset>/<node>/pred/.
#SBATCH -J gdb-bench
set -uo pipefail
NODE="$1" REPS="${REPS:-10}" IMPLS="${IMPLS:-cuml xgboost}" DATASETS="${DATASETS:-higgs mnist8m}"
DEPTH="${DEPTH:-10}" FORESTS="${FORESTS:-1:identical}" WARM_FORESTS="${WARM_FORESTS:-}"
B=$HOME/gpu-dt-bench RESULTS="${RESULTS:-$HOME/gpu-dt-bench/results}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets}"
SCRATCH=$(mktemp -d /tmp/gdb.XXXXXX)  # node-local tmpfs
export TMPDIR=$SCRATCH  # every temp file of the job, removed with it
drop_cache() {  # posix_fadvise DONTNEED (dd iflag=nocache count=0) on every dataset file
  find -L "$DATA_ROOT/${1:-}" -type f -exec dd if={} iflag=nocache count=0 status=none \; 2>/dev/null
}
# Leave the node as we found it, also when the job fails or SLURM ends it.
trap 'rm -rf "$SCRATCH"; drop_cache' EXIT
trap 'exit 143' TERM INT
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 DATA_ROOT
# Start-up: the kernels load with the CUDA context, before the timer.
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"  # CUDA's default; EAGER: see BENCHMARK.md "Time"
if [ -z "${NUMA_NODE:-}" ]; then  # the GPU with the most memory (see bench/cluster/job.sh)
  # The node of a GPU with the most memory (ampere: GPUs 0-1 are A100 80 GB
  # on node 0, GPUs 2-3 A100 40 GB on node 1; free memory alone picked node 1
  # on 2026-11-17), then of those the node with the most free memory.
  NUMA_NODE="$(nvidia-smi --query-gpu=memory.total,pci.bus_id --format=csv,noheader,nounits | tr A-Z a-z |
    while IFS=', ' read -r total bus; do
      node="$(cat "/sys/bus/pci/devices/${bus: -12}/numa_node")"
      echo "$total $(numactl -H | awk -v n="$node" '$1 == "node" && $2 == n && $3 == "free:" { print $4 }') $node"
    done | sort -k1,1nr -k2,2nr | head -1 | cut -d' ' -f3)"
fi
if [ -n "${NUMA_NODE:-}" ]; then
  # nvidia-smi's bus id (00000000:2A:00.0) ends in the sysfs one (0000:2a:00.0).
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.total,pci.bus_id --format=csv,noheader,nounits | tr A-Z a-z |
    while IFS=', ' read -r index total bus; do  # the node's GPU with the most memory, the first on a tie
      [ "$(cat "/sys/bus/pci/devices/${bus: -12}/numa_node")" = "$NUMA_NODE" ] && echo "$total $index"
    done | sort -s -k1,1nr | head -1 | cut -d' ' -f2 || true)"
  [ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "no GPU on NUMA node $NUMA_NODE" >&2; exit 1; }
fi
# Every GPU run is on an 80 GB card (ampere also has two A100 40 GB, GPUs
# 2-3): a smaller GPU stops the job (GPU_MIN_MIB, default 80000).
gpu_mib="$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.total --format=csv,noheader,nounits)"
[ "${gpu_mib:-0}" -ge "${GPU_MIN_MIB:-80000}" ] ||
  { echo "GPU $CUDA_VISIBLE_DEVICES has ${gpu_mib:-?} MiB, under ${GPU_MIN_MIB:-80000}: not an 80 GB card" >&2; exit 1; }
# The CPUs, as ./tree's CUDA runs get them (bench/cluster/job.sh): every CPU
# of the node, the GPU's NUMA node's first; memory first-touch.
CPUS="$(cat "/sys/devices/system/node/node$NUMA_NODE/cpulist")$(for n in /sys/devices/system/node/node[0-9]*; do
  [ "${n##*node}" = "$NUMA_NODE" ] || printf ,%s "$(cat "$n/cpulist")"; done)"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
PY=$B/venv/bin/python
RUSAGE="${RUSAGE:-}"
[ -z "$RUSAGE" ] || [ -x "$RUSAGE" ] || { echo "RUSAGE=$RUSAGE is not executable" >&2; exit 1; }
echo "job ${SLURM_JOB_ID:-} on $(hostname), $(nproc) CPUs, $(date '+%F %T'), code ${TREE_GIT_DESCRIBE:-unknown}"
echo "$REPS round(s) of one process per case; cases: $IMPLS x $DATASETS x trees:forest $FORESTS${WARM_FORESTS:+, warm: $WARM_FORESTS}; depth <= $DEPTH; results: $RESULTS"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "free memory per NUMA node: $(numactl -H | awk '/ free:/ { printf "%snode %s %d GiB", (n++ ? ", " : ""), $2, $4 / 1024 }')"
echo "GPU $CUDA_VISIBLE_DEVICES on NUMA node $NUMA_NODE; CPUs $CPUS (taskset, as ./tree's CUDA runs), memory first-touch; CUDA_MODULE_LOADING=$CUDA_MODULE_LOADING"
echo "CPU governor: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version,power.limit --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/GPU process: /'

for dataset in $DATASETS; do
  OUT=$RESULTS/$dataset/$NODE
  mkdir -p "$OUT/pred"
  MACHINE=$OUT/machine.json
  [ -e "$MACHINE" ] && MACHINE=$OUT/machine-${SLURM_JOB_ID:-$$}.json
  {
    echo "{\"host\": \"$(hostname)\", \"cpu\": \"$(lscpu | sed -n 's/^Model name: *//p' | head -1)\","
    echo " \"cpus\": $(nproc), \"gpu\": \"$(nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=name,memory.total --format=csv,noheader)\","
    echo " \"driver\": \"$(nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=driver_version --format=csv,noheader)\","
    echo " \"cuda\": \"$(/usr/local/cuda/bin/nvcc --version | tail -2 | head -1)\","
    echo " \"numa_node\": \"${NUMA_NODE:-}\", \"gpu_index\": $CUDA_VISIBLE_DEVICES,"
    echo " \"commit\": \"${TREE_GIT_DESCRIBE:-}\", \"dataset\": \"$dataset\", \"date\": \"$(date '+%F %T')\"}"
  } > "$MACHINE"
  if [ -n "${REPLACE:-}" ] && [ -e "$OUT/results.jsonl" ]; then
    mv "$OUT/results.jsonl" "$OUT/results.jsonl.before-${SLURM_JOB_ID:-$$}"
    IMPLS="$IMPLS" $PY -c 'import json, os, sys
drop = set(os.environ["IMPLS"].split())
for line in open(sys.argv[1]):
    if json.loads(line)["impl"] not in drop:
        sys.stdout.write(line)' "$OUT/results.jsonl.before-${SLURM_JOB_ID:-$$}" > "$OUT/results.jsonl"
  fi
done

# The cases, CART only: no GPU library grows C4.5 trees, and an entropy tree
# is not C4.5, so C4.5 has no GPU competitor.
CASES=()
for dataset in $DATASETS; do for impl in $IMPLS; do for forest in $FORESTS; do
  CASES+=("$impl $dataset ${forest%%:*} ${forest#*:} 0")
done; for forest in $WARM_FORESTS; do
  CASES+=("$impl $dataset ${forest%%:*} ${forest#*:} 1")
done; done; done

# one <round> <impl> <dataset> <trees> <forest> <warm> [N_STREAMS]: one fresh process,
# its row (gpubench.py's) completed with the launcher's measurements and
# appended to the dataset's results.jsonl. A process that does not finish in 4
# hours is reported as such; --foreground keeps it in the job's process
# group, so scancel stops it.
one() {
  local round="$1" impl="$2" dataset="$3" trees="$4" forest="$5" warm="$6" streams="${7:-}"
  local out=$RESULTS/$dataset/$NODE row=$SCRATCH/row.jsonl usage=$SCRATCH/rusage start end status
  rm -f "$row" "$usage"
  drop_cache "$dataset"
  echo "=== $(date +%H:%M:%S) round $round: $impl $dataset, $trees tree(s), $forest$([ "$warm" = 1 ] && echo ", warm")${streams:+, N_STREAMS=$streams}"
  start=$(date +%s.%N)
  FOREST=$forest WARM=$warm ${streams:+N_STREAMS=$streams} timeout --foreground 14400 \
    ${RUSAGE:+"$RUSAGE" "$usage"} taskset -c "$CPUS" $PY "$B/gpubench.py" "$impl" "$dataset" cart "$row" "$out/pred" "$trees" "$DEPTH"
  status=$?
  end=$(date +%s.%N)
  if [ $status -eq 0 ] && [ -s "$row" ]; then
    $PY - "$row" "$usage" "$start" "$end" "$round" "${SLURM_JOB_ID:-}" >> "$out/results.jsonl" <<'EOF'
import json, os, sys
row_file, usage_file, start, end, rnd, job = sys.argv[1:]
row = json.loads(open(row_file).read().strip().splitlines()[-1])
row.update(round=int(rnd), job=job, wall_seconds=float(end) - float(start))
if os.path.exists(usage_file):
    rss_kib, user, system = open(usage_file).read().split()
    row.update(peak_rss_bytes=int(rss_kib) * 1024, user_seconds=float(user), system_seconds=float(system))
print(json.dumps(row))
EOF
    return 0
  fi
  echo "{\"failed\": \"$impl $dataset $trees $forest\", \"warm\": $warm, \"round\": $round, \"status\": $status, \"n_streams\": \"$streams\"}" |
    tee -a "$out/failed.jsonl"
  return $status
}

for round in $(seq 1 "$REPS"); do
  # a new order every round (seeded by the round, as run.py's)
  mapfile -t order < <(printf '%s\n' "${CASES[@]}" | $PY -c 'import random, sys
cases = sys.stdin.read().splitlines(); random.Random(int(sys.argv[1])).shuffle(cases); print("\n".join(cases))' "$round")
  for c in "${order[@]}"; do
    read -r impl dataset trees forest warm <<< "$c"
    one "$round" $impl $dataset $trees $forest $warm
    status=$?  # 124: timed out, not worth a second try
    if [ $status -ne 0 ] && [ $status -ne 124 ] && [ "$impl" = cuml ] && [ "$trees" -gt 1 ]; then
      one "$round" $impl $dataset $trees $forest $warm 1
    fi
  done
done
echo "done $(date '+%F %T')"

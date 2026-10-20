#!/usr/bin/env bash
# Where ./tree's training time goes (./tree --profile, src/core/trace.h), on
# mordor3: CUDA on hopper (H100) and ampere (A100), parallel on epyc1. Each
# node runs every case twice, one job after the other: with --profile into
# bench/results/<dir>/<run>/ (results.jsonl + profiles/*.json), and without
# into bench/results/<dir>/plain/<run>/, to see what profiling costs. Run it
# on the login node (it only calls sbatch); build first (README.md).
#
#   bench/cluster/profile.sh -o cluster-ecbdl-profile/gpu-plus-cpu   # ECBDL'14 15M sample, three protocols
#   bench/cluster/profile.sh -m 3 -d higgs    # HIGGS, each case 3 times
#   bench/cluster/profile.sh cpu-epyc-ecbdl_15m   # only the named runs
#   bench/cluster/profile.sh -o cluster-ecbdl-profile/gpu-only \
#       gpuonly-hopper-ecbdl_15m gpuonly-ampere-ecbdl_15m  # --gpu-only, every CPU
#                                                          # (only when named)
#
# -p: run.py's protocols (default cart_alpha_nodepth,cart_full_nodepth,c45);
# -o: the directory under bench/results/ (default cluster-<dataset>-profile, ecbdl_15m: cluster-ecbdl-profile);
# -n: threads, memory and GPU on that NUMA node (submit.sh -n); the CUDA runs
# take their GPU's NUMA node (one socket) without it too.

set -euo pipefail
RUNS=1
DATASET=ecbdl_15m
PROTOCOLS=cart_alpha_nodepth,cart_full_nodepth,c45
OUTDIR=
NUMA=
while getopts "m:d:p:o:n:h" option; do
  case "$option" in
    m) RUNS="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    p) PROTOCOLS="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    n) NUMA="$OPTARG" ;;
    *) sed -n '2,19p' "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "-m needs a positive number of runs" >&2; exit 2; }
DIR="${OUTDIR:-cluster-${DATASET%_15m}-profile}"

cd "$(dirname "$0")/../.."  # repository root: the jobs start here (SLURM_SUBMIT_DIR)
for f in tree tree_cpu bench/.tools/rusage bench/.venv/bin/python "bench/data/$DATASET/train.csv"; do
  [ -e "$f" ] || { echo "$f missing: build and prepare first (bench/cluster/README.md)" >&2; exit 1; }
done

# name, partition, node, implementation, threads
JOBS=("cuda-hopper-$DATASET hopper hopper tree_cuda all"
      "cuda-ampere-$DATASET ampere ampere tree_cuda all"
      "cpu-epyc-$DATASET    epyc   epyc1  tree      all")
OPTIONAL=("gpuonly-hopper-$DATASET hopper hopper tree_gpu_only all"
          "gpuonly-ampere-$DATASET ampere ampere tree_gpu_only all")
if [ $# -gt 0 ]; then
  selected=()
  for job in "${JOBS[@]}" "${OPTIONAL[@]}"; do
    read -r name _ _ _ <<<"$job"
    for wanted in "$@"; do [ "$wanted" = "$name" ] && selected+=("$job"); done
  done
  [ ${#selected[@]} -eq $# ] || { echo "unknown run name in: $*" >&2; exit 2; }
  JOBS=("${selected[@]}")
fi
for job in "${JOBS[@]}"; do
  read -r name _ _ _ <<<"$job"
  for out in "$DIR/$name" "$DIR/plain/$name"; do
    if [ -e "bench/results/$out/results.jsonl" ]; then
      echo "bench/results/$out already has results: move them away before running again" >&2
      exit 1
    fi
  done
done

git diff --quiet HEAD -- src bench Makefile ||
  echo "warning: uncommitted changes in src/, bench/ or Makefile: machine.json will say -dirty"
export TREE_GIT_DESCRIBE="$(git describe --always --dirty --abbrev=12)"
echo "code: $TREE_GIT_DESCRIBE, $DATASET, $PROTOCOLS -> bench/results/$DIR, runs per case: $RUNS${NUMA:+, NUMA node $NUMA}"

ids=()
for job in "${JOBS[@]}"; do
  read -r name partition node impl threads <<<"$job"
  mkdir -p "bench/results/$DIR/$name" "bench/results/$DIR/plain/$name"
  profiled=$(sbatch --parsable --hold -J "dt-prof-$name" -p "$partition" -w "$node" --exclusive \
    -t 04:00:00 --export=ALL,RUN_EXTRA=--profile${NUMA:+,NUMA_NODE=$NUMA} -o "bench/results/$DIR/$name/run.log" \
    bench/cluster/job.sh "$DIR" "$name" "$DATASET" "$impl" "$RUNS" "$PROTOCOLS" "$threads" 3600)
  plain=$(sbatch --parsable -J "dt-plain-$name" -p "$partition" -w "$node" --exclusive \
    -t 04:00:00 --dependency=afterany:"$profiled" --export=ALL${NUMA:+,NUMA_NODE=$NUMA} -o "bench/results/$DIR/plain/$name/run.log" \
    bench/cluster/job.sh "$DIR/plain" "$name" "$DATASET" "$impl" "$RUNS" "$PROTOCOLS" "$threads" 3600)
  ids+=("$profiled")
  echo "$name: jobs $profiled (profiled), then $plain (plain) on $node"
done
scontrol release "${ids[@]}"
squeue -a -u "$USER" -o "%.8i %.10P %.20j %.2t %.10M %R"

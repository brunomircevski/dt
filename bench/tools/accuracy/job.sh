#!/usr/bin/env bash
# SLURM job of the pruned-accuracy benchmark: acc.py for one dataset and one
# side (cpu: epyc / dell nodes; gpu: hopper / ampere). Submit from the
# repository root; time is not measured, so nodes are shared (no --exclusive):
#
#   sbatch -p epyc -w epyc2 -c 48 -t 1-00:00:00 -J acc-susy-cpu -o bench/cluster-accuracy-pruned/logs/%j.out \
#     bench/tools/accuracy/job.sh susy cpu [acc.py options]
#   sbatch -p hopper -c 16 --oversubscribe -t 12:00:00 -J acc-susy-gpu -o ... \
#     bench/tools/accuracy/job.sh susy gpu
set -euo pipefail
DATASET="$1" SIDE="$2"; shift 2
GPU=; [[ "$SIDE" == gpu:* ]] && GPU="${SIDE#gpu:}" SIDE=gpu  # gpu:N: that GPU
cd "$SLURM_SUBMIT_DIR"
export TMPDIR="$(mktemp -d "/tmp/dt-acc-$SLURM_JOB_ID.XXXXXX")"
trap 'rm -rf "$TMPDIR"' EXIT
trap 'exit 143' TERM INT
DTBENCH="${DTBENCH:-$HOME/micromamba/envs/dtbench}"
[ ! -d "$DTBENCH/bin" ] || export PATH="$DTBENCH/bin:$PATH"
export ACC_TOOLS="${ACC_TOOLS:-$HOME/dt-final/bench/tools/.tools}"  # weka.jar, bounce.jar, yadt/
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} CPUs, $(date '+%F %T'), commit ${TREE_GIT_DESCRIBE:-?}"
if [ "$SIDE" = gpu ]; then
  # the GPU with the least memory in use (no GRES on mordor3)
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${GPU:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    sort -t, -k2,2n | head -1 | cut -d, -f1)}"
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
else
  mkdir -p bench/tools/accuracy/.classes
  [ bench/tools/accuracy/.classes/J48Acc.class -nt bench/tools/accuracy/J48Acc.java ] ||
  javac -nowarn -d bench/tools/accuracy/.classes -cp "$ACC_TOOLS/weka.jar:$ACC_TOOLS/bounce.jar" \
    bench/tools/adapters/J48Fit.java bench/tools/accuracy/J48Acc.java
fi
bench/tools/.venv/bin/python bench/tools/accuracy/acc.py "$DATASET" --side "$SIDE" \
  --out "${ACC_OUT:-bench/cluster-accuracy-pruned/runs}" "$@"
echo "done $(date '+%F %T')"

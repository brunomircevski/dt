#!/usr/bin/env bash
# SLURM job on hopper (the CPU nodes have no CUDA toolkit): build ./tree (CPU +
# CUDA for RTX 2080 Ti, A100 and H100), ./tree_cpu and the rusage launcher. Submit it from
# the repository root (never build on the login node):
#
#   sbatch bench/results/cluster/build.sh
#
#SBATCH -J dt-build
#SBATCH -p hopper
#SBATCH -c 16
#SBATCH -t 00:20:00
#SBATCH -o bench/results/cluster/build.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# hopper has no git: the commit comes from the login node, if given
#   sbatch --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty)" bench/results/cluster/build.sh
echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, $(date '+%F %T'), commit ${TREE_GIT_DESCRIBE:-unknown}"
CUDA_ARCH="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90"
# make does not see a changed CUDA_ARCH: rebuild the CUDA object when it changes.
if [ "$(cat build/cuda_arch 2>/dev/null)" != "$CUDA_ARCH" ]; then
  rm -f build/build/gpu_grower.o tree
  mkdir -p build && echo "$CUDA_ARCH" > build/cuda_arch
fi
make -j "$SLURM_CPUS_PER_TASK" tree tree_cpu NVCC=/usr/local/cuda/bin/nvcc \
  CUDA_LIB=/usr/local/cuda/lib64 CUDA_ARCH="$CUDA_ARCH"
mkdir -p bench/.tools
[ -x bench/.tools/rusage ] ||
  cc -O2 -static -Wall -o bench/.tools/rusage bench/adapters/rusage.c 2>/dev/null ||
  cc -O2 -Wall -o bench/.tools/rusage bench/adapters/rusage.c
/usr/local/cuda/bin/cuobjdump --list-elf tree | sed 's/^/kernels: /'
echo "done $(date '+%F %T')"

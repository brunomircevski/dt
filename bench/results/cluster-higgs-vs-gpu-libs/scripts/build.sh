#!/usr/bin/env bash
# SLURM job: build ./tree (commit 35cf315, in its own clone ~/gpu-dt-bench/dt,
# so rebuilds of ~/dt cannot change it mid-run) for the A100 and the H100.
# Submit from ~/gpu-dt-bench:  sbatch build.sh
#SBATCH -J gdb-build
#SBATCH -p hopper
#SBATCH --exclusive
#SBATCH -t 00:20:00
#SBATCH -o build.log
set -euo pipefail
B=$HOME/gpu-dt-bench
echo "job $SLURM_JOB_ID on $(hostname), $(nproc) CPUs, $(date '+%F %T')"
export PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64
cd $B/dt
make -j 32 tree NVCC=/usr/local/cuda/bin/nvcc CUDA_LIB=/usr/local/cuda/lib64 \
  CUDA_ARCH="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90"
cuobjdump --list-elf tree
echo "done $(date '+%F %T')"

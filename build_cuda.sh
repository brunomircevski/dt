#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NVCC_FLAGS=(
  -std=c++20
  -O3
  -arch=native
  -x cu
  -I.
  -I/usr/include/cccl
)

NVCC="${NVCC:-/opt/cuda/bin/nvcc}"

"${NVCC}" "${NVCC_FLAGS[@]}" \
  tree_cuda.cpp \
  tree_base.cpp \
  tree_serial.cpp \
  tree_parallel.cpp \
  task_executor.cpp \
  dataset.cpp \
  node.cpp \
  pruning/pruning.cpp \
  main.cpp \
  -o tree

echo "Built ./tree"

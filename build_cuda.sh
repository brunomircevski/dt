#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" == "clean" ]]; then
  rm -rf build tree
  echo "Cleaned build/ and tree"
  exit 0
fi

BUILD_DIR=build
OUTPUT=tree
NVCC="${NVCC:-/opt/cuda/bin/nvcc}"
CXX="${CXX:-/usr/bin/g++}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/opt/cuda/lib64}"

CXX_FLAGS=(-std=c++20 -O2 -pthread -I.)
NVCC_FLAGS=(-std=c++20 -O2 -arch=native -x cu -I. -I/usr/include/cccl)
LINK_FLAGS=(-std=c++20 -O2 -pthread -L"${CUDA_LIB_DIR}" -lcudart)

CPU_SOURCES=(
  main.cpp
  tree_base.cpp
  tree_parallel.cpp
  tree_serial.cpp
  task_executor.cpp
  dataset.cpp
  node.cpp
  options.cpp
  pruning/pruning.cpp
)

mkdir -p "${BUILD_DIR}"

pids=()

compile_cpu() {
  local source=$1
  local object="${BUILD_DIR}/$(basename "${source}" .cpp).o"
  if [[ ! -f "${object}" || "${source}" -nt "${object}" ]]; then
    "${CXX}" "${CXX_FLAGS[@]}" -c "${source}" -o "${object}"
  fi
}

for source in "${CPU_SOURCES[@]}"; do
  compile_cpu "${source}" &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

cuda_object="${BUILD_DIR}/tree_cuda.o"
if [[ ! -f "${cuda_object}" || tree_cuda.cpp -nt "${cuda_object}" ]]; then
  "${NVCC}" "${NVCC_FLAGS[@]}" -c tree_cuda.cpp -o "${cuda_object}"
fi

needs_link=false
if [[ ! -f "${OUTPUT}" ]]; then
  needs_link=true
else
  for object in "${BUILD_DIR}"/*.o; do
    if [[ "${object}" -nt "${OUTPUT}" ]]; then
      needs_link=true
      break
    fi
  done
fi

if [[ "${needs_link}" == true ]]; then
  mapfile -t objects < <(find "${BUILD_DIR}" -maxdepth 1 -name '*.o' -print | sort)
  "${CXX}" "${LINK_FLAGS[@]}" "${objects[@]}" -o "${OUTPUT}"
fi

echo "Built ./${OUTPUT}"

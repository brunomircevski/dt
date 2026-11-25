#!/usr/bin/env bash
# Rerun after the 2026-11-25 SOREL 4M sweep below 4096: --hybrid only, the depths whose value changed.
# $1: the protocols (comma-separated) for this node.
set -uo pipefail
cd /home/bmircevski/dt-r25
export SLURM_SUBMIT_DIR=$PWD TREE_GIT_DESCRIBE=cfd18321d9d3 RUNS=10 KINDS=cuda
NODE="$(hostname -s)"
for _ in $(seq "$(nproc)"); do timeout 30 sh -c 'while :; do :; done' & done; wait
DIR=hybrid-vs-gpu-only-r26 DATASETS=sorel_4m PROFILE_DATASETS= PROTOCOLS="$1" \
  bash bench/tools/cluster/hybrid_vs_gpu_only.sh "$NODE"

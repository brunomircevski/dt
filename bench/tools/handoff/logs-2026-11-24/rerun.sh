#!/usr/bin/env bash
# Rerun after the 2026-11-24 hand-off re-sweep: --hybrid only, the cases whose value changed.
set -uo pipefail
cd /home/bmircevski/dt-r24
export SLURM_SUBMIT_DIR=$PWD TREE_GIT_DESCRIBE=0b6ff5094490 RUNS=10 KINDS=cuda
NODE="$(hostname -s)"; rc=0
for _ in $(seq "$(nproc)"); do timeout 30 sh -c 'while :; do :; done' & done; wait
DIR=hybrid-vs-gpu-only-r25 DATASETS="higgs mnist8m sorel_4m" PROFILE_DATASETS=mnist8m \
  PROTOCOLS=cart_depth15,cart_depth20,cart_depth25,cart_depth30,c45_depth15_unpruned,c45_depth20_unpruned,c45_depth25_unpruned,c45_depth30_unpruned \
  bash bench/tools/cluster/hybrid_vs_gpu_only.sh "$NODE" || rc=1
DIR=hybrid-vs-gpu-only-r25b DATASETS=higgs PROFILE_DATASETS= PROTOCOLS=cart_full_nodepth,c45_unpruned \
  bash bench/tools/cluster/hybrid_vs_gpu_only.sh "$NODE" || rc=1
exit $rc

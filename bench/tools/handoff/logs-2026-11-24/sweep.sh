#!/usr/bin/env bash
# Hand-off re-sweep 2026-11-24: one part (A: higgs+mnist8m depths 15-30, B: sorel_4m depths 15-30,
# C: no-limit confirmation) on this node, run from ~/dt-r24.
set -uo pipefail
PART="$1" NODE="$(hostname -s)"
cd /home/bmircevski/dt-r24
export SLURM_SUBMIT_DIR=$PWD TREE_GIT_DESCRIBE=0b6ff5094490 SWEEP_RUNS=5
rc=0
sweep() { DEPTH="$1" DATASETS="$2" VALUES="$3" bash bench/tools/handoff/handoff_sweep.sh "$NODE" || rc=1; }
case "$PART" in
  A) for d in 15 20 25 30; do
       sweep $d higgs "default 128 256 512 1024 2048 4096 8192"
       sweep $d mnist8m "default 512 1024 2048 4096 8192 16384 32768"
     done ;;
  B) for d in 15 20 25 30; do sweep $d sorel_4m "default 4096 8192 16384 32768 65536"; done ;;
  C) if [ "$NODE" = hopper ]; then sweep none higgs "64 128 256 512 1024"
     else sweep none higgs "128 256 512 1024 2048"; fi
     sweep none mnist8m "default 2048 4096 8192 16384 32768"
     sweep none sorel_4m "4096 8192 16384 32768 65536" ;;
  D) # catch-up: values already done are kept, only missing ones run; picks redone
     for d in 15 20 25 30; do
       sweep $d higgs "default 128 256 512 1024 2048 4096 8192"
       sweep $d mnist8m "default 512 1024 2048 4096 8192 16384 32768"
       sweep $d sorel_4m "default 4096 8192 16384 32768 65536"
     done ;;
esac
exit $rc

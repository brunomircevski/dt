#!/usr/bin/env bash
# SOREL 4M hand-off sweep 2026-11-25: values below 4096 at depths 15-30, with 4096 and 8192
# measured again in the same job (the 2026-11-24 runs of those two moved aside first).
set -uo pipefail
NODE="$(hostname -s)"
cd /home/bmircevski/dt-r25
export SLURM_SUBMIT_DIR=$PWD TREE_GIT_DESCRIBE=cfd18321d9d3 SWEEP_RUNS=5
rc=0
for d in 15 20 25 30; do
  DEPTH=$d DATASETS=sorel_4m VALUES="1024 2048 4096 8192" bash bench/tools/handoff/handoff_sweep.sh "$NODE" || rc=1
done
exit $rc

#!/usr/bin/env bash
# The libraries' forests of 10 trees with their own random-forest defaults
# (gpubench.py FOREST=default: only the criterion and depth <= 10 set), on
# hopper (H100, NUMA node 1) and ampere (A100, NUMA node 0). The charts show
# the forest's time divided by 10 next to ./tree's and the libraries' one tree
# (submit-forest.sh, TREES=1). Pass a job id to start only after it ended well.
set -euo pipefail
cd ~/gpu-dt-bench
mkdir -p results-forest-default
common=(--exclusive -t 04:00:00 ${1:+--dependency=afterok:$1})
export IMPLS="cuml xgboost" DATASETS="higgs susy airline ecbdl_15m chess_1b" \
  TREES=10 FOREST=default DEPTH=10 CUML_WARMUP_ROWS=0 RESULTS=$HOME/gpu-dt-bench/results-forest-default
h=$(NUMA_NODE=1 sbatch --parsable -J gdb-forestd-hopper -p hopper "${common[@]}" -o results-forest-default/hopper.log job.sh hopper)
a=$(NUMA_NODE=0 sbatch --parsable -J gdb-forestd-ampere -p ampere "${common[@]}" -o results-forest-default/ampere.log job.sh ampere)
echo "hopper $h ampere $a"
squeue -a -u bmircevski -o "%i %j %T %R"

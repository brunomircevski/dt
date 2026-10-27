#!/usr/bin/env bash
# The libraries' 1-tree and 10-tree fits at depth <= 10, scored on each
# dataset's official test set (airline: the 2008 flights; chess: its own test
# file), on hopper (H100, NUMA node 1) and ampere (A100, NUMA node 0).
# Every tree of a forest is the same tree (no bootstrap, all features), so
# (T10 - T1) / 9 is the cost of one more tree and T1 minus that the setup.
# ./tree is not run here: its depth-10 unpruned times are in cluster-higgs,
# cluster-susy, cluster-airline, cluster-ecbdl and cluster-chess.
set -euo pipefail
cd ~/gpu-dt-bench
mkdir -p results-forest
common=(--exclusive -t 06:00:00)
export IMPLS="cuml xgboost" DATASETS="higgs susy airline ecbdl_15m chess_1b" \
  TREES="1 10" DEPTH=10 CUML_WARMUP_ROWS=0 RESULTS=$HOME/gpu-dt-bench/results-forest
h=$(NUMA_NODE=1 sbatch --parsable -J gdb-forest-hopper -p hopper "${common[@]}" -o results-forest/hopper.log job.sh hopper)
a=$(NUMA_NODE=0 sbatch --parsable -J gdb-forest-ampere -p ampere "${common[@]}" -o results-forest/ampere.log job.sh ampere)
echo "hopper $h ampere $a"
squeue -a -u bmircevski -o "%i %j %T %R"

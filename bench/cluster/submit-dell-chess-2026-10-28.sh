#!/usr/bin/env bash
# The rest of the one-build reruns of 2026-10-28 (after submit-final-2026-10-28.sh):
#   - dell: every ./tree_cpu run there (parallel on both sockets, one socket,
#     dell2's 512 GB, --serial) was of a build before the spin fix and the
#     distinct-values fix; they run again on the final build, into
#     cluster-<dataset>-fix/, replacing the old folders. --simple on dell does
#     not use that code and stays. No new dell variants: the same runs as before.
#   - chess: every folder was of 2026-10-20 builds (before GPU only on every CPU,
#     3ccf4ba, and later changes). The epyc runs are in submit-final; the GPU
#     runs start over too, into cluster-chess-fix/, instead of the top-ups that
#     would have merged into the old folders (cancelled). Later the same day the
#     GPU jobs (619-634) were cancelled too: the CUDA backend is being reworked.
# Nodes: dell1 runs the parallel runs, then --serial (long) after them; dell2
# only its 512 GB chess runs. Runs a case: about 2 minutes or less 10, under
# about 10 minutes 5, longer once. Run on the login node.
set -euo pipefail
cd "$(dirname "$0")/../.."
S=bench/cluster/submit.sh
P=cart_depth10,c45_depth10_unpruned
C=cart_full_nodepth,c45_unpruned
out() { case $1 in ecbdl*) echo cluster-ecbdl-fix ;; chess*) echo cluster-chess-fix ;; *) echo cluster-$1-fix ;; esac; }
ids() { sed -n 's/.*: job \([0-9]*\) on .*/\1/p'; }

# serial-dell-sorel (job 338) ran across two binary swaps; the GPU chess top-ups.
scancel 338 491 492 493 494 498 499 500 501 505 506 507 508 512 513 514 515 || true

# dell1, parallel.
par=()
for d in airline higgs susy ecbdl_15m mnist8m sorel; do
  par+=($($S -m 10 -e -d $d -p $P -o $(out $d) cpu-dell-$d | ids))
done
par+=($($S -m 5 -e -d ecbdl -p $P -o $(out ecbdl) cpu-dell-ecbdl | ids))
for d in mnist8m sorel; do par+=($($S -m 10 -e -d $d -p $P -o $(out $d) socket-dell-$d | ids)); done
for d in chess_1m chess_10m chess_100m chess_1b; do
  par+=($($S -m 10 -d $d -p $C -o $(out $d) cpu-dell-$d socket-dell-$d | ids))
  # dell2
  $S -m 10 -d $d -p $C -o $(out $d) cpu-dell512-$d | ids
done

# dell1, --serial, one after another after the parallel runs.
serial=()
for d in chess_1m chess_10m chess_100m; do serial+=($($S -a ${par[-1]} -m 10 -d $d -p $C -o $(out $d) serial-dell-$d | ids)); done
serial+=($($S -a ${par[-1]} -m 5 -d chess_1b -p $C -o $(out chess_1b) serial-dell-chess_1b | ids))
serial+=($($S -a ${par[-1]} -m 5 -e -d mnist8m -p $P -o $(out mnist8m) serial-dell-mnist8m | ids))
serial+=($($S -a ${par[-1]} -m 1 -e -d sorel -p $P -o $(out sorel) serial-dell-sorel | ids))
prev=$(IFS=:; echo "${par[*]}")
for j in "${serial[@]}"; do scontrol update JobId=$j Dependency=afterany:$prev; prev=$j; done

# GPU chess, from scratch.
for d in chess_1m chess_10m chess_100m chess_1b; do
  $S -m 10 -d $d -p $C -o $(out $d) {cuda,gpuonly}-{hopper,ampere}-$d | ids
done
echo "dell parallel: ${par[*]}; dell serial: ${serial[*]}"
squeue -a -u "$USER" -o "%.6i %.26j %.8T %.8N %.20E" | grep -E "dell|chess|JOBID"

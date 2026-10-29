#!/usr/bin/env bash
# The parallel epyc runs again on the build with the spin fix (2026-10-28,
# partitionInBlocks sleeps after a short spin: before it, a descheduled thread
# kept 47 others spinning, and small epyc runs took up to 8x as long). Only the
# parallel backend uses those blocks (CUDA hybrid hands the CPU nodes under
# 4,096 rows, --serial and --simple have no pool), so only cpu-epyc-* and
# socket-epyc-* start over: into cluster-<dataset>-fix/, replacing the old
# folders of cluster-<dataset>/ when done (not merged: the old runs are of the
# old build). Cancels their jobs of submit-variants/submit-topup first. 10 runs
# a case (5 for full ECBDL: about 10 minutes a run). Run on the login node,
# after the build (bench/cluster/build.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
export TREE_GIT_DESCRIBE=$(git describe --always --dirty --abbrev=12)
S=bench/cluster/submit.sh
P=cart_depth10,c45_depth10_unpruned
C=cart_full_nodepth,c45_unpruned
out() { case $1 in ecbdl*) echo cluster-ecbdl-fix ;; chess*) echo cluster-chess-fix ;; *) echo cluster-$1-fix ;; esac; }

scancel 484 485 487 489 490 495 496 502 503 509 510 516 517 521 523 526 528 531 533 535
for d in airline higgs susy ecbdl_15m mnist8m; do
  $S -m 10 -e -d $d -p $P -o $(out $d) {cpu,socket}-epyc-$d
done
$S -m 5 -e -d ecbdl -p $P -o $(out ecbdl) {cpu,socket}-epyc-ecbdl
$S -m 10 -e -d sorel -p $P -o $(out sorel) cpu-epyc-sorel  # one socket's memory is too small
for d in chess_1m chess_10m chess_100m chess_1b; do
  $S -m 10 -d $d -p $C -o $(out $d) {cpu,socket}-epyc-$d
done

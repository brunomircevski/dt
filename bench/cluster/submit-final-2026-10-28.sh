#!/usr/bin/env bash
# Every ./tree_cpu epyc run (parallel on both sockets, on one socket, --serial)
# on one build: the spin fix and the distinct-values fix of 2026-10-28
# (cpu_grower.cpp: C4.5's distinct values counted and written in place on the
# pool, one pass with --serial). The other backends are untouched by both
# fixes (CUDA hybrid and GPU only: GPU presort; --simple: no presort).
#   - parallel runs that ran on the build before the distinct-values fix
#     (jobs 550-556) again, into cluster-<dataset>-fix/ (old ones moved to
#     bench/results-old-build/); the held spin-fix jobs 557-570 run as they are.
#   - --serial again for every dataset, into cluster-<dataset>-fix/ (replacing
#     the serial folders, not merged); the pending serial top-ups are cancelled.
# Nodes: the long runs (full ECBDL and SOREL parallel, every --serial, then the
# pending --simple jobs) one after another on epyc2; the short parallel runs
# on epyc1, which is then free for short and interactive runs.
# Runs a case: about 2 minutes or less a run 10, under about 10 minutes 5,
# longer once. Run on the login node after installing the build.
set -euo pipefail
cd "$(dirname "$0")/../.."
S=bench/cluster/submit.sh
P=cart_depth10,c45_depth10_unpruned
C=cart_full_nodepth,c45_unpruned
R=bench/results
OLD=bench/results-old-build
out() { case $1 in ecbdl*) echo cluster-ecbdl-fix ;; chess*) echo cluster-chess-fix ;; *) echo cluster-$1-fix ;; esac; }
move() { mkdir -p "$OLD/$(dirname "$1")"; [ ! -e "$R/$1" ] || mv "$R/$1" "$OLD/$1"; }

# Old builds out of the way.
for f in cluster-airline-fix/socket-epyc-airline cluster-higgs-fix/{cpu,socket}-epyc-higgs \
         cluster-susy-fix/{cpu,socket}-epyc-susy cluster-ecbdl-fix/{cpu,socket}-epyc-ecbdl_15m; do
  move "$f"
done
for f in $R/cluster-*-new/serial-epyc-* $R/cluster-*-top/serial-epyc-*; do
  if [ -e "$f" ]; then move "${f#$R/}"; fi
done
scancel 486 488 497 504 511 518 522 527 532 || true

# Parallel, epyc1 (short).
$S -m 10 -e -w epyc1 -d airline -p $P -o $(out airline) socket-epyc-airline
for d in higgs susy ecbdl_15m; do
  $S -m 10 -e -w epyc1 -d $d -p $P -o $(out $d) {cpu,socket}-epyc-$d
done
for j in 557 558 562 563 564 565 566 567 568 569 570; do scontrol update JobId=$j ReqNodeList=epyc1; done
for j in 559 560 561; do scontrol update JobId=$j ReqNodeList=epyc2; done
scontrol update JobId=560 Dependency=afterany:559
scontrol update JobId=561 Dependency=afterany:560
scontrol release 557,558,559,560,561,562,563,564,565,566,567,568,569,570

# --serial, epyc2 (long), after SOREL's parallel run.
serial=()
sub() { serial+=($($S -a 561 -w epyc2 "$@" | sed -n 's/.*: job \([0-9]*\) on .*/\1/p')); }
for d in airline higgs susy; do sub -m 10 -e -d $d -p $P -o $(out $d) serial-epyc-$d; done
sub -m 5 -e -d mnist8m -p $P -o $(out mnist8m) serial-epyc-mnist8m
sub -m 5 -e -d ecbdl_15m -p $P -o $(out ecbdl_15m) serial-epyc-ecbdl_15m
sub -m 1 -e -d ecbdl -p $P -o $(out ecbdl) serial-epyc-ecbdl
sub -m 1 -e -d sorel -p $P -o $(out sorel) serial-epyc-sorel
for d in chess_1m chess_10m chess_100m; do sub -m 10 -d $d -p $C -o $(out $d) serial-epyc-$d; done
sub -m 5 -d chess_1b -p $C -o $(out chess_1b) serial-epyc-chess_1b
prev=561
for j in "${serial[@]}"; do scontrol update JobId=$j Dependency=afterany:$prev; prev=$j; done

# --simple last, epyc2.
for j in 460 461 468 469 536 537 538 539 540 541 542 543 544 545; do
  scontrol update JobId=$j ReqNodeList=epyc2 Dependency=afterany:$prev
  prev=$j
done
echo "serial jobs: ${serial[*]}"
squeue -a -u "$USER" -o "%.6i %.24j %.8T %.8N %.30E" | grep -E "epyc|PART"

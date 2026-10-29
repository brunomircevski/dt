#!/usr/bin/env bash
# Every cluster-<dataset> benchmark with every variant on epyc and the GPUs,
# runs under about 10 minutes five times (2026-10-28). Results go to
# bench/results/cluster-<dataset>-new/, to replace the epyc and GPU folders of
# cluster-<dataset>/ when done (the dell folders stay). Run on the login node.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TREE_GIT_DESCRIBE=$(git describe --always --dirty --abbrev=12)
S=bench/cluster/submit.sh
P=cart_depth10,c45_depth10_unpruned
out() { case $1 in ecbdl*) echo cluster-ecbdl-new ;; chess*) echo cluster-chess-new ;; *) echo cluster-$1-new ;; esac; }

# Test sets: the GPU runs on their GPU's NUMA node (-n 0), as on 2026-10-23.
for d in airline higgs susy ecbdl_15m mnist8m; do
  $S -m 5 -e -n 0 -d $d -p $P -o $(out $d) {cuda,gpuonly}-{hopper,ampere}-$d
  $S -m 5 -e -d $d -p $P -o $(out $d) {cpu,socket,serial}-epyc-$d
done
$S -m 5 -e -d ecbdl -p $P -o $(out ecbdl) {cpu,socket,serial}-epyc-ecbdl
$S -m 5 -e -d sorel -p $P -o $(out sorel) cpu-epyc-sorel
# Too big for an 80 GB GPU: one run each, to record how they fail.
for d in ecbdl sorel; do
  $S -m 1 -e -n 0 -d $d -p $P -o $(out $d) {cuda,gpuonly}-{hopper,ampere}-$d
done

# Chess: the same protocols as before (no depth limit, training accuracy only).
for d in chess_1m chess_10m chess_100m chess_1b; do
  $S -m 5 -n 0 -d $d -p cart_full_nodepth,c45_unpruned -o $(out $d) {cuda,gpuonly}-{hopper,ampere}-$d
  $S -m 5 -d $d -p cart_full_nodepth,c45_unpruned -o $(out $d) {cpu,socket,serial}-epyc-$d
done

# --simple: one job per protocol, straight to job.sh.
simple() {  # dataset runs limit timeout accuracy protocols...
  local d=$1 runs=$2 limit=$3 timeout=$4 accuracy=$5 dir n="simple-epyc-$1"; shift 5
  dir=$(out $d); mkdir -p bench/results/$dir/$n
  for p in "$@"; do
    echo "$dir/$n ${p%%_*}: job $(sbatch --parsable -J dt-$n-${p%%_*} -p epyc -w epyc1 --exclusive -t $limit \
      --export=ALL,ACCURACY=$accuracy -o bench/results/$dir/$n/run-${p%%_*}.log \
      bench/cluster/job.sh $dir $n $d tree_simple $runs $p 1 $timeout)"
  done
}
simple higgs 5 06:00:00 7200 --test-accuracy cart_depth10 c45_depth10_unpruned
simple susy 5 03:00:00 3600 --test-accuracy cart_depth10 c45_depth10_unpruned
simple sorel 1 3-00:00:00 172800 --test-accuracy cart_depth10 c45_depth10_unpruned
for d in chess_1m chess_10m chess_100m; do
  simple $d 5 06:00:00 7200 --train-accuracy-only cart_full_nodepth c45_unpruned
done
simple chess_1b 1 1-00:00:00 43200 --train-accuracy-only cart_full_nodepth c45_unpruned

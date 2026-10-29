#!/usr/bin/env bash
# Top-up of submit-variants-2026-10-28.sh (2026-10-28, later): a run of about
# 2 minutes or less (the whole process, reading the data included) gets 10 runs
# per case, one under about 10 minutes 5, a longer one keeps what it has. The
# runs already in bench/results/cluster-<dataset>/<run>/ count: only the
# missing ones are run. The pending jobs of submit-variants are cancelled and
# queued again with these counts (into cluster-<dataset>-new/); the jobs it had
# already started ran 5 runs, and get the rest in cluster-<dataset>-top/. All
# three folders are merged into cluster-<dataset>/<run>/ when done. Run on the
# login node.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TREE_GIT_DESCRIBE=$(git describe --always --dirty --abbrev=12)
S=bench/cluster/submit.sh
P=cart_depth10,c45_depth10_unpruned
C=cart_full_nodepth,c45_unpruned
out() { case $1 in ecbdl*) echo cluster-ecbdl-$2 ;; chess*) echo cluster-chess-$2 ;; *) echo cluster-$1-$2 ;; esac; }

# Pending jobs of submit-variants whose count changes. Kept: 417 (socket-epyc-ecbdl,
# new, 5 runs), 420-427 (GPUs on full ECBDL and SOREL, 1 run), 460/461 (--simple
# SOREL, 1 run), 468/469 (--simple chess_1b, 1 run). 418 (--serial full ECBDL,
# 15-18 min a run, 3 runs) is not needed.
scancel 382 384 389 391 396 398 403 404 405 407 408 409 410 411 412 413 414 415 416 418 419 \
  428 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 \
  451 452 453 454 455 456 457 458 459 462 463 464 465 466 467

# Into -new: 10 runs minus the 3 there are (7), or the 5 runs of a slower case minus 3 (2).
for d in airline higgs susy ecbdl_15m mnist8m; do
  $S -m 7 -e -n 0 -d $d -p $P -o $(out $d new) {cuda,gpuonly}-ampere-$d
done
$S -m 7 -e -n 0 -d mnist8m -p $P -o $(out mnist8m new) {cuda,gpuonly}-hopper-mnist8m
$S -m 7 -e -n 0 -d ecbdl_15m -p $P -o $(out ecbdl_15m new) gpuonly-hopper-ecbdl_15m
$S -m 7 -e -d mnist8m -p $P -o $(out mnist8m new) {cpu,socket}-epyc-mnist8m
$S -m 2 -e -d mnist8m -p $P -o $(out mnist8m new) serial-epyc-mnist8m
$S -m 10 -e -d ecbdl_15m -p $P -o $(out ecbdl_15m new) socket-epyc-ecbdl_15m
$S -m 2 -e -d ecbdl_15m -p $P -o $(out ecbdl_15m new) serial-epyc-ecbdl_15m
$S -m 2 -e -d ecbdl -p $P -o $(out ecbdl new) cpu-epyc-ecbdl
$S -m 7 -e -d sorel -p $P -o $(out sorel new) cpu-epyc-sorel
for d in chess_1m chess_10m chess_100m chess_1b; do
  $S -m 7 -n 0 -d $d -p $C -o $(out $d new) {cuda,gpuonly}-{hopper,ampere}-$d
  $S -m 7 -d $d -p $C -o $(out $d new) {cpu,socket,serial}-epyc-$d
done

# Into -top: the started jobs' 5 runs plus 3 there were (2 more), socket-epyc 5 more.
for d in airline higgs susy; do
  $S -m 2 -e -n 0 -d $d -p $P -o $(out $d top) {cuda,gpuonly}-hopper-$d
  $S -m 2 -e -d $d -p $P -o $(out $d top) {cpu,serial}-epyc-$d
  $S -m 5 -e -d $d -p $P -o $(out $d top) socket-epyc-$d
done
$S -m 2 -e -n 0 -d ecbdl_15m -p $P -o $(out ecbdl_15m top) cuda-hopper-ecbdl_15m
$S -m 2 -e -d ecbdl_15m -p $P -o $(out ecbdl_15m top) cpu-epyc-ecbdl_15m

# --simple: one job per protocol, straight to job.sh (into -new).
simple() {  # dataset runs limit timeout accuracy protocols...
  local d=$1 runs=$2 limit=$3 timeout=$4 accuracy=$5 dir n="simple-epyc-$1"; shift 5
  dir=$(out $d new); mkdir -p bench/results/$dir/$n
  for p in "$@"; do
    echo "$dir/$n ${p%%_*}: job $(sbatch --parsable -J dt-$n-${p%%_*} -p epyc --exclusive -t $limit \
      --export=ALL,ACCURACY=$accuracy -o bench/results/$dir/$n/run-${p%%_*}.log \
      bench/cluster/job.sh $dir $n $d tree_simple $runs $p 1 $timeout)"
  done
}
simple higgs 2 03:00:00 7200 --test-accuracy cart_depth10 c45_depth10_unpruned
simple susy 7 03:00:00 3600 --test-accuracy cart_depth10 c45_depth10_unpruned
for d in chess_1m chess_10m chess_100m; do
  simple $d 10 06:00:00 7200 --train-accuracy-only cart_full_nodepth c45_unpruned
done

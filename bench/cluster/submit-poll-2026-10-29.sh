#!/usr/bin/env bash
# Every parallel ./tree_cpu run again (2026-10-29 night), on ~/dt/tree_cpu3:
# ~/dt-final + waitNonZero() in src/build/cpu_builder.cpp for the two waits of
# the parallel partition (lefts[] in partitionInBlocks, done[] in the column
# partition): 2,048 pauses, then sched_yield for up to 2 ms, then 50 us sleeps
# between looks; release stores, no atomic::wait/notify.
#   - The spin fix of 2026-10-28 slept (atomic::wait) right after its pauses,
#     and waking made dell's chess runs about 2x slower than spinning without
#     end (dell2 chess_100m 2.8 s vs 1.25, chess_10m 0.31 vs 0.18).
#   - libstdc++ 13's notify can miss a waiter after a release store (about 1 in
#     40 big --parallel runs hung on dell, exp/perf-d10's ece66ab). Its fix,
#     seq_cst stores, waits for the partition's streaming stores: ECBDL 15M
#     C4.5 8.41 s vs 7.80 (~/dt-final3/ab-job.sh). submit-yield-2026-10-29.sh
#     (jobs 759-762) had run on that build (tree_cpu3): its runs were dropped
#     (~/dt/bench/results-old-build/par-seqcst on mordor3). Polling needs no
#     notify, so nothing to miss. (The 8% was in fact code alignment, below.)
#   - Built with -falign-functions=64 -falign-loops=32 (Makefile): with the
#     default alignment the change moved C4.5's sweep to another offset, and
#     epyc's ECBDL 15M C4.5 was 8% slower, the seq_cst build's "cost" too
#     (~/dt-final5/job.sh: old 7.80 s, aligned 7.99; mnist8m C4.5 5% faster,
#     chess_100m 7%; dell within 3% of old everywhere).
#   - Same trees as ./tree_cpu (~/dt-final4/job.sh, ~/dt-final5). Only the
#     parallel backend has those waits: --serial and --simple stay as they are.
# Parallel on both sockets and on one, dell and epyc, every dataset, and
# dell2's 512 GB chess runs. Runs a case: 10, full ECBDL 3, SOREL once (about
# 35 minutes). Into cluster-<x>-par2/ (a new folder: a compute node may still
# see a moved folder under its old name), moved over cluster-<x>-fix/'s when
# done. Nodes: epyc1 the short runs, epyc2 full ECBDL and SOREL, dell1 all its
# runs after its --serial batch and before its --simple ones, dell2 its chess
# runs. Run on the login node, from the repository root.
set -euo pipefail
cd "$(dirname "$0")/../.."
S=bench/cluster/submit.sh
P=cart_depth10,c45_depth10_unpruned
C=cart_full_nodepth,c45_unpruned
PLANS=bench/cluster/plans
[ -x tree_cpu3 ] || { echo "tree_cpu3 missing" >&2; exit 1; }
out() { case $1 in ecbdl*) echo cluster-ecbdl-par2 ;; chess*) echo cluster-chess-par2 ;; *) echo cluster-$1-par2 ;; esac; }
protocols() { case $1 in chess*) echo $C ;; *) echo $P ;; esac; }
accuracy() { case $1 in chess*) ;; *) printf %s -e ;; esac; }
plan() {  # group runs dataset name...
  local group=$1 runs=$2 d=$3; shift 3
  $S -P $PLANS/$group.txt -m $runs $(accuracy $d) -d $d -p $(protocols $d) -o $(out $d) "$@" > /dev/null
}
rm -f $PLANS/*-par2.txt

for d in airline higgs susy ecbdl_15m mnist8m chess_1m chess_10m chess_100m chess_1b; do
  plan epyc1-par2 10 $d cpu-epyc-$d socket-epyc-$d
done
plan epyc2-par2 1 sorel cpu-epyc-sorel  # first: its data check died once after ECBDL (killed, cause unknown)
plan epyc2-par2 3 ecbdl cpu-epyc-ecbdl socket-epyc-ecbdl
for d in chess_1m chess_10m chess_100m chess_1b airline higgs susy ecbdl_15m mnist8m; do
  plan dell1-par2 10 $d cpu-dell-$d socket-dell-$d
done
plan dell1-par2 3 ecbdl cpu-dell-ecbdl socket-dell-ecbdl
plan dell1-par2 1 sorel cpu-dell-sorel socket-dell-sorel
for d in chess_1m chess_10m chess_100m chess_1b; do plan dell2-par2 10 $d cpu-dell512-$d; done

export TREE_BIN=$PWD/tree_cpu3 TREE_GIT_DESCRIBE="67495f9-dirty-poll"
batch() {  # group partition node [dependency]
  sbatch --parsable -J dt-batch-$1 -p $2 -w $3 --exclusive -t 12:00:00 --export=ALL \
    ${4:+--dependency=$4} -o $PLANS/$1.log bench/cluster/batch.sh $PLANS/$1.txt
}
serial=$(squeue -a -h -u "$USER" -n dt-batch-dell1-serial -o %i)
simple=$(squeue -a -h -u "$USER" -n dt-batch-dell1-simple -o %i)
epyc2=$(squeue -a -h -u "$USER" -n dt-batch-epyc2-simple -o %i)
e1=$(batch epyc1-par2 epyc epyc1)
e2=$(batch epyc2-par2 epyc epyc2 ${epyc2:+afterany:$epyc2})
d1=$(batch dell1-par2 dell dell1 ${serial:+afterany:$serial})
[ -z "$simple" ] || scontrol update JobId=$simple Dependency=afterany:$d1
d2=$(batch dell2-par2 dell-512GB dell2)
echo "epyc1 $e1, epyc2 $e2, dell1 $d1 (then --simple $simple), dell2 $d2"
wc -l $PLANS/*-par2.txt
squeue -a -u "$USER" -o "%.6i %.26j %.8T %.8N %.20E"

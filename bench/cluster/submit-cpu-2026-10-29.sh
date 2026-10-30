#!/usr/bin/env bash
# Superseded by submit-main-2026-10-29.sh, kept as the record of jobs 797-799:
# 797 and 799 were cancelled (they ran the binary from before main's presort);
# 798 finished and its --simple runs are kept (--simple's code did not change).
# CPU-only reruns of the benchmarks around the test-set ones (2026-10-29
# evening), ./tree only (no other libraries, no CUDA: another session is
# changing the CUDA code), every tree at depth <= 10 without pruning
# (cart_depth10, c45_depth10_unpruned):
#   - cluster-cpu-scaling: HIGGS and MNIST8M, 1, 2, 4, then every 4 threads up
#     to 44 on epyc and dell1; the 48-thread point is the test-set benchmark's
#     cpu-<node>-<data> run (the same binary and command), merged in after.
#     Runs per point: 10, except MNIST8M at 1 and 2 threads (4-6 minutes a
#     run): 3.
#   - cluster-profiling: the parallel backend profiled (./tree --profile) on
#     epyc and, new, dell1; chess 10M-2000M 30 runs a case, profiled and plain;
#     HIGGS, airline and ECBDL 15M 10 runs profiled, their plain runs are the
#     test-set benchmarks' cpu-<node>-<data> runs.
#   - cluster-cpu-vs-others: ./tree's bars for Covertype, chess 10M and chess
#     1000M at depth 10 (--simple, --serial, parallel on one socket and on
#     both); HIGGS, SUSY and ECBDL 15M take theirs from the test-set
#     benchmarks. Chess 30 runs a case (chess_1b --serial 3, --simple 1),
#     Covertype 10.
# Builds as in the test-set reruns: parallel on ~/dt/tree_cpu3 (8bdf641's CPU
# code, git describe 67495f9-dirty-poll), --serial and --simple on ~/dt/tree_cpu
# (67495f9-dirty); every scaling and profiling run on tree_cpu3, so that a
# curve or a profile is one binary. Nodes: epyc1 and epyc2 (the same hardware)
# share the epyc work, short runs first, about 5.5 h each; dell1 (1 TB, the
# dell of every other benchmark) runs its part after job 645, short runs first.
# Into new folders (cluster-<x>-new*), moved into place when done. Run on the
# login node, from the repository root.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=cart_depth10,c45_depth10_unpruned
PLANS=bench/cluster/plans
PAR="TREE_BIN=$PWD/tree_cpu3,TREE_GIT_DESCRIBE=67495f9-dirty-poll"
ONE="TREE_BIN=$PWD/tree_cpu,TREE_GIT_DESCRIBE=67495f9-dirty"
VS=",ACCURACY=--test-accuracy,RUN_EXTRA=--hash-trees"
THREADS="4,$(seq -s, 8 4 44)"
for f in tree_cpu tree_cpu3; do [ -x $f ] || { echo "$f missing" >&2; exit 1; }; done
rm -f $PLANS/{epyc1,epyc2,dell1}-cpu.txt

line() {  # plan dir name dataset impls runs threads env
  echo "_ $2 $3 $4 $5 $6 $P $7 3600 $8" >> $PLANS/$1.txt
}
vs() {  # plan dataset runs(parallel) runs(serial) runs(simple) [backends]
  local plan=$1 d=$2 rp=$3 rs=$4 rm=$5 o=cluster-cpu-vs-others-new
  for b in ${6:-cpu socket serial simple}; do
    case $b in
      cpu) line $plan $o cpu-epyc-$d $d tree $rp all "$PAR$VS" ;;
      socket) line $plan $o socket-epyc-$d $d tree $rp all "$PAR$VS,SOCKET=auto" ;;
      serial) line $plan $o serial-epyc-$d $d tree $rs 1 "$ONE$VS" ;;
      simple) line $plan $o simple-epyc-$d $d tree_simple $rm 1 "$ONE$VS" ;;
    esac
  done
}
profile() {  # plan node runs plain(yes/no) datasets...
  local plan=$1 node=$2 runs=$3 plain=$4 d; shift 4
  for d in "$@"; do
    line $plan cluster-profiling-new cpu-$node-$d $d tree $runs all "$PAR,RUN_EXTRA=--profile"
    [ $plain = no ] || line $plan cluster-profiling-new/plain cpu-$node-$d $d tree $runs all "$PAR"
  done
}

# epyc1: HIGGS scaling, the small vs-others sets, then profiling.
line epyc1-cpu cluster-cpu-scaling-new cpu-epyc-higgs higgs tree 10 "1,2,$THREADS" "$PAR"
vs epyc1-cpu covertype 10 10 10
vs epyc1-cpu chess_10m 30 30 30
profile epyc1-cpu epyc 10 no higgs airline ecbdl_15m
profile epyc1-cpu epyc 30 yes chess_10m chess_50m chess_100m chess_500m chess_1b chess_2b

# epyc2: chess 1000M for vs-others, MNIST8M scaling, then chess 1000M --simple.
vs epyc2-cpu chess_1b 30 3 1 "cpu socket serial"
line epyc2-cpu cluster-cpu-scaling-new cpu-epyc-mnist8m mnist8m tree 10 "$THREADS" "$PAR"
line epyc2-cpu cluster-cpu-scaling-new2 cpu-epyc-mnist8m mnist8m tree 3 1,2 "$PAR"
vs epyc2-cpu chess_1b 30 3 1 simple

# dell1, after job 645: HIGGS scaling, profiling up to 500M, MNIST8M scaling,
# then profiling chess 1000M and 2000M.
line dell1-cpu cluster-cpu-scaling-new cpu-dell-higgs higgs tree 10 "1,2,$THREADS" "$PAR"
profile dell1-cpu dell 10 no higgs airline ecbdl_15m
profile dell1-cpu dell 30 yes chess_10m chess_50m chess_100m chess_500m
line dell1-cpu cluster-cpu-scaling-new cpu-dell-mnist8m mnist8m tree 10 "$THREADS" "$PAR"
line dell1-cpu cluster-cpu-scaling-new2 cpu-dell-mnist8m mnist8m tree 3 1,2 "$PAR"
profile dell1-cpu dell 30 yes chess_1b chess_2b

batch() {  # group partition node limit [dependency]
  sbatch --parsable -J dt-batch-$1 -p $2 -w $3 --exclusive -t $4 --export=ALL \
    ${5:+--dependency=$5} -o $PLANS/$1.log bench/cluster/batch.sh $PLANS/$1.txt
}
before=$(squeue -a -h -u "$USER" -n dt-batch-dell1-simple-ecbdl -o %i)
e1=$(batch epyc1-cpu epyc epyc1 12:00:00)
e2=$(batch epyc2-cpu epyc epyc2 12:00:00)
d1=$(batch dell1-cpu dell-long dell1 1-00:00:00 ${before:+afterany:$before})
echo "epyc1 $e1, epyc2 $e2, dell1 $d1 (after ${before:-nothing})"
wc -l $PLANS/*-cpu.txt
squeue -a -u "$USER" -o "%.6i %.26j %.8T %.8N %.20E"

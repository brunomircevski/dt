#!/usr/bin/env bash
# Every parallel ./tree_cpu run again (2026-10-29 night), on the build that
# yields before it sleeps (~/dt/tree_cpu2: ~/dt-final + partitionInBlocks
# waiting with 2,048 pauses, then sched_yield for up to 2 ms, then the futex
# wait). The spin fix of 2026-10-28 slept right after its pauses, and waking
# made dell's chess runs about 2x slower than spinning without end (chess_100m
# 2.8 s vs 1.25, chess_10m 0.31 vs 0.18 on dell2, ~/dt-spin/hang-job.sh); the
# yield keeps epyc free of the stall the spin fix was for. It also stores the
# values it waits on seq_cst (from exp/perf-d10's ece66ab): after a release
# store libstdc++ 13 could miss the waiter, and about 1 in 40 big runs hung on
# dell. Same trees (~/dt-final2/validate-job.sh). Only the parallel backend
# has those waits: --serial and --simple stay as they are.
#   - parallel on both sockets and on one, dell and epyc, every dataset, and
#     dell2's 512 GB chess runs. Runs a case: 10, full ECBDL 3, SOREL once
#     (a run of about 35 minutes).
#   - into cluster-<x>-par/ (a new folder: a compute node may still see a moved
#     folder under its old name), moved over cluster-<x>-fix/'s when done.
# Nodes: epyc1 the short runs, epyc2 full ECBDL and SOREL (after its --simple
# batch), dell1 all its runs after its --serial batch and before its --simple
# ones (job 644 waits for them), dell2 its chess runs. Run on the login node,
# from the repository root, after ~/dt/tree_cpu2 is installed. As run:
# squeue lacked -a (it hides the jobs then), so the dependencies were set by
# hand: 760 after 641, 761 after 643, 644 after 761. Jobs 759–762.
# Its runs were dropped: the seq_cst stores cost up to 8% on many columns
# (ECBDL 15M C4.5); submit-poll-2026-10-29.sh runs them again.
set -euo pipefail
cd "$(dirname "$0")/../.."
S=bench/cluster/submit.sh
P=cart_depth10,c45_depth10_unpruned
C=cart_full_nodepth,c45_unpruned
PLANS=bench/cluster/plans
[ -x tree_cpu2 ] || { echo "tree_cpu2 missing" >&2; exit 1; }
out() { case $1 in ecbdl*) echo cluster-ecbdl-par ;; chess*) echo cluster-chess-par ;; *) echo cluster-$1-par ;; esac; }
protocols() { case $1 in chess*) echo $C ;; *) echo $P ;; esac; }
accuracy() { case $1 in chess*) ;; *) printf %s -e ;; esac; }
plan() {  # group runs dataset name...
  local group=$1 runs=$2 d=$3; shift 3
  $S -P $PLANS/$group.txt -m $runs $(accuracy $d) -d $d -p $(protocols $d) -o $(out $d) "$@" > /dev/null
}
rm -f $PLANS/*-par.txt

for d in airline higgs susy ecbdl_15m mnist8m chess_1m chess_10m chess_100m chess_1b; do
  plan epyc1-par 10 $d cpu-epyc-$d socket-epyc-$d
done
plan epyc2-par 3 ecbdl cpu-epyc-ecbdl socket-epyc-ecbdl
plan epyc2-par 1 sorel cpu-epyc-sorel
for d in chess_1m chess_10m chess_100m chess_1b airline higgs susy ecbdl_15m mnist8m; do
  plan dell1-par 10 $d cpu-dell-$d socket-dell-$d
done
plan dell1-par 3 ecbdl cpu-dell-ecbdl socket-dell-ecbdl
plan dell1-par 1 sorel cpu-dell-sorel socket-dell-sorel
for d in chess_1m chess_10m chess_100m chess_1b; do plan dell2-par 10 $d cpu-dell512-$d; done

export TREE_BIN=$PWD/tree_cpu2 TREE_GIT_DESCRIBE="67495f9-dirty-yield"
batch() {  # group partition node [dependency]
  sbatch --parsable -J dt-batch-$1 -p $2 -w $3 --exclusive -t 12:00:00 --export=ALL \
    ${4:+--dependency=$4} -o $PLANS/$1.log bench/cluster/batch.sh $PLANS/$1.txt
}
serial=$(squeue -a -h -u "$USER" -n dt-batch-dell1-serial -o %i)
simple=$(squeue -a -h -u "$USER" -n dt-batch-dell1-simple -o %i)
epyc2=$(squeue -a -h -u "$USER" -n dt-batch-epyc2-simple -o %i)
e1=$(batch epyc1-par epyc epyc1)
e2=$(batch epyc2-par epyc epyc2 ${epyc2:+afterany:$epyc2})
d1=$(batch dell1-par dell dell1 ${serial:+afterany:$serial})
[ -z "$simple" ] || scontrol update JobId=$simple Dependency=afterany:$d1
d2=$(batch dell2-par dell-512GB dell2)
echo "epyc1 $e1, epyc2 $e2, dell1 $d1 (then --simple $simple), dell2 $d2"
wc -l $PLANS/*-par.txt
squeue -a -u "$USER" -o "%.6i %.26j %.8T %.8N %.20E"
